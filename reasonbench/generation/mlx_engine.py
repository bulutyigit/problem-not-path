"""Instrumented text generation on Apple Silicon through MLX-VLM."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reasonbench.exceptions import InstrumentationError
from reasonbench.generation.engine import (
    ContinuationResult,
    GenerationResult,
    _infer_finish_reason,
)
from reasonbench.generation.mlx_modeling import MLXModelBundle
from reasonbench.generation.segments import segment_generated_tokens
from reasonbench.instrumentation.mlx_recorder import MLXTokenSignalRecorder
from reasonbench.verification.extract import split_reasoning_and_answer


def _prepare_prompt(bundle: MLXModelBundle, problem: str) -> tuple[Any, Any]:
    import mlx.core as mx

    plan = bundle.adapter.build_prompt(problem, bundle.model_config.mode)
    template_source = bundle.processor or bundle.tokenizer
    try:
        prompt = template_source.apply_chat_template(
            plan.messages,
            tokenize=False,
            add_generation_prompt=True,
            **plan.template_kwargs,
        )
    except TypeError as exc:
        raise InstrumentationError(
            f"MLX chat template rejected the registered prompt options: {exc}"
        ) from exc
    token_ids = bundle.tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > bundle.model_config.max_prompt_tokens:
        raise InstrumentationError(
            f"Prompt has {len(token_ids)} tokens, exceeding max_prompt_tokens="
            f"{bundle.model_config.max_prompt_tokens}"
        )
    input_ids = mx.array([token_ids])
    return input_ids, mx.ones_like(input_ids)


def _language_model_forward(
    bundle: MLXModelBundle,
    input_ids: Any,
    cache: list[Any],
    *,
    inputs_embeds: Any | None = None,
    prompt_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Return raw logits and the exact final-normalized head input."""

    language_model = bundle.model.language_model
    kwargs = dict(prompt_kwargs or {})
    if bundle.model_config.adapter == "ministral3":
        hidden = language_model.model(
            input_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )
        language_config = getattr(
            language_model,
            "args",
            getattr(language_model, "config", None),
        )
        if language_config is None:
            raise InstrumentationError("Ministral MLX language config is unavailable")
        if language_config.tie_word_embeddings:
            logits = language_model.model.embed_tokens.as_linear(hidden)
        else:
            logits = language_model.lm_head(hidden)
        return logits, hidden

    if type(language_model).__module__.endswith(".qwen3.language"):
        # mlx_vlm's text-only Qwen3 wrapper swallows return_hidden and
        # reports hidden_states=None; its inner model returns the
        # final-normalized head input directly (same shape contract as the
        # Ministral branch, different embedding kwarg name).
        hidden = language_model.model(
            input_ids,
            cache=cache,
            input_embeddings=inputs_embeds,
        )
        language_config = getattr(
            language_model,
            "args",
            getattr(language_model, "config", None),
        )
        if language_config is not None and getattr(language_config, "tie_word_embeddings", False):
            logits = language_model.model.embed_tokens.as_linear(hidden)
        else:
            logits = language_model.lm_head(hidden)
        return logits, hidden

    outputs = language_model(
        input_ids,
        inputs_embeds=inputs_embeds,
        cache=cache,
        return_hidden=True,
        **kwargs,
    )
    if not outputs.hidden_states:
        raise InstrumentationError(
            f"{bundle.model_config.model_id} did not return final hidden states"
        )
    hidden = outputs.hidden_states[-1]
    if bundle.model_config.adapter == "gemma4":
        # Gemma 4 exposes the final decoder output before its last RMSNorm;
        # the CUDA recorder observes the normalized language-head input.
        hidden = language_model.model.norm(hidden)
    return outputs.logits, hidden


def _initial_forward(bundle: MLXModelBundle, input_ids: Any, mask: Any, cache: list[Any]):
    features = bundle.model.get_input_embeddings(input_ids, None, mask=mask)
    prompt_kwargs = {
        key: value
        for key, value in features.to_dict().items()
        if key != "inputs_embeds" and value is not None
    }
    return _language_model_forward(
        bundle,
        input_ids,
        cache,
        inputs_embeds=features.inputs_embeds,
        prompt_kwargs=prompt_kwargs,
    )


def _all_eos_ids(bundle: MLXModelBundle) -> set[int]:
    values: list[Any] = [getattr(bundle.tokenizer, "eos_token_id", None)]
    values.append(getattr(bundle.model.config, "eos_token_id", None))
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        else:
            result.add(int(value))
    return result


def _make_generation_cache(bundle: MLXModelBundle) -> list[Any]:
    cache_factory = getattr(bundle.model, "make_cache", None)
    if cache_factory is None:
        cache_factory = getattr(bundle.model.language_model, "make_cache", None)
    if cache_factory is None:
        raise InstrumentationError(
            f"{bundle.model_config.model_id} exposes no MLX generation-cache factory"
        )
    return cache_factory()


def _reset_generation_positions(bundle: MLXModelBundle) -> None:
    if hasattr(bundle.model.language_model, "_position_ids"):
        bundle.model.language_model._position_ids = None
    if hasattr(bundle.model.language_model, "_rope_deltas"):
        bundle.model.language_model._rope_deltas = None


def _sample_from_generated_prefix(
    bundle: MLXModelBundle,
    problem: str,
    generated_prefix: list[int],
    *,
    max_new_tokens: int,
    stop_sequences: tuple[tuple[int, ...], ...] = (),
) -> tuple[list[int], str]:
    """Replay an exact prefix and sample a lightweight MLX continuation."""

    import mlx.core as mx
    from mlx_vlm.sample_utils import make_logits_processors, make_sampler

    if max_new_tokens <= 0:
        raise InstrumentationError("MLX continuation max_new_tokens must be positive")
    config = bundle.model_config
    prompt_ids, _ = _prepare_prompt(bundle, problem)
    prefix = mx.array([generated_prefix], dtype=prompt_ids.dtype)
    combined = mx.concatenate([prompt_ids, prefix], axis=-1)
    mask = mx.ones_like(combined)
    _reset_generation_positions(bundle)
    cache = _make_generation_cache(bundle)
    logits, hidden = _initial_forward(bundle, combined, mask, cache)
    del hidden

    temperature = config.sampling.temperature if config.sampling.do_sample else 0.0
    sampler = make_sampler(
        temp=temperature,
        top_p=config.sampling.top_p,
        min_p=config.sampling.min_p,
        top_k=config.sampling.top_k,
    )
    processors = make_logits_processors(
        repetition_penalty=(
            None
            if abs(config.sampling.repetition_penalty - 1.0) < 1e-12
            else config.sampling.repetition_penalty
        ),
        presence_penalty=config.sampling.presence_penalty,
    )
    history = mx.array(generated_prefix, dtype=prompt_ids.dtype) if processors else None
    eos_ids = _all_eos_ids(bundle)
    generated: list[int] = []
    finish_reason = "max_new_tokens"
    for token_index in range(max_new_tokens):
        step_logits = logits[:, -1, :].astype(mx.float32)
        for processor in processors:
            if history is None:
                raise InstrumentationError("MLX continuation history is unavailable")
            step_logits = processor(history, step_logits)
        log_probabilities = step_logits - mx.logsumexp(step_logits, axis=-1, keepdims=True)
        sampled = sampler(log_probabilities)
        mx.eval(sampled, log_probabilities)
        token_id = int(sampled.item())
        generated.append(token_id)
        if history is not None:
            history = mx.concatenate([history, sampled.reshape(-1)])
            mx.eval(history)
        if token_id in eos_ids:
            finish_reason = "eos"
            break
        if any(
            len(generated) >= len(sequence) and tuple(generated[-len(sequence) :]) == sequence
            for sequence in stop_sequences
        ):
            finish_reason = "reasoning_close"
            break
        next_ids = sampled.reshape(1, 1).astype(prompt_ids.dtype)
        logits, hidden = _language_model_forward(bundle, next_ids, cache)
        del hidden
        if token_index % 256 == 0:
            mx.clear_cache()
    return generated, finish_reason


class MLXInstrumentedGenerator:
    """Generate one fully instrumented reasoning trajectory on a Metal GPU."""

    def __init__(self, bundle: MLXModelBundle) -> None:
        self.bundle = bundle
        if bundle.model_config.reasoning_budget is not None:
            raise InstrumentationError(
                "Assigned-budget generation is not implemented for MLX; Phase 4B uses "
                "native stopping under max_new_tokens."
            )

    def generate(
        self,
        problem: str,
        on_token: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        import mlx.core as mx
        from mlx_vlm.sample_utils import make_logits_processors, make_sampler

        config = self.bundle.model_config
        input_ids, mask = _prepare_prompt(self.bundle, problem)
        if hasattr(self.bundle.model.language_model, "_position_ids"):
            self.bundle.model.language_model._position_ids = None
        if hasattr(self.bundle.model.language_model, "_rope_deltas"):
            self.bundle.model.language_model._rope_deltas = None
        cache_factory = getattr(self.bundle.model, "make_cache", None)
        if cache_factory is None:
            cache_factory = getattr(self.bundle.model.language_model, "make_cache", None)
        if cache_factory is None:
            # Text-only checkpoints wrapped by mlx_vlm (e.g. Qwen3-8B) carry no
            # make_cache method; mlx_lm builds their KV cache externally.
            try:
                from mlx_lm.models.cache import make_prompt_cache
            except ImportError:
                make_prompt_cache = None
            if make_prompt_cache is not None:
                language_model = self.bundle.model.language_model
                cache_factory = lambda: make_prompt_cache(language_model)  # noqa: E731
        if cache_factory is None:
            raise InstrumentationError(f"{config.model_id} exposes no MLX generation-cache factory")
        cache = cache_factory()
        recorder = MLXTokenSignalRecorder(
            self.bundle.tokenizer,
            hidden_state_stride=config.hidden_state_stride,
            capture_hidden_states=config.capture_hidden_states,
        )
        temperature = config.sampling.temperature if config.sampling.do_sample else 0.0
        sampler = make_sampler(
            temp=temperature,
            top_p=config.sampling.top_p,
            min_p=config.sampling.min_p,
            top_k=config.sampling.top_k,
        )
        processors = make_logits_processors(
            repetition_penalty=(
                None
                if abs(config.sampling.repetition_penalty - 1.0) < 1e-12
                else config.sampling.repetition_penalty
            ),
            presence_penalty=config.sampling.presence_penalty,
        )
        generated: list[int] = []
        history = mx.array([], dtype=input_ids.dtype) if processors else None
        eos_ids = _all_eos_ids(self.bundle)

        logits, hidden = _initial_forward(self.bundle, input_ids, mask, cache)
        finish_reason = "max_new_tokens"
        for token_index in range(config.max_new_tokens):
            step_logits = logits[:, -1, :].astype(mx.float32)
            for processor in processors:
                if history is None:
                    raise InstrumentationError("MLX logits processor history is unavailable")
                step_logits = processor(history, step_logits)
            log_probabilities = step_logits - mx.logsumexp(step_logits, axis=-1, keepdims=True)
            sampled = sampler(log_probabilities)
            mx.eval(sampled, log_probabilities, hidden)
            token_id = int(sampled.item())
            recorder.finalize_sampled_token(
                token_id,
                log_probabilities.reshape(-1),
                hidden,
            )
            generated.append(token_id)
            if history is not None:
                history = mx.concatenate([history, sampled.reshape(-1)])
                mx.eval(history)
            if on_token is not None:
                on_token(token_index + 1)
            if token_id in eos_ids:
                finish_reason = "eos"
                break
            next_ids = sampled.reshape(1, 1).astype(input_ids.dtype)
            logits, hidden = _language_model_forward(
                self.bundle,
                next_ids,
                cache,
            )
            if token_index % 256 == 0:
                mx.clear_cache()

        recorder.validate_alignment(generated)
        segments = segment_generated_tokens(
            generated,
            self.bundle.tokenizer,
            config.mode,
            reasoning_close_markers=self.bundle.adapter.reasoning_close_markers(),
            reasoning_open_markers=self.bundle.adapter.reasoning_open_markers(),
        )
        recorder.set_segments(segments)
        text = self.bundle.tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        reasoning, final_response, boundary = split_reasoning_and_answer(text)
        if finish_reason != "eos":
            finish_reason = _infer_finish_reason(
                generated,
                token_limit=config.max_new_tokens,
                eos_token_id=list(eos_ids),
                limit_reason="max_new_tokens",
            )
        return GenerationResult(
            generated_text=text,
            reasoning_text=reasoning,
            final_response_text=final_response,
            boundary_status=boundary,
            generated_token_ids=generated,
            signals=recorder.records,
            hidden_state_indices=recorder.hidden_state_indices,
            hidden_states=recorder.hidden_states,
            finish_reason=finish_reason,
            inserted_boundary_token_count=0,
            reasoning_boundary_forced=False,
            reasoning_stage_token_count=None,
        )

    def continue_from_prefix_with_reasoning_budget(
        self,
        problem: str,
        generated_prefix_token_ids: list[int],
        *,
        reasoning_continuation_budget: int,
        final_answer_reserve: int,
        max_total_generated_tokens: int | None = None,
    ) -> ContinuationResult:
        """Replay a stored prefix and run the frozen two-stage MLX branch."""

        if reasoning_continuation_budget <= 0 or final_answer_reserve <= 0:
            raise InstrumentationError("Continuation and answer budgets must be positive")
        config = self.bundle.model_config
        if config.reasoning_budget is not None:
            raise InstrumentationError(
                "Breakthrough continuation requires a standard-generation configuration"
            )
        total_budget = max_total_generated_tokens or config.max_new_tokens
        available = total_budget - len(generated_prefix_token_ids)
        if available <= final_answer_reserve:
            raise InstrumentationError("Exact prefix leaves no room for the frozen answer reserve")
        if reasoning_continuation_budget > available - final_answer_reserve:
            raise InstrumentationError(
                "Frozen continuation budget does not fit beside the answer reserve"
            )
        reasoning_limit = reasoning_continuation_budget
        close_sequences = tuple(
            tuple(sequence)
            for marker in self.bundle.adapter.reasoning_close_markers()
            if (sequence := self.bundle.tokenizer.encode(marker, add_special_tokens=False))
        )
        reasoning_ids, _ = _sample_from_generated_prefix(
            self.bundle,
            problem,
            generated_prefix_token_ids,
            max_new_tokens=reasoning_limit,
            stop_sequences=close_sequences,
        )
        already_closed = any(
            len(reasoning_ids) >= len(sequence)
            and tuple(reasoning_ids[-len(sequence) :]) == sequence
            for sequence in close_sequences
        )
        eos_ids = _all_eos_ids(self.bundle)
        if not already_closed and reasoning_ids and reasoning_ids[-1] in eos_ids:
            reasoning_ids = reasoning_ids[:-1]
        inserted_ids: list[int] = []
        if not already_closed:
            inserted_ids = self.bundle.tokenizer.encode(
                self.bundle.adapter.forced_reasoning_close_text(),
                add_special_tokens=False,
            )
        answer_budget = min(
            final_answer_reserve,
            total_budget - len(generated_prefix_token_ids) - len(reasoning_ids) - len(inserted_ids),
        )
        if answer_budget <= 0:
            raise InstrumentationError("Forced close left no tokens for the answer reserve")
        if answer_budget != final_answer_reserve:
            raise InstrumentationError(
                "Frozen total budget cannot preserve the full answer reserve"
            )
        answer_prefix = [
            *generated_prefix_token_ids,
            *reasoning_ids,
            *inserted_ids,
        ]
        answer_ids, answer_finish_reason = _sample_from_generated_prefix(
            self.bundle,
            problem,
            answer_prefix,
            max_new_tokens=answer_budget,
        )
        continuation_ids = [*reasoning_ids, *inserted_ids, *answer_ids]
        combined_generated_ids = [*generated_prefix_token_ids, *continuation_ids]
        text = self.bundle.tokenizer.decode(
            combined_generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        finish_reason = (
            "answer_reserve" if answer_finish_reason == "max_new_tokens" else answer_finish_reason
        )
        return ContinuationResult(
            generated_text=text,
            generated_token_ids=combined_generated_ids,
            continuation_token_ids=continuation_ids,
            finish_reason=finish_reason,
            inserted_boundary_token_count=len(inserted_ids),
            reasoning_boundary_forced=not already_closed,
            reasoning_continuation_token_count=len(reasoning_ids),
        )
