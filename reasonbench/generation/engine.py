"""Instrumented, batch-size-one Transformers generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from reasonbench.config import ModelConfig
from reasonbench.exceptions import InstrumentationError
from reasonbench.generation.modeling import ModelBundle
from reasonbench.generation.segments import segment_generated_tokens
from reasonbench.instrumentation.recorder import (
    RecorderStoppingCriteria,
    TokenSignal,
    TokenSignalRecorder,
)
from reasonbench.verification.extract import split_reasoning_and_answer


@dataclass
class GenerationResult:
    """One complete instrumented model response."""

    generated_text: str
    reasoning_text: str
    final_response_text: str
    boundary_status: str
    generated_token_ids: list[int]
    signals: list[TokenSignal]
    hidden_state_indices: list[int]
    hidden_states: list[Any]
    finish_reason: str
    inserted_boundary_token_count: int
    reasoning_boundary_forced: bool
    reasoning_stage_token_count: int | None


@dataclass
class ContinuationResult:
    """A lightweight continuation from an exact stored generation prefix.

    Continuation branches are used only to label whether a prefix lies in a
    stable success basin. They intentionally do not duplicate the expensive
    token-signal and hidden-state instrumentation already stored for the base
    rollout.
    """

    generated_text: str
    generated_token_ids: list[int]
    continuation_token_ids: list[int]
    finish_reason: str
    inserted_boundary_token_count: int = 0
    reasoning_boundary_forced: bool = False
    reasoning_continuation_token_count: int | None = None


def _prepare_inputs(bundle: ModelBundle, problem: str, mode: str) -> dict[str, Any]:
    plan = bundle.adapter.build_prompt(problem, mode)
    tokenizer = bundle.tokenizer
    template_source = bundle.processor or tokenizer
    template_arguments = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        **plan.template_kwargs,
    }
    try:
        encoded = template_source.apply_chat_template(plan.messages, **template_arguments)
    except TypeError as exc:
        if "return_dict" not in str(exc):
            raise
        template_arguments.pop("return_dict")
        encoded = template_source.apply_chat_template(plan.messages, **template_arguments)
    if hasattr(encoded, "to"):
        encoded = encoded.to(bundle.model.device)
    if isinstance(encoded, Mapping):
        encoded = dict(encoded)
    else:
        encoded = {"input_ids": encoded.to(bundle.model.device)}
    input_ids = encoded.get("input_ids")
    if input_ids is None or isinstance(input_ids, Mapping) or not hasattr(input_ids, "shape"):
        raise InstrumentationError(
            "The chat template did not return a tensor-valued input_ids field; "
            f"received {type(input_ids).__name__}"
        )
    if "attention_mask" not in encoded:
        import torch

        encoded["attention_mask"] = torch.ones_like(input_ids)
    prompt_length = int(input_ids.shape[-1])
    if prompt_length > bundle.model_config.max_prompt_tokens:
        raise InstrumentationError(
            f"Prompt has {prompt_length} tokens, exceeding max_prompt_tokens="
            f"{bundle.model_config.max_prompt_tokens}"
        )
    return encoded


def _token_is_eos(token_id: int, eos_token_id: Any) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (tuple, list, set)):
        return token_id in eos_token_id
    return token_id == eos_token_id


def _infer_finish_reason(
    token_ids: list[int],
    *,
    token_limit: int,
    eos_token_id: Any,
    limit_reason: str,
) -> str:
    """Distinguish an early Transformers stop from exhausting the hard limit."""

    if token_ids and _token_is_eos(token_ids[-1], eos_token_id):
        return "eos"
    # Transformers can use model-generation EOS IDs that differ from the tokenizer's
    # primary eos_token_id. With no custom early-stop criterion in these stages,
    # returning before the requested limit is still an EOS completion.
    if len(token_ids) < token_limit:
        return "eos"
    return limit_reason


def _generation_kwargs(config: ModelConfig) -> dict[str, Any]:
    tokenizer_generation = {
        "do_sample": config.sampling.do_sample,
        "temperature": config.sampling.temperature,
        "top_p": config.sampling.top_p,
        "top_k": config.sampling.top_k,
        "repetition_penalty": config.sampling.repetition_penalty,
        "use_cache": True,
        "return_dict_in_generate": False,
    }
    if config.sampling.min_p > 0:
        tokenizer_generation["min_p"] = config.sampling.min_p
    return tokenizer_generation


class InstrumentedGenerator:
    """Generate responses while computing compact token-level signals."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def _run_generate(
        self,
        model_inputs: dict[str, Any],
        max_new_tokens: int,
        stop_on_reasoning_close: bool,
        on_token: Callable[[int], None] | None = None,
    ) -> tuple[list[int], TokenSignalRecorder, Any]:
        import torch
        from transformers import StoppingCriteriaList

        config = self.bundle.model_config
        recorder = TokenSignalRecorder(
            tokenizer=self.bundle.tokenizer,
            hidden_state_stride=config.hidden_state_stride,
            capture_hidden_states=config.capture_hidden_states,
        )
        recorder.attach(self.bundle.model)
        close_sequences = [
            self.bundle.tokenizer.encode(marker, add_special_tokens=False)
            for marker in self.bundle.adapter.reasoning_close_markers()
        ]
        criteria = RecorderStoppingCriteria(
            recorder,
            stop_sequences=close_sequences if stop_on_reasoning_close else [],
            on_token=on_token,
        )
        prompt_length = int(model_inputs["input_ids"].shape[-1])
        try:
            with torch.inference_mode():
                sequences = self.bundle.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=StoppingCriteriaList([criteria]),
                    pad_token_id=(
                        self.bundle.tokenizer.pad_token_id
                        if self.bundle.tokenizer.pad_token_id is not None
                        else self.bundle.tokenizer.eos_token_id
                    ),
                    **_generation_kwargs(config),
                )
        finally:
            recorder.detach()
        generated = sequences[0, prompt_length:].tolist()
        recorder.validate_alignment(generated)
        return generated, recorder, sequences

    def _standard_generation(
        self,
        problem: str,
        on_token: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        config = self.bundle.model_config
        inputs = _prepare_inputs(self.bundle, problem, config.mode)
        token_ids, recorder, _ = self._run_generate(
            inputs,
            max_new_tokens=config.max_new_tokens,
            stop_on_reasoning_close=False,
            on_token=on_token,
        )
        segments = segment_generated_tokens(
            token_ids,
            self.bundle.tokenizer,
            config.mode,
            reasoning_close_markers=self.bundle.adapter.reasoning_close_markers(),
            reasoning_open_markers=self.bundle.adapter.reasoning_open_markers(),
        )
        recorder.set_segments(segments)
        text = self.bundle.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        reasoning, final_response, boundary = split_reasoning_and_answer(text)
        eos_id = self.bundle.tokenizer.eos_token_id
        finish_reason = _infer_finish_reason(
            token_ids,
            token_limit=config.max_new_tokens,
            eos_token_id=eos_id,
            limit_reason="max_new_tokens",
        )
        return GenerationResult(
            generated_text=text,
            reasoning_text=reasoning,
            final_response_text=final_response,
            boundary_status=boundary,
            generated_token_ids=token_ids,
            signals=recorder.records,
            hidden_state_indices=recorder.hidden_state_indices,
            hidden_states=recorder.hidden_states,
            finish_reason=finish_reason,
            inserted_boundary_token_count=0,
            reasoning_boundary_forced=False,
            reasoning_stage_token_count=None,
        )

    def _budgeted_generation(
        self,
        problem: str,
        on_token: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        import torch

        config = self.bundle.model_config
        if config.reasoning_budget is None:
            raise InstrumentationError("Budgeted generation requires reasoning_budget")
        if not self.bundle.adapter.supports_assigned_reasoning_budget:
            raise InstrumentationError(
                f"{self.bundle.adapter.key} does not support assigned reasoning budgets"
            )
        inputs = _prepare_inputs(self.bundle, problem, "reasoning")
        reasoning_ids, reasoning_recorder, sequences = self._run_generate(
            inputs,
            max_new_tokens=config.reasoning_budget,
            stop_on_reasoning_close=True,
            on_token=on_token,
        )
        close_sequences = [
            self.bundle.tokenizer.encode(marker, add_special_tokens=False)
            for marker in self.bundle.adapter.reasoning_close_markers()
        ]
        already_closed = any(
            close_ids
            and len(reasoning_ids) >= len(close_ids)
            and reasoning_ids[-len(close_ids) :] == close_ids
            for close_ids in close_sequences
        )
        eos_id = self.bundle.tokenizer.eos_token_id
        if not already_closed and reasoning_ids and _token_is_eos(reasoning_ids[-1], eos_id):
            removed_index = len(reasoning_recorder.records) - 1
            reasoning_ids = reasoning_ids[:-1]
            sequences = sequences[:, :-1]
            reasoning_recorder.records.pop()
            if (
                reasoning_recorder.hidden_state_indices
                and reasoning_recorder.hidden_state_indices[-1] == removed_index
            ):
                reasoning_recorder.hidden_state_indices.pop()
                reasoning_recorder.hidden_states.pop()
        inserted_ids: list[int] = []
        if not already_closed:
            inserted_ids = self.bundle.tokenizer.encode(
                self.bundle.adapter.forced_reasoning_close_text(),
                add_special_tokens=False,
            )
        full_prefix = torch.cat(
            [
                sequences,
                torch.tensor([inserted_ids], dtype=sequences.dtype, device=sequences.device),
            ],
            dim=-1,
        )
        answer_inputs = {
            "input_ids": full_prefix,
            "attention_mask": torch.ones_like(full_prefix),
        }
        answer_ids, answer_recorder, _ = self._run_generate(
            answer_inputs,
            max_new_tokens=config.final_answer_reserve,
            stop_on_reasoning_close=False,
            on_token=(
                (lambda token_count: on_token(len(reasoning_ids) + token_count))
                if on_token is not None
                else None
            ),
        )
        reasoning_recorder.set_segments(
            segment_generated_tokens(
                reasoning_ids,
                self.bundle.tokenizer,
                mode="reasoning",
                reasoning_close_markers=self.bundle.adapter.reasoning_close_markers(),
                reasoning_open_markers=self.bundle.adapter.reasoning_open_markers(),
            )
        )
        for record in answer_recorder.records:
            record.token_index += len(reasoning_recorder.records)
            record.segment = "final_answer"
        combined_ids = reasoning_ids + inserted_ids + answer_ids
        text = self.bundle.tokenizer.decode(
            combined_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        reasoning, final_response, boundary = split_reasoning_and_answer(text)
        finish_reason = _infer_finish_reason(
            answer_ids,
            token_limit=config.final_answer_reserve,
            eos_token_id=eos_id,
            limit_reason="answer_reserve",
        )
        return GenerationResult(
            generated_text=text,
            reasoning_text=reasoning,
            final_response_text=final_response,
            boundary_status=boundary,
            generated_token_ids=combined_ids,
            signals=reasoning_recorder.records + answer_recorder.records,
            hidden_state_indices=(
                reasoning_recorder.hidden_state_indices
                + [
                    index + len(reasoning_recorder.records)
                    for index in answer_recorder.hidden_state_indices
                ]
            ),
            hidden_states=reasoning_recorder.hidden_states + answer_recorder.hidden_states,
            finish_reason=finish_reason,
            inserted_boundary_token_count=len(inserted_ids),
            reasoning_boundary_forced=not already_closed,
            reasoning_stage_token_count=len(reasoning_ids),
        )

    def generate(
        self,
        problem: str,
        on_token: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        """Run standard or assigned-budget generation."""

        config = self.bundle.model_config
        if config.reasoning_budget is not None:
            return self._budgeted_generation(problem, on_token=on_token)
        return self._standard_generation(problem, on_token=on_token)

    def continue_from_generated_prefix(
        self,
        problem: str,
        generated_prefix_token_ids: list[int],
        *,
        max_total_generated_tokens: int | None = None,
    ) -> ContinuationResult:
        """Continue an exact stored prefix without recomputing branch features.

        The prefix is replayed through the model to reconstruct its state. This
        is scientifically equivalent to conditioning on the exact token prefix,
        although it is slower than retaining a live KV cache. The total budget
        includes both the supplied prefix and newly sampled continuation.
        """

        import torch

        config = self.bundle.model_config
        if config.reasoning_budget is not None:
            raise InstrumentationError(
                "Breakthrough continuation requires a standard-generation configuration"
            )
        total_budget = max_total_generated_tokens or config.max_new_tokens
        if total_budget <= len(generated_prefix_token_ids):
            raise InstrumentationError(
                "The continuation budget must exceed the stored generated-prefix length"
            )
        inputs = _prepare_inputs(self.bundle, problem, config.mode)
        input_ids = inputs["input_ids"]
        prefix_tensor = torch.tensor(
            [generated_prefix_token_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        combined_input_ids = torch.cat([input_ids, prefix_tensor], dim=-1)
        continuation_inputs = dict(inputs)
        continuation_inputs["input_ids"] = combined_input_ids
        continuation_inputs["attention_mask"] = torch.ones_like(combined_input_ids)
        # Some text processors emit auxiliary token-aligned fields. Extending
        # stale prompt-length values would misalign the branch, so let the model
        # derive standard causal positions from the complete attention mask.
        for key in ("position_ids", "cache_position", "token_type_ids"):
            continuation_inputs.pop(key, None)
        remaining = total_budget - len(generated_prefix_token_ids)
        prompt_and_prefix_length = int(combined_input_ids.shape[-1])
        with torch.inference_mode():
            sequences = self.bundle.model.generate(
                **continuation_inputs,
                max_new_tokens=remaining,
                pad_token_id=(
                    self.bundle.tokenizer.pad_token_id
                    if self.bundle.tokenizer.pad_token_id is not None
                    else self.bundle.tokenizer.eos_token_id
                ),
                **_generation_kwargs(config),
            )
        continuation_ids = sequences[0, prompt_and_prefix_length:].tolist()
        combined_generated_ids = [*generated_prefix_token_ids, *continuation_ids]
        text = self.bundle.tokenizer.decode(
            combined_generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        finish_reason = _infer_finish_reason(
            continuation_ids,
            token_limit=remaining,
            eos_token_id=self.bundle.tokenizer.eos_token_id,
            limit_reason="max_new_tokens",
        )
        return ContinuationResult(
            generated_text=text,
            generated_token_ids=combined_generated_ids,
            continuation_token_ids=continuation_ids,
            finish_reason=finish_reason,
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
        """Test whether a prefix can solve within a fixed additional budget.

        The model may close its reasoning naturally. If it does not, the
        adapter's official reasoning-close marker is inserted after the fixed
        continuation window and a separate answer reserve is generated. This
        makes anchor success comparable across early and late prefixes and
        prevents every probe from receiving an uninformative full 16K restart.
        """

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

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
        inputs = _prepare_inputs(self.bundle, problem, config.mode)
        prompt_ids = inputs["input_ids"]
        prefix_tensor = torch.tensor(
            [generated_prefix_token_ids], dtype=prompt_ids.dtype, device=prompt_ids.device
        )
        combined_prompt = torch.cat([prompt_ids, prefix_tensor], dim=-1)
        continuation_inputs = dict(inputs)
        continuation_inputs["input_ids"] = combined_prompt
        continuation_inputs["attention_mask"] = torch.ones_like(combined_prompt)
        for key in ("position_ids", "cache_position", "token_type_ids"):
            continuation_inputs.pop(key, None)

        close_sequences = [
            sequence
            for marker in self.bundle.adapter.reasoning_close_markers()
            if (sequence := self.bundle.tokenizer.encode(marker, add_special_tokens=False))
        ]

        class StopOnClose(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
                del scores, kwargs
                tokens = input_ids[0].tolist()
                should_stop = any(
                    len(tokens) >= len(sequence) and tokens[-len(sequence) :] == sequence
                    for sequence in close_sequences
                )
                return torch.tensor([should_stop], dtype=torch.bool, device=input_ids.device)

        prompt_length = int(combined_prompt.shape[-1])
        with torch.inference_mode():
            sequences = self.bundle.model.generate(
                **continuation_inputs,
                max_new_tokens=reasoning_limit,
                stopping_criteria=StoppingCriteriaList([StopOnClose()]),
                pad_token_id=(
                    self.bundle.tokenizer.pad_token_id
                    if self.bundle.tokenizer.pad_token_id is not None
                    else self.bundle.tokenizer.eos_token_id
                ),
                **_generation_kwargs(config),
            )
        reasoning_ids = sequences[0, prompt_length:].tolist()
        already_closed = any(
            len(reasoning_ids) >= len(sequence) and reasoning_ids[-len(sequence) :] == sequence
            for sequence in close_sequences
        )
        eos_id = self.bundle.tokenizer.eos_token_id
        if not already_closed and reasoning_ids and _token_is_eos(reasoning_ids[-1], eos_id):
            reasoning_ids = reasoning_ids[:-1]
            sequences = sequences[:, :-1]
        inserted_ids = []
        if not already_closed:
            inserted_ids = self.bundle.tokenizer.encode(
                self.bundle.adapter.forced_reasoning_close_text(), add_special_tokens=False
            )
        answer_prefix = torch.cat(
            [
                sequences,
                torch.tensor([inserted_ids], dtype=sequences.dtype, device=sequences.device),
            ],
            dim=-1,
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
        with torch.inference_mode():
            completed = self.bundle.model.generate(
                input_ids=answer_prefix,
                attention_mask=torch.ones_like(answer_prefix),
                max_new_tokens=answer_budget,
                pad_token_id=(
                    self.bundle.tokenizer.pad_token_id
                    if self.bundle.tokenizer.pad_token_id is not None
                    else eos_id
                ),
                **_generation_kwargs(config),
            )
        answer_ids = completed[0, answer_prefix.shape[-1] :].tolist()
        continuation_ids = [*reasoning_ids, *inserted_ids, *answer_ids]
        combined_generated_ids = [*generated_prefix_token_ids, *continuation_ids]
        text = self.bundle.tokenizer.decode(
            combined_generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return ContinuationResult(
            generated_text=text,
            generated_token_ids=combined_generated_ids,
            continuation_token_ids=continuation_ids,
            finish_reason=_infer_finish_reason(
                answer_ids,
                token_limit=answer_budget,
                eos_token_id=eos_id,
                limit_reason="answer_reserve",
            ),
            inserted_boundary_token_count=len(inserted_ids),
            reasoning_boundary_forced=not already_closed,
            reasoning_continuation_token_count=len(reasoning_ids),
        )
