#!/usr/bin/env python
"""Run instrumented BF16 smoke tests for every selected checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from tqdm.auto import tqdm

from reasonbench.config import load_model_config
from reasonbench.generation import InstrumentedGenerator
from reasonbench.generation.engine import _generation_kwargs, _prepare_inputs
from reasonbench.generation.modeling import load_model_bundle, unload_model_bundle
from reasonbench.runtime import set_global_seed, write_runtime_manifest
from reasonbench.storage import ensure_directory, write_json_atomic

SMOKE_PROTOCOL_VERSION = "phase4b_instrumentation_preflight_v2"
SMOKE_PROBLEM = (
    "A box contains 12 red balls and 8 blue balls. Five red balls are removed. "
    "How many balls remain in the box?"
)
SMOKE_SEED = 20260728
SMOKE_STAGES = (
    "clear CUDA cache",
    "load checkpoint",
    "read checkpoint metadata",
    "build and validate prompts",
    "audit final hidden capture",
    "run reference generation",
    "run instrumented generation",
    "exercise reasoning controls",
    "validate and save result",
)


class TokenProgress:
    """Display per-token generation progress without storing model outputs."""

    def __init__(self, description: str, total: int) -> None:
        self.total = total
        self.completed = 0
        self.bar = tqdm(
            total=total,
            desc=description,
            unit="token",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def update(self, completed: int) -> None:
        target = min(completed, self.total)
        if target > self.completed:
            self.bar.update(target - self.completed)
            self.completed = target

    def close(self) -> None:
        self.bar.close()


class ProgressStoppingCriteria:
    """Update a token bar while allowing ordinary Transformers generation to continue."""

    def __init__(self, prompt_length: int, progress: TokenProgress) -> None:
        self.prompt_length = prompt_length
        self.progress = progress

    def __call__(self, input_ids, scores, **kwargs):
        del scores, kwargs
        self.progress.update(int(input_ids.shape[-1]) - self.prompt_length)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--model-config",
        action="append",
        type=Path,
        dest="model_configs",
        required=True,
    )
    parser.add_argument("--maximum-allocated-gib", type=float, default=35.0)
    return parser.parse_args()


def _find_hidden_states(output):
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        return hidden_states
    for attribute in (
        "language_model_output",
        "model_output",
        "text_model_output",
    ):
        nested = getattr(output, attribute, None)
        hidden_states = getattr(nested, "hidden_states", None)
        if hidden_states:
            return hidden_states
    return None


def _validate_final_hidden_capture(bundle, inputs) -> dict:
    import torch

    captured = {}

    def capture(module, arguments):
        del module
        captured["hidden"] = arguments[0].detach()

    handle = bundle.model.get_output_embeddings().register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            output = bundle.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    hidden_states = _find_hidden_states(output)
    if not hidden_states:
        raise RuntimeError(
            "The official model output did not expose hidden_states for capture audit"
        )
    if "hidden" not in captured:
        raise RuntimeError("The language-model-head input hook did not fire")
    head_input = captured["hidden"].reshape(-1, captured["hidden"].shape[-1])[-1].float()
    reported = hidden_states[-1].reshape(-1, hidden_states[-1].shape[-1])[-1].float()
    if head_input.shape != reported.shape:
        raise RuntimeError(
            f"Final hidden shape {reported.shape} differs from head input {head_input.shape}"
        )
    maximum_difference = float((head_input - reported).abs().max().item())
    if not torch.allclose(head_input, reported, rtol=1e-3, atol=2e-3):
        raise RuntimeError(
            "The captured language-model-head input does not match the official "
            f"final hidden state; max absolute difference={maximum_difference:.6g}"
        )
    return {
        "status": "passed",
        "hidden_width": int(head_input.numel()),
        "maximum_absolute_difference": maximum_difference,
        "normalization_semantics": (
            "Matches the checkpoint output's final hidden state at the last text position"
        ),
    }


def _validate_reasoning_controls(bundle) -> dict:
    """Verify that official template controls and budget boundaries are operational."""

    adapter = bundle.adapter
    result = {
        "supports_reasoning_off": adapter.supports_reasoning_off,
        "supports_assigned_reasoning_budget": (
            adapter.supports_assigned_reasoning_budget
        ),
    }
    if adapter.supports_reasoning_off:
        reasoning_inputs = _prepare_inputs(bundle, SMOKE_PROBLEM, "reasoning")
        non_reasoning_inputs = _prepare_inputs(bundle, SMOKE_PROBLEM, "non_reasoning")
        reasoning_ids = reasoning_inputs["input_ids"][0].tolist()
        non_reasoning_ids = non_reasoning_inputs["input_ids"][0].tolist()
        if reasoning_ids == non_reasoning_ids:
            raise RuntimeError(
                "The official thinking on/off template controls produced identical prompts"
            )
        result["mode_prompt_difference"] = "passed"
        result["reasoning_prompt_tokens"] = len(reasoning_ids)
        result["non_reasoning_prompt_tokens"] = len(non_reasoning_ids)
    if adapter.supports_assigned_reasoning_budget:
        markers = adapter.reasoning_close_markers()
        marker_ids = [
            bundle.tokenizer.encode(marker, add_special_tokens=False)
            for marker in markers
        ]
        inserted_ids = bundle.tokenizer.encode(
            adapter.forced_reasoning_close_text(),
            add_special_tokens=False,
        )
        if not markers or not all(marker_ids) or not inserted_ids:
            raise RuntimeError(
                "Assigned-budget reasoning boundaries do not tokenize to nonempty sequences"
            )
        result["budget_boundary_check"] = "passed"
        result["reasoning_close_markers"] = list(markers)
        result["forced_boundary_token_count"] = len(inserted_ids)
    return result


def _validate_controlled_generations(
    bundle,
    base_config,
    token_progress_factory,
) -> dict:
    """Exercise reasoning-off and externally capped generation when supported."""

    audit: dict = {}
    original_config = bundle.model_config
    try:
        if bundle.adapter.supports_reasoning_off:
            bundle.model_config = replace(
                base_config,
                mode="non_reasoning",
                reasoning_budget=None,
                reasoning_budget_policy="none",
                max_new_tokens=32,
                final_answer_reserve=32,
            )
            set_global_seed(SMOKE_SEED)
            token_progress = token_progress_factory("reasoning-off", 32)
            try:
                non_reasoning = InstrumentedGenerator(bundle).generate(
                    SMOKE_PROBLEM,
                    on_token=token_progress.update,
                )
            finally:
                token_progress.close()
            if not non_reasoning.signals:
                raise RuntimeError("Reasoning-off smoke generation captured no signals")
            if non_reasoning.boundary_status == "gemma_thought_channel":
                raise RuntimeError(
                    "Gemma emitted a thought channel when thinking was explicitly disabled"
                )
            audit["reasoning_off_generation"] = "passed"
            audit["reasoning_off_generated_tokens"] = len(
                non_reasoning.generated_token_ids
            )
        if bundle.adapter.supports_assigned_reasoning_budget:
            budget = 16
            bundle.model_config = replace(
                base_config,
                mode="reasoning",
                reasoning_budget=budget,
                reasoning_budget_policy="external_hard_cap",
                final_answer_reserve=32,
                max_new_tokens=budget + 32,
            )
            set_global_seed(SMOKE_SEED)
            token_progress = token_progress_factory("budgeted reasoning", budget + 32)
            try:
                budgeted = InstrumentedGenerator(bundle).generate(
                    SMOKE_PROBLEM,
                    on_token=token_progress.update,
                )
            finally:
                token_progress.close()
            if budgeted.reasoning_stage_token_count is None:
                raise RuntimeError("Budgeted smoke generation did not report stage length")
            if budgeted.reasoning_stage_token_count > budget:
                raise RuntimeError("Budgeted smoke generation exceeded its assigned cap")
            if budgeted.reasoning_boundary_forced != (
                budgeted.inserted_boundary_token_count > 0
            ):
                raise RuntimeError("Forced-boundary metadata is internally inconsistent")
            if budgeted.boundary_status != "gemma_thought_channel":
                raise RuntimeError(
                    "Budgeted Gemma generation did not produce a parseable thought boundary"
                )
            audit["assigned_budget_generation"] = "passed"
            audit["assigned_budget_tokens"] = budget
            audit["reasoning_stage_tokens"] = budgeted.reasoning_stage_token_count
            audit["reasoning_boundary_forced"] = (
                budgeted.reasoning_boundary_forced
            )
            audit["inserted_boundary_tokens"] = (
                budgeted.inserted_boundary_token_count
            )
    finally:
        bundle.model_config = original_config
    return audit


def _validate_signal_schema(generated) -> dict[str, object]:
    """Validate every Phase 4b token-level observable on a real checkpoint."""

    if len(generated.signals) < 2:
        raise RuntimeError("Instrumentation smoke needs at least two generated tokens")
    allowed_segments = {"thinking", "solution", "final_answer", "special"}
    js_upper_bound = math.log(2.0) + 1e-6
    for index, signal in enumerate(generated.signals):
        if not 0.0 <= signal.top1_probability <= signal.top5_probability_mass <= 1.0 + 1e-6:
            raise RuntimeError("Top-1/top-5 probability masses violate their probability bounds")
        if not math.isclose(
            signal.top5_probability_mass + signal.probability_tail_mass,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError("Top-5 mass and probability tail mass do not sum to one")
        if signal.effective_vocabulary_size < 1.0:
            raise RuntimeError("Effective vocabulary size is below one")
        if signal.sampled_token_regret < -1e-6 or signal.surprisal < -1e-6:
            raise RuntimeError("Sampled-token regret or surprisal is negative")
        if signal.segment not in allowed_segments:
            raise RuntimeError(f"Unexpected token segment {signal.segment!r}")
        transitions = (signal.successive_kl_divergence, signal.successive_js_divergence)
        geometry = (signal.relative_l2_step, signal.cosine_drift)
        if index == 0:
            if transitions != (None, None) or geometry != (None, None):
                raise RuntimeError("First token must have no predecessor-dependent signals")
        else:
            if any(value is None or not math.isfinite(value) for value in (*transitions, *geometry)):
                raise RuntimeError("Later token is missing a finite transition or geometry signal")
            if signal.successive_kl_divergence < -1e-6:
                raise RuntimeError("Successive KL divergence is negative")
            if not -1e-6 <= signal.successive_js_divergence <= js_upper_bound:
                raise RuntimeError("Successive JS divergence is outside [0, log(2)]")
    thinking_tokens = sum(signal.segment == "thinking" for signal in generated.signals)
    if thinking_tokens == 0:
        raise RuntimeError("Reasoning-mode smoke generation captured no thinking tokens")
    return {
        "status": "passed",
        "signal_count": len(generated.signals),
        "thinking_token_count": thinking_tokens,
        "transition_schema": "first token null; later KL/JS/velocity/drift finite",
        "distribution_schema": "top1<=top5, top5+tail=1, JS in [0, log(2)]",
    }


class SmokeProgress:
    """Render a Colab-friendly smoke-test progress bar and durable status file."""

    def __init__(self, output_dir: Path, model_count: int) -> None:
        self.output_dir = output_dir
        self.model_count = model_count
        self.completed_models = 0
        self.current_model: str | None = None
        self.started = time.perf_counter()
        self.bar = tqdm(
            total=model_count * len(SMOKE_STAGES),
            desc="Phase 0 smoke test",
            unit="stage",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self._write_status(stage="starting")

    def stage(self, model_key: str, stage: str) -> None:
        self.current_model = model_key
        self.bar.set_postfix_str(
            f"model {self.completed_models + 1}/{self.model_count}: {model_key} | {stage}"
        )
        self.bar.update(1)
        self._write_status(stage=stage)

    def complete_model(self) -> None:
        self.completed_models += 1
        self._write_status(stage="model complete")

    def close(self) -> None:
        self.current_model = None
        self._write_status(stage="complete")
        self.bar.close()

    def _write_status(self, stage: str) -> None:
        write_json_atomic(
            self.output_dir / "smoke_progress.json",
            {
                "status": "complete" if stage == "complete" else "running",
                "current_model": self.current_model,
                "current_stage": stage,
                "completed_models": self.completed_models,
                "total_models": self.model_count,
                "completed_stages": self.bar.n,
                "total_stages": self.bar.total,
                "elapsed_seconds": time.perf_counter() - self.started,
            },
        )


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={"purpose": "phase_00_model_smoke_test"},
    )
    readiness: dict = {
        "smoke_protocol_version": SMOKE_PROTOCOL_VERSION,
        "models": {},
        "all_ready": True,
    }
    progress = SmokeProgress(output_dir, len(args.model_configs))
    for config_path in args.model_configs:
        base_config = load_model_config(args.project_root / config_path)
        config = replace(
            base_config,
            max_new_tokens=args.max_new_tokens,
            reasoning_budget=None,
            capture_hidden_states=True,
            hidden_state_stride=8,
            sampling=replace(
                base_config.sampling,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
            ),
        )
        started = time.perf_counter()
        bundle = None
        result_record: dict
        try:
            import torch

            progress.stage(config.key, "clear CUDA cache")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            progress.stage(config.key, "load checkpoint")
            bundle = load_model_bundle(config)
            progress.stage(config.key, "read checkpoint metadata")
            from huggingface_hub import HfApi

            hub_information = HfApi().model_info(
                config.model_id,
                revision=bundle.resolved_revision,
            )
            card_data = hub_information.card_data
            license_name = getattr(card_data, "license", None) if card_data is not None else None
            progress.stage(config.key, "build and validate prompts")
            inputs = _prepare_inputs(bundle, SMOKE_PROBLEM, config.mode)
            reasoning_control_audit = _validate_reasoning_controls(bundle)
            progress.stage(config.key, "audit final hidden capture")
            hidden_capture_audit = _validate_final_hidden_capture(bundle, inputs)
            prompt_length = int(inputs["input_ids"].shape[-1])
            set_global_seed(SMOKE_SEED)
            progress.stage(config.key, "run reference generation")
            reference_progress = TokenProgress(
                f"{config.key}: reference generation",
                config.max_new_tokens,
            )
            try:
                from transformers import StoppingCriteriaList

                with torch.inference_mode():
                    reference_sequences = bundle.model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        stopping_criteria=StoppingCriteriaList(
                            [ProgressStoppingCriteria(prompt_length, reference_progress)]
                        ),
                        pad_token_id=(
                            bundle.tokenizer.pad_token_id
                            if bundle.tokenizer.pad_token_id is not None
                            else bundle.tokenizer.eos_token_id
                        ),
                        **_generation_kwargs(config),
                    )
            finally:
                reference_progress.close()
            reference_ids = reference_sequences[0, prompt_length:].tolist()
            set_global_seed(SMOKE_SEED)
            generator = InstrumentedGenerator(bundle)
            progress.stage(config.key, "run instrumented generation")
            instrumented_progress = TokenProgress(
                f"{config.key}: instrumented generation",
                config.max_new_tokens,
            )
            try:
                generated = generator.generate(
                    SMOKE_PROBLEM,
                    on_token=instrumented_progress.update,
                )
            finally:
                instrumented_progress.close()
            progress.stage(config.key, "exercise reasoning controls")
            controlled_generation_audit = _validate_controlled_generations(
                bundle,
                config,
                lambda operation, token_total, model_key=config.key: TokenProgress(
                    f"{model_key}: {operation}",
                    token_total,
                ),
            )
            allocated_gib = torch.cuda.max_memory_allocated(0) / 1024**3
            reserved_gib = torch.cuda.max_memory_reserved(0) / 1024**3
            if allocated_gib >= args.maximum_allocated_gib:
                raise RuntimeError(
                    f"Peak allocated VRAM {allocated_gib:.2f} GiB exceeds "
                    f"{args.maximum_allocated_gib:.2f} GiB"
                )
            if not generated.signals:
                raise RuntimeError("No token signals were captured")
            if generated.generated_token_ids != reference_ids:
                raise RuntimeError(
                    "Instrumented generation changed the sampled output relative to "
                    "the same-seed standard generation diagnostic"
                )
            scalar_names = (
                "entropy",
                "normalized_entropy",
                "top1_top2_logit_margin",
                "top1_top2_probability_margin",
                "sampled_logprob",
                "surprisal",
                "hidden_norm",
            )
            if not all(
                math.isfinite(getattr(signal, name))
                for signal in generated.signals
                for name in scalar_names
            ):
                raise RuntimeError("Captured scalar signals contain NaN or Inf")
            signal_schema_audit = _validate_signal_schema(generated)
            progress.stage(config.key, "validate and save result")
            output_head = bundle.model.get_output_embeddings()
            capture_module = next(
                (name for name, module in bundle.model.named_modules() if module is output_head),
                "<unresolved>",
            )
            result_record = {
                "status": "ready",
                "model_id": config.model_id,
                "model_key": config.key,
                "resolved_revision": bundle.resolved_revision,
                "license": license_name,
                "architecture": bundle.architecture,
                "bf16_parameter_fraction": bundle.bf16_parameter_fraction,
                "generated_tokens": len(generated.generated_token_ids),
                "captured_signals": len(generated.signals),
                "captured_hidden_states": len(generated.hidden_states),
                "token_logit_alignment": "passed",
                "same_seed_output_equivalence": "passed",
                "finite_signal_check": "passed",
                "signal_schema_audit": signal_schema_audit,
                "smoke_protocol_version": SMOKE_PROTOCOL_VERSION,
                "capture_module": capture_module,
                "capture_tensor_semantics": (
                    "last-position input to the exposed language-model output head"
                ),
                "hidden_capture_audit": hidden_capture_audit,
                "reasoning_control_audit": reasoning_control_audit,
                "controlled_generation_audit": controlled_generation_audit,
                "prompt_source": getattr(bundle.adapter, "prompt_source", "chat_template"),
                "peak_allocated_gib": allocated_gib,
                "peak_reserved_gib": reserved_gib,
                "elapsed_seconds": time.perf_counter() - started,
                "generated_preview": generated.generated_text[:1000],
            }
        except Exception as exc:
            readiness["all_ready"] = False
            result_record = {
                "status": "needs_adapter",
                "model_id": config.model_id,
                "model_key": config.key,
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            if bundle is not None:
                unload_model_bundle(bundle)
        readiness["models"][config.key] = result_record
        write_json_atomic(output_dir / f"{config.key}_smoke_test.json", result_record)
        progress.complete_model()
    write_json_atomic(output_dir / "model_readiness.json", readiness)
    progress.close()
    print(f"Model readiness written to {output_dir / 'model_readiness.json'}")


if __name__ == "__main__":
    main()
