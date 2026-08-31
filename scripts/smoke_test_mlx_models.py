#!/usr/bin/env python
"""Real-checkpoint MLX smoke tests and immutable readiness manifest."""

from __future__ import annotations

import argparse
import math
import time
import traceback
from dataclasses import replace
from pathlib import Path

from reasonbench.config import load_model_config
from reasonbench.generation.mlx_engine import MLXInstrumentedGenerator
from reasonbench.generation.mlx_modeling import (
    load_mlx_model_bundle,
    unload_mlx_model_bundle,
)
from reasonbench.runtime import set_global_seed, write_runtime_manifest
from reasonbench.storage import ensure_directory, read_json, write_json_atomic

SMOKE_PROTOCOL_VERSION = "phase4b_mlx_int4_instrumentation_v1"
SMOKE_PROBLEM = (
    "A box contains 12 red balls and 8 blue balls. Five red balls are removed. "
    "How many balls remain in the box?"
)
SMOKE_SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--maximum-allocated-gib", type=float, default=36.0)
    parser.add_argument(
        "--model-config",
        action="append",
        type=Path,
        dest="model_configs",
        required=True,
    )
    return parser.parse_args()


def _validate(generated) -> None:
    if len(generated.signals) < 2:
        raise RuntimeError("MLX smoke test captured fewer than two token signals")
    if len(generated.signals) != len(generated.generated_token_ids):
        raise RuntimeError("MLX token and signal counts differ")
    if not generated.hidden_states:
        raise RuntimeError("MLX smoke test captured no hidden states")
    for index, signal in enumerate(generated.signals):
        scalar_names = (
            "entropy",
            "normalized_entropy",
            "top1_top2_logit_margin",
            "top1_top2_probability_margin",
            "top1_probability",
            "top5_probability_mass",
            "probability_tail_mass",
            "sampled_logprob",
            "sampled_token_regret",
            "surprisal",
            "hidden_norm",
        )
        if not all(math.isfinite(getattr(signal, name)) for name in scalar_names):
            raise RuntimeError(f"MLX signal {index} contains NaN or Inf")
        if not 0 <= signal.top1_probability <= signal.top5_probability_mass <= 1 + 1e-5:
            raise RuntimeError("MLX top-1/top-5 masses violate probability bounds")
        if not math.isclose(
            signal.top5_probability_mass + signal.probability_tail_mass,
            1.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError("MLX top-5 and tail masses do not sum to one")
        if index == 0:
            if any(
                value is not None
                for value in (
                    signal.successive_kl_divergence,
                    signal.successive_js_divergence,
                    signal.relative_l2_step,
                    signal.cosine_drift,
                )
            ):
                raise RuntimeError("First MLX token has predecessor-dependent metrics")
        elif any(
            value is None or not math.isfinite(value)
            for value in (
                signal.successive_kl_divergence,
                signal.successive_js_divergence,
                signal.relative_l2_step,
                signal.cosine_drift,
            )
        ):
            raise RuntimeError("Later MLX token is missing a transition metric")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 2:
        raise ValueError("max-new-tokens must be at least two")
    output_dir = ensure_directory(args.output_dir)
    write_runtime_manifest(
        output_dir / "runtime_manifest.json",
        project_root=args.project_root,
        extra={"purpose": "phase_04b_mlx_model_smoke_test"},
    )
    readiness_path = output_dir / "model_readiness.json"
    readiness = (
        read_json(readiness_path)
        if readiness_path.exists()
        else {
            "smoke_protocol_version": SMOKE_PROTOCOL_VERSION,
            "models": {},
            "all_ready": True,
        }
    )
    readiness["smoke_protocol_version"] = SMOKE_PROTOCOL_VERSION
    for relative_path in args.model_configs:
        base_config = load_model_config(args.project_root / relative_path)
        config = replace(
            base_config,
            max_new_tokens=args.max_new_tokens,
            hidden_state_stride=1,
            sampling=replace(
                base_config.sampling,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
            ),
        )
        bundle = None
        started = time.perf_counter()
        try:
            import mlx.core as mx

            mx.clear_cache()
            mx.reset_peak_memory()
            print(f"Loading {config.key} ({config.model_id})...", flush=True)
            bundle = load_mlx_model_bundle(config)
            set_global_seed(SMOKE_SEED)
            generation_started = time.perf_counter()
            generated = MLXInstrumentedGenerator(bundle).generate(SMOKE_PROBLEM)
            generation_elapsed = time.perf_counter() - generation_started
            _validate(generated)
            peak_gib = mx.get_peak_memory() / 1024**3
            if peak_gib >= args.maximum_allocated_gib:
                raise RuntimeError(
                    f"Peak MLX memory {peak_gib:.2f} GiB exceeds the "
                    f"{args.maximum_allocated_gib:.2f} GiB safety gate"
                )
            elapsed = time.perf_counter() - started
            record = {
                "status": "ready",
                "model_id": config.model_id,
                "source_model_id": config.source_model_id,
                "model_key": config.key,
                "backend": config.backend,
                "dtype": config.dtype,
                "resolved_revision": bundle.resolved_revision,
                "architecture": bundle.architecture,
                "quantization_bits": bundle.quantization_bits,
                "quantization_group_size": bundle.quantization_group_size,
                "prompt_source": bundle.prompt_source or "checkpoint_chat_template",
                "generated_tokens": len(generated.generated_token_ids),
                "captured_signals": len(generated.signals),
                "captured_hidden_states": len(generated.hidden_states),
                "token_logit_alignment": "passed",
                "finite_signal_check": "passed",
                "head_input_semantics": "final-normalized language-model-head input",
                "peak_allocated_gib": peak_gib,
                "elapsed_seconds": elapsed,
                "generation_elapsed_seconds": generation_elapsed,
                "generation_tokens_per_second": (
                    len(generated.generated_token_ids) / generation_elapsed
                    if generation_elapsed
                    else None
                ),
                "generated_preview": generated.generated_text[:1000],
                "smoke_protocol_version": SMOKE_PROTOCOL_VERSION,
            }
        except Exception as exc:
            record = {
                "status": "needs_adapter",
                "model_id": config.model_id,
                "source_model_id": config.source_model_id,
                "model_key": config.key,
                "backend": config.backend,
                "dtype": config.dtype,
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            if bundle is not None:
                unload_mlx_model_bundle(bundle)
        readiness["models"][config.key] = record
        write_json_atomic(output_dir / f"{config.key}_smoke_test.json", record)
        print(f"{config.key}: {record['status']}", flush=True)
    readiness["all_ready"] = all(
        record.get("status") == "ready" for record in readiness["models"].values()
    )
    write_json_atomic(readiness_path, readiness)
    print(f"Readiness: {readiness_path}")
    if not readiness["all_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
