"""Leakage-resistant per-trajectory feature extraction."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from reasonbench.evaluation.outcomes import derive_terminal_outcomes
from reasonbench.features.scalar import summarize_scalar
from reasonbench.features.spectral import summarize_spectrum
from reasonbench.storage import read_json

SIGNAL_COLUMNS = (
    "normalized_entropy",
    "top1_top2_logit_margin",
    "top1_top2_probability_margin",
    "top1_probability",
    "top5_probability_mass",
    "probability_tail_mass",
    "effective_vocabulary_size",
    "sampled_logprob",
    "sampled_token_regret",
    "surprisal",
    "successive_kl_divergence",
    "successive_js_divergence",
    "hidden_norm",
    "relative_l2_step",
    "cosine_drift",
)

SPECTRAL_COLUMNS = (
    "normalized_entropy",
    "surprisal",
    "top1_top2_logit_margin",
    "successive_js_divergence",
    "sampled_token_regret",
    "relative_l2_step",
    "cosine_drift",
)


def _analysis_window(frame: pd.DataFrame) -> pd.DataFrame:
    if (frame["segment"] == "thinking").any():
        selected = frame[frame["segment"] == "thinking"]
    else:
        selected = frame[frame["segment"] == "solution"]
    return selected.sort_values("token_index").reset_index(drop=True)


def _surface_features(problem: str) -> dict[str, int]:
    return {
        "problem_character_count": len(problem),
        "problem_token_proxy_count": len(problem.split()),
        "problem_numeric_count": len(re.findall(r"-?\d+(?:\.\d+)?", problem)),
        "problem_operator_count": len(re.findall(r"[+\-*/=<>]", problem)),
        "problem_equation_count": problem.count("="),
    }


def _geometry_features(frame: pd.DataFrame) -> dict[str, float]:
    relative = frame["relative_l2_step"].dropna().to_numpy(dtype=np.float64)
    drift = frame["cosine_drift"].dropna().to_numpy(dtype=np.float64)
    path_length = float(relative.sum()) if len(relative) else math.nan
    net_proxy = float(np.sqrt(np.square(relative).sum())) if len(relative) else math.nan
    efficiency = net_proxy / path_length if path_length and path_length > 0 else math.nan
    return {
        "geometry_normalized_path_length": path_length,
        "geometry_net_displacement_proxy": net_proxy,
        "geometry_efficiency_proxy": efficiency,
        "geometry_mean_relative_velocity": (float(relative.mean()) if len(relative) else math.nan),
        "geometry_velocity_variance": (float(relative.var()) if len(relative) else math.nan),
        "geometry_mean_cosine_drift": float(drift.mean()) if len(drift) else math.nan,
        "geometry_max_cosine_drift": float(drift.max()) if len(drift) else math.nan,
        "geometry_cosine_drift_variance": float(drift.var()) if len(drift) else math.nan,
    }


def _hidden_geometry_features(
    trajectory_directory: Path,
    included_token_indices: set[int],
) -> dict[str, float]:
    """Load one hidden-state artifact and summarize the requested token subset."""

    return _hidden_geometry_features_from_arrays(
        _load_hidden_payload(trajectory_directory),
        included_token_indices,
    )


def _load_hidden_payload(
    trajectory_directory: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    path = trajectory_directory / "hidden_states.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as payload:
            # Geometry summaries are descriptive features, not a numerically delicate
            # linear solve. Float32 halves local memory pressure and is sufficient for
            # norms, cosine drift, and turning-angle summaries.
            return payload["token_indices"], payload["hidden_states"].astype(np.float32)
    except (BadZipFile, OSError, ValueError) as exc:
        warnings.warn(
            f"Could not load hidden-state payload at {path}; retaining the trajectory "
            f"with hidden-geometry features marked missing ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _hidden_geometry_features_from_arrays(
    payload: tuple[np.ndarray, np.ndarray] | None,
    included_token_indices: set[int],
) -> dict[str, float]:
    """Summarize cached hidden states without rereading the NPZ artifact."""

    names = (
        "geometry_normalized_net_displacement",
        "geometry_sparse_path_length",
        "geometry_trajectory_efficiency",
        "geometry_turning_angle_mean",
        "geometry_turning_angle_variance",
    )
    if payload is None:
        return {name: math.nan for name in names}
    token_indices, hidden = payload
    hidden = hidden[
        np.asarray(
            [int(index) in included_token_indices for index in token_indices],
            dtype=bool,
        )
    ]
    if len(hidden) < 2:
        return {name: math.nan for name in names}
    step_vectors = np.diff(hidden, axis=0)
    previous_norms = np.linalg.norm(hidden[:-1], axis=1)
    step_norms = np.linalg.norm(step_vectors, axis=1)
    relative_steps = step_norms / np.maximum(previous_norms, 1e-12)
    sparse_path = float(relative_steps.sum())
    net = float(
        np.linalg.norm(hidden[-1] - hidden[0]) / max(float(np.linalg.norm(hidden[0])), 1e-12)
    )
    angles = np.asarray([], dtype=np.float64)
    if len(step_vectors) >= 2:
        left = step_vectors[:-1]
        right = step_vectors[1:]
        denominator = np.maximum(
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1),
            1e-12,
        )
        cosine = np.sum(left * right, axis=1) / denominator
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return {
        "geometry_normalized_net_displacement": net,
        "geometry_sparse_path_length": sparse_path,
        "geometry_trajectory_efficiency": (net / sparse_path if sparse_path > 0 else math.nan),
        "geometry_turning_angle_mean": (float(angles.mean()) if len(angles) else math.nan),
        "geometry_turning_angle_variance": (float(angles.var()) if len(angles) else math.nan),
    }


def _one_feature_row(
    trajectory_directory: Path,
    include_spectral: bool,
    prefix_length: int | None,
) -> dict[str, Any]:
    metadata = read_json(trajectory_directory / "metadata.json")
    token_frame = pd.read_parquet(trajectory_directory / "token_metrics.parquet")
    return _feature_row_from_loaded(
        trajectory_directory,
        metadata,
        token_frame,
        include_spectral=include_spectral,
        prefix_length=prefix_length,
        hidden_payload=_load_hidden_payload(trajectory_directory),
    )


def _feature_row_from_loaded(
    trajectory_directory: Path,
    metadata: dict[str, Any],
    token_frame: pd.DataFrame,
    *,
    include_spectral: bool,
    prefix_length: int | None,
    hidden_payload: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build one feature row from already-loaded trajectory artifacts."""

    full_analysis = _analysis_window(token_frame)
    analysis = full_analysis
    if prefix_length is not None:
        analysis = analysis.iloc[:prefix_length]
    verification = metadata["verification"]
    row: dict[str, Any] = {
        "run_id": metadata["run_id"],
        "experiment_id": metadata["experiment_id"],
        "phase_id": metadata["phase_id"],
        "model_key": metadata["model_key"],
        "model_mode": metadata["model_mode"],
        "dataset": metadata["dataset"],
        "problem_id": metadata["problem_id"],
        "research_split": metadata["research_split"],
        "seed": metadata["seed"],
        "level": metadata.get("level"),
        "category": metadata.get("category"),
        "assigned_reasoning_budget": metadata.get("assigned_reasoning_budget"),
        "reasoning_budget_policy": metadata.get("reasoning_budget_policy", "none"),
        "trajectory_token_count": len(analysis),
        "full_trajectory_token_count": len(full_analysis),
        "observed_token_count": len(analysis),
        "generated_tokens": metadata["generated_tokens"],
        "signal_tokens": metadata["signal_tokens"],
        "elapsed_seconds": metadata["elapsed_seconds"],
        "peak_allocated_gib": metadata.get("peak_allocated_gib"),
        "peak_reserved_gib": metadata.get("peak_reserved_gib"),
        "inserted_boundary_tokens": metadata.get("inserted_boundary_tokens", 0),
        "reasoning_boundary_forced": bool(
            metadata.get("reasoning_boundary_forced", False)
        ),
        "reasoning_stage_tokens": metadata.get("reasoning_stage_tokens"),
        "finish_reason": metadata["finish_reason"],
        "boundary_status": metadata["boundary_status"],
        "correct": bool(verification["correct"]),
        "parse_status": verification["extraction_status"],
        "prefix_length": prefix_length,
    }
    # Endpoint labels are terminal metadata, retained for supervised analysis
    # only.  They are excluded from every predictor feature set.
    row.update(
        derive_terminal_outcomes(
            correct=bool(verification["correct"]),
            finish_reason=str(metadata["finish_reason"]),
        )
    )
    row.update(_surface_features(metadata["problem"]))
    for signal in SIGNAL_COLUMNS:
        # Older committed trajectories predate some distributional features.
        # Preserve them for comparative analyses and mark only the unavailable
        # feature family missing rather than rejecting the complete run.
        values = analysis[signal].to_numpy() if signal in analysis else np.asarray([])
        row.update(summarize_scalar(values, signal))
    row.update(_geometry_features(analysis))
    row.update(
        _hidden_geometry_features_from_arrays(
            hidden_payload,
            set(analysis["token_index"].astype(int)),
        )
    )
    if include_spectral:
        for signal in SPECTRAL_COLUMNS:
            values = analysis[signal].to_numpy() if signal in analysis else np.asarray([])
            row.update(
                summarize_spectrum(
                    values,
                    prefix=f"spectral_{signal}",
                )
            )
    return row


def trajectory_directories(run_directories: Iterable[str | Path]) -> list[Path]:
    """Find every committed trajectory under one or more experiment roots."""

    directories: list[Path] = []
    for root in run_directories:
        directories.extend(marker.parent for marker in Path(root).rglob("complete.json"))
    return sorted(set(directories))


def extract_feature_table(
    run_directories: Iterable[str | Path],
    include_spectral: bool = True,
    prefix_length: int | None = None,
) -> pd.DataFrame:
    """Extract one feature row per committed trajectory."""

    directories = trajectory_directories(run_directories)
    rows = [
        _one_feature_row(directory, include_spectral, prefix_length) for directory in directories
    ]
    if not rows:
        raise ValueError("No complete trajectories were found")
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "problem_id", "model_key", "seed"])
        .reset_index(drop=True)
    )


def _feature_rows_for_directory(
    directory: Path,
    include_spectral: bool,
    prefixes: tuple[int, ...],
) -> dict[int | None, dict[str, Any]]:
    """Read one trajectory once and derive its complete and prefix rows."""

    metadata = read_json(directory / "metadata.json")
    token_frame = pd.read_parquet(directory / "token_metrics.parquet")
    hidden_payload = _load_hidden_payload(directory)
    return {
        prefix: _feature_row_from_loaded(
            directory,
            metadata,
            token_frame,
            include_spectral=include_spectral,
            prefix_length=prefix,
            hidden_payload=hidden_payload,
        )
        for prefix in (None, *prefixes)
    }


def extract_feature_tables(
    run_directories: Iterable[str | Path],
    *,
    include_spectral: bool = True,
    prefix_lengths: Iterable[int] = (),
    workers: int = 1,
) -> dict[int | None, pd.DataFrame]:
    """Extract the full and fixed-prefix tables in one artifact-read pass.

    CPU-only post-processing previously reread every Parquet and NPZ artifact once
    per prefix.  Holding one trajectory's payload only while all its rows are
    derived avoids repeated disk scans without accumulating the full corpus in RAM.
    """

    prefixes = tuple(sorted(set(prefix_lengths)))
    rows: dict[int | None, list[dict[str, Any]]] = {None: []}
    rows.update({prefix: [] for prefix in prefixes})
    directories = trajectory_directories(run_directories)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    def append_rows(table_rows: dict[int | None, dict[str, Any]]) -> None:
        for prefix, row in table_rows.items():
            rows[prefix].append(row)

    if workers == 1:
        for directory in directories:
            append_rows(_feature_rows_for_directory(directory, include_spectral, prefixes))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _feature_rows_for_directory,
                    directory,
                    include_spectral,
                    prefixes,
                )
                for directory in directories
            ]
            for future in as_completed(futures):
                append_rows(future.result())
    if not rows[None]:
        raise ValueError("No complete trajectories were found")
    return {
        prefix: (
            pd.DataFrame(table_rows)
            .sort_values(["dataset", "problem_id", "model_key", "seed"])
            .reset_index(drop=True)
        )
        for prefix, table_rows in rows.items()
    }
