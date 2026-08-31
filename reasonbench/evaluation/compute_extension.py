"""Leakage-safe utilities for uncertainty-conditioned compute extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

UNCERTAINTY_BLOCKS = {
    # One representative local ambiguity feature. Surprisal and probability
    # margin are intentionally omitted here because Phase 4B showed that they
    # are near-duplicates of normalized entropy (Spearman 0.90--0.99).
    "predictive_ambiguity": ("normalized_entropy_mean",),
    # Time-domain instability in uncertainty and next-token distributions.
    "temporal_instability": (
        "normalized_entropy_std",
        "normalized_entropy_robust_slope",
        "normalized_entropy_max_rise",
        "successive_js_divergence_mean",
    ),
    # Hidden-state movement and meandering. Lower trajectory efficiency is
    # oriented as greater uncertainty below.
    "geometry_instability": (
        "geometry_mean_relative_velocity",
        "geometry_mean_cosine_drift",
        "geometry_trajectory_efficiency",
        "geometry_turning_angle_variance",
    ),
    # Frequency-domain irregularity across predictive and hidden-state traces.
    "spectral_instability": (
        "spectral_normalized_entropy_entropy",
        "spectral_surprisal_entropy",
        "spectral_successive_js_divergence_entropy",
        "spectral_relative_l2_step_entropy",
    ),
}
UNCERTAINTY_FEATURES = tuple(
    feature for features in UNCERTAINTY_BLOCKS.values() for feature in features
)
UNCERTAINTY_SIGNS = {feature: 1.0 for feature in UNCERTAINTY_FEATURES}
UNCERTAINTY_SIGNS["geometry_trajectory_efficiency"] = -1.0
UNCERTAINTY_TRANSFORMS = {feature: "identity" for feature in UNCERTAINTY_FEATURES}
UNCERTAINTY_TRANSFORMS["normalized_entropy_robust_slope"] = "absolute"
UNCERTAINTY_SCORE_VERSION = "phase04c_u512_model_ecdf_block_balanced_v3"
PHASE04C_U_ANCHOR = 512
PHASE04C_U_TOTAL_REASONING_TARGETS = {
    "short": 1024,
    "medium": 4096,
    "long": 24576,
}
PHASE04C_U_FINAL_ANSWER_RESERVE = 512
PHASE04C_U_MAX_TOTAL_GENERATED_TOKENS = 25600
PHASE04C_U_CONTINUATIONS_PER_ARM = 4


def validate_phase04c_u_protocol(protocol: dict[str, Any]) -> None:
    """Fail closed unless a manifest matches the frozen three-arm contract."""

    if int(protocol.get("primary_anchor", -1)) != PHASE04C_U_ANCHOR:
        raise ValueError("Phase 4C-U must use the frozen 512-token anchor")
    if protocol.get("budget_semantics") != ("target_total_reasoning_tokens_including_anchor"):
        raise ValueError("Phase 4C-U budget semantics are missing or incompatible")
    if int(protocol.get("final_answer_reserve", -1)) != PHASE04C_U_FINAL_ANSWER_RESERVE:
        raise ValueError("Phase 4C-U must preserve the frozen 512-token answer reserve")
    if int(protocol.get("max_total_generated_tokens", -1)) != PHASE04C_U_MAX_TOTAL_GENERATED_TOKENS:
        raise ValueError("Phase 4C-U must use the frozen 25,600-token generated cap")
    if int(protocol.get("continuations_per_arm", -1)) != PHASE04C_U_CONTINUATIONS_PER_ARM:
        raise ValueError("Phase 4C-U requires four matched continuations per arm")
    expected_overhead_reserve = (
        PHASE04C_U_MAX_TOTAL_GENERATED_TOKENS
        - PHASE04C_U_TOTAL_REASONING_TARGETS["long"]
        - PHASE04C_U_FINAL_ANSWER_RESERVE
    )
    if (
        int(protocol.get("nominal_prefix_and_boundary_overhead_reserve", -1))
        != expected_overhead_reserve
    ):
        raise ValueError("Phase 4C-U prefix/boundary overhead reserve is incompatible")
    arms = protocol.get("arms", {})
    if set(arms) != set(PHASE04C_U_TOTAL_REASONING_TARGETS):
        raise ValueError("Phase 4C-U requires exactly short, medium, and long arms")
    for arm, target in PHASE04C_U_TOTAL_REASONING_TARGETS.items():
        arm_protocol = arms[arm]
        if int(arm_protocol.get("target_total_reasoning_tokens", -1)) != target:
            raise ValueError(f"Phase 4C-U {arm} target differs from the frozen protocol")
        expected_continuation = target - PHASE04C_U_ANCHOR
        if int(arm_protocol.get("reasoning_continuation_budget", -1)) != expected_continuation:
            raise ValueError(f"Phase 4C-U {arm} continuation does not match target minus anchor")
    if not protocol.get("paired_branch_seeds"):
        raise ValueError("Phase 4C-U requires matched branch seeds")
    if not protocol.get("nested_token_paths_required"):
        raise ValueError("Phase 4C-U requires nested token paths")


def validate_compute_extension_protocol(protocol: dict[str, Any]) -> None:
    """Validate either the historical Phase 4C-U or frozen Phase 5 contract."""

    version = protocol.get("protocol_schema_version", "phase04c_u_v1")
    if version == "phase04c_u_v1":
        validate_phase04c_u_protocol(protocol)
        return
    if version != "phase05_breakthrough_controller_v1":
        raise ValueError(f"Unknown compute extension protocol: {version}")
    if int(protocol.get("primary_anchor", -1)) != PHASE04C_U_ANCHOR:
        raise ValueError("Phase 5 must route from the frozen 512-token anchor")
    if protocol.get("budget_semantics") != "target_total_reasoning_tokens_including_anchor":
        raise ValueError("Phase 5 reasoning-budget semantics are incompatible")
    if int(protocol.get("final_answer_reserve", -1)) != 4096:
        raise ValueError("Phase 5 requires an equal 4,096-token final-answer reserve")
    if int(protocol.get("max_total_generated_tokens", -1)) != 29696:
        raise ValueError("Phase 5 requires the frozen 29,696-token safety cap")
    if int(protocol.get("continuations_per_arm", -1)) != 1:
        raise ValueError("Phase 5 confirmatory HARP run uses one seed per arm")
    arms = protocol.get("arms", {})
    if set(arms) != set(PHASE04C_U_TOTAL_REASONING_TARGETS):
        raise ValueError("Phase 5 requires exactly short, medium, and long arms")
    for arm, target in PHASE04C_U_TOTAL_REASONING_TARGETS.items():
        if int(arms[arm].get("target_total_reasoning_tokens", -1)) != target:
            raise ValueError(f"Phase 5 {arm} target differs from the frozen protocol")
        if int(arms[arm].get("reasoning_continuation_budget", -1)) != target - 512:
            raise ValueError(f"Phase 5 {arm} continuation budget is incompatible")
    if not protocol.get("paired_branch_seeds") or not protocol.get("nested_token_paths_required"):
        raise ValueError("Phase 5 requires paired seeds and nested token paths")


@dataclass(frozen=True)
class EmpiricalReference:
    """Frozen, outcome-independent reference distribution for one feature."""

    sorted_oriented_values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.sorted_oriented_values),
            "sorted_oriented_values": list(self.sorted_oriented_values),
        }


def fit_percentile_references(
    frame: pd.DataFrame,
    *,
    model_column: str = "model_key",
    split_column: str = "research_split",
    training_split: str = "train",
) -> dict[str, dict[str, EmpiricalReference]]:
    """Freeze within-model empirical CDFs on the immutable training split.

    Features are oriented before fitting, so larger reference values always
    mean greater local predictive uncertainty. Correctness and trajectory
    outcomes never enter this transformation.
    """

    required = {model_column, split_column, *UNCERTAINTY_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Uncertainty feature table is missing columns: {sorted(missing)}")
    training = frame[frame[split_column] == training_split]
    if training.empty:
        raise ValueError(f"No rows belong to the frozen training split {training_split!r}")
    result: dict[str, dict[str, EmpiricalReference]] = {}
    for model_key, group in training.groupby(model_column, sort=True):
        result[str(model_key)] = {}
        for feature in UNCERTAINTY_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                raise ValueError(f"No finite training values for {model_key}:{feature}")
            oriented = np.sort(_orient_uncertainty_values(feature, values))
            result[str(model_key)][feature] = EmpiricalReference(
                sorted_oriented_values=tuple(float(value) for value in oriented)
            )
    return result


def _orient_uncertainty_values(feature: str, values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=float)
    transform = UNCERTAINTY_TRANSFORMS[feature]
    if transform == "absolute":
        transformed = np.abs(transformed)
    elif transform != "identity":
        raise ValueError(f"Unknown uncertainty transform for {feature}: {transform}")
    return transformed * UNCERTAINTY_SIGNS[feature]


def score_uncertainty_components(
    frame: pd.DataFrame,
    references: dict[str, dict[str, EmpiricalReference]],
    *,
    model_column: str = "model_key",
) -> pd.DataFrame:
    """Return feature, block, and overall model-conditional uncertainty scores.

    Each oriented feature is mapped through its training-only empirical CDF.
    Features are averaged within their pre-specified conceptual block; the four
    block scores are then averaged. Consequently a block with four correlated
    summaries cannot outweigh the single-feature predictive-ambiguity block.
    The result is a relative uncertainty index, not a calibrated probability
    of failure.
    """

    required = {model_column, *UNCERTAINTY_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Uncertainty feature table is missing columns: {sorted(missing)}")
    rows: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        model_key = str(row[model_column])
        if model_key not in references:
            raise ValueError(f"No frozen uncertainty reference for model {model_key}")
        feature_scores: dict[str, float] = {}
        for feature in UNCERTAINTY_FEATURES:
            value = float(row[feature])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite uncertainty feature for {model_key}:{feature}")
            reference = references[model_key][feature]
            oriented = float(_orient_uncertainty_values(feature, np.asarray([value]))[0])
            # The right-continuous empirical CDF maps the training maximum to
            # one and values below the training minimum to zero. Frozen train
            # references prevent validation/test leakage.
            percentile = np.searchsorted(
                reference.sorted_oriented_values,
                oriented,
                side="right",
            ) / len(reference.sorted_oriented_values)
            feature_scores[feature] = float(percentile)
        block_scores = {
            block: float(np.mean([feature_scores[feature] for feature in features]))
            for block, features in UNCERTAINTY_BLOCKS.items()
        }
        normalized_score = float(np.mean(list(block_scores.values())))
        if not 0.0 <= normalized_score <= 1.0:
            raise RuntimeError("Normalized uncertainty index escaped [0, 1]")
        rows.append(
            {
                **{
                    f"uncertainty_feature_percentile__{feature}": score
                    for feature, score in feature_scores.items()
                },
                **{
                    f"uncertainty_block__{block}": score
                    for block, score in block_scores.items()
                },
                "uncertainty_score": normalized_score,
            }
        )
    return pd.DataFrame(rows, index=frame.index, dtype=float)


def score_uncertainty_rows(
    frame: pd.DataFrame,
    references: dict[str, dict[str, EmpiricalReference]],
    *,
    model_column: str = "model_key",
) -> pd.Series:
    """Return the block-balanced uncertainty index in the interval [0, 1]."""

    scored = score_uncertainty_components(
        frame,
        references,
        model_column=model_column,
    )
    return scored["uncertainty_score"].rename("uncertainty_score")


def assign_balanced_uncertainty_strata(
    frame: pd.DataFrame,
    *,
    score_column: str = "uncertainty_score",
    group_columns: tuple[str, ...] = ("model_key", "level"),
    identity_column: str = "run_id",
) -> pd.Series:
    """Assign deterministic low/high halves within model-difficulty cells.

    Exact ranks, with ``run_id`` as the frozen tie breaker, avoid unstable
    median equality behavior. Odd cells leave the median-ranked row unassigned;
    it is retained in the manifest but excluded from the primary interaction.
    """

    required = {*group_columns, score_column, identity_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot stratify uncertainty; missing columns: {sorted(missing)}")
    result = pd.Series("unassigned", index=frame.index, dtype="object")
    for _, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        ordered = group.sort_values([score_column, identity_column], kind="mergesort")
        half = len(ordered) // 2
        if half < 1:
            continue
        result.loc[ordered.index[:half]] = "low"
        result.loc[ordered.index[-half:]] = "high"
    return result.rename("uncertainty_stratum")


def serialize_references(
    references: dict[str, dict[str, EmpiricalReference]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        model: {feature: reference.to_dict() for feature, reference in features.items()}
        for model, features in references.items()
    }


def paired_budget_effects(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive paired three-arm correctness effects.

    The confirmatory contrast is long minus medium. Short-to-medium and
    short-to-long are pre-specified secondary dose-response contrasts.
    """

    required = {
        "problem_id",
        "source_run_id",
        "branch_index",
        "uncertainty_stratum",
        "short_correct",
        "medium_correct",
        "long_correct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Paired extension table is missing columns: {sorted(missing)}")
    summary: dict[str, Any] = {"paired_branches": int(len(frame)), "strata": {}}
    contrasts_by_stratum: dict[str, dict[str, float]] = {}
    for stratum in ("low", "high"):
        group = frame[frame["uncertainty_stratum"] == stratum]
        if group.empty:
            summary["strata"][stratum] = {
                "paired_branches": 0,
                "problems": 0,
                "short_accuracy": None,
                "medium_accuracy": None,
                "long_accuracy": None,
                "contrasts": {},
            }
            continue
        short_accuracy = float(group["short_correct"].astype(float).mean())
        medium_accuracy = float(group["medium_correct"].astype(float).mean())
        long_accuracy = float(group["long_correct"].astype(float).mean())
        contrasts = {
            "medium_minus_short": medium_accuracy - short_accuracy,
            "long_minus_medium": long_accuracy - medium_accuracy,
            "long_minus_short": long_accuracy - short_accuracy,
        }
        contrasts_by_stratum[stratum] = contrasts
        summary["strata"][stratum] = {
            "paired_branches": int(len(group)),
            "problems": int(group["problem_id"].nunique()),
            "short_accuracy": short_accuracy,
            "medium_accuracy": medium_accuracy,
            "long_accuracy": long_accuracy,
            "contrasts": contrasts,
            "transitions": {
                "medium_vs_short_rescued": int(
                    (~group["short_correct"] & group["medium_correct"]).sum()
                ),
                "medium_vs_short_harmed": int(
                    (group["short_correct"] & ~group["medium_correct"]).sum()
                ),
                "long_vs_medium_rescued": int(
                    (~group["medium_correct"] & group["long_correct"]).sum()
                ),
                "long_vs_medium_harmed": int(
                    (group["medium_correct"] & ~group["long_correct"]).sum()
                ),
                "long_vs_short_rescued": int(
                    (~group["short_correct"] & group["long_correct"]).sum()
                ),
                "long_vs_short_harmed": int(
                    (group["short_correct"] & ~group["long_correct"]).sum()
                ),
            },
        }
    summary["interactions_high_minus_low"] = (
        {
            contrast: contrasts_by_stratum["high"][contrast] - contrasts_by_stratum["low"][contrast]
            for contrast in (
                "medium_minus_short",
                "long_minus_medium",
                "long_minus_short",
            )
        }
        if {"high", "low"}.issubset(contrasts_by_stratum)
        else {}
    )
    summary["primary_contrast"] = "long_minus_medium"
    return summary
