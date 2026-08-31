"""Problem-level seed instability and cross-block trajectory associations."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _normalized_binary_entropy(success_rate: float) -> float:
    if success_rate <= 0.0 or success_rate >= 1.0:
        return 0.0
    return float(
        -(
            success_rate * math.log(success_rate)
            + (1.0 - success_rate) * math.log(1.0 - success_rate)
        )
        / math.log(2.0)
    )


def problem_seed_instability(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Aggregate repeated seeds into one independent row per problem."""

    required = {"problem_id", "seed", "correct", "level", "category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Seed-instability input is missing columns: {missing}")
    features = [column for column in feature_columns if column in frame]
    rows: list[dict[str, object]] = []
    for problem_id, group in frame.groupby("problem_id", sort=True):
        success_rate = float(group["correct"].astype(float).mean())
        row: dict[str, object] = {
            "problem_id": problem_id,
            "level": int(group["level"].iloc[0]),
            "category": str(group["category"].iloc[0]),
            "seeds": int(group["seed"].nunique()),
            "correct_seeds": int(group["correct"].astype(bool).sum()),
            "success_rate": success_rate,
            "failure_rate": 1.0 - success_rate,
            "seed_instability": _normalized_binary_entropy(success_rate),
        }
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            row[f"mean__{feature}"] = float(values.mean())
            row[f"sd__{feature}"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _control_matrix(frame: pd.DataFrame, controls: Iterable[str]) -> np.ndarray:
    available = [column for column in controls if column in frame]
    if not available:
        return np.ones((len(frame), 1), dtype=float)
    encoded = pd.get_dummies(frame[available], drop_first=False, dtype=float)
    encoded = encoded.apply(pd.to_numeric, errors="coerce")
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    return np.column_stack([np.ones(len(encoded), dtype=float), encoded.to_numpy(dtype=float)])


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(controls, values, rcond=None)
    return values - controls @ coefficients


def residualized_spearman(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    controls: Iterable[str] = ("level", "category", "mean__trajectory_token_count"),
) -> float:
    """Spearman association after linearly removing preregistered controls."""

    columns = [left, right, *[column for column in controls if column in frame]]
    working = frame[columns].copy()
    working[left] = pd.to_numeric(working[left], errors="coerce")
    working[right] = pd.to_numeric(working[right], errors="coerce")
    working = working.dropna(subset=[left, right])
    if len(working) < 4:
        return math.nan
    left_values = working[left].to_numpy(dtype=float)
    right_values = working[right].to_numpy(dtype=float)
    if np.std(left_values) < 1e-12 or np.std(right_values) < 1e-12:
        return math.nan
    matrix = _control_matrix(working, controls)
    left_residual = _residualize(left_values, matrix)
    right_residual = _residualize(right_values, matrix)
    if np.std(left_residual) < 1e-12 or np.std(right_residual) < 1e-12:
        return math.nan
    return float(spearmanr(left_residual, right_residual).statistic)


def bootstrap_residualized_associations(
    frame: pd.DataFrame,
    target: str,
    features: Iterable[str],
    *,
    controls: Iterable[str] = ("level", "category", "mean__trajectory_token_count"),
    repetitions: int = 2000,
    seed: int = 20260728,
) -> pd.DataFrame:
    """Bootstrap problem rows for controlled seed-instability associations."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []
    for feature in features:
        if feature not in frame:
            continue
        point = residualized_spearman(frame, target, feature, controls=controls)
        draws: list[float] = []
        for _ in range(repetitions):
            sampled = frame.iloc[rng.integers(0, len(frame), size=len(frame))].reset_index(
                drop=True
            )
            value = residualized_spearman(sampled, target, feature, controls=controls)
            if np.isfinite(value):
                draws.append(value)
        rows.append(
            {
                "target": target,
                "feature": feature,
                "association": point,
                "ci_low": float(np.quantile(draws, 0.025)) if draws else math.nan,
                "ci_high": float(np.quantile(draws, 0.975)) if draws else math.nan,
                "problems": len(frame),
            }
        )
    return pd.DataFrame(rows)


def residualized_cross_block_correlations(
    frame: pd.DataFrame,
    left_features: Iterable[str],
    right_features: Iterable[str],
    *,
    controls: Iterable[str] = ("level", "category", "mean__trajectory_token_count"),
) -> pd.DataFrame:
    """Return a controlled geometry-by-spectral Spearman matrix."""

    left = [feature for feature in left_features if feature in frame]
    right = [feature for feature in right_features if feature in frame]
    matrix = pd.DataFrame(index=left, columns=right, dtype=float)
    for left_feature in left:
        for right_feature in right:
            matrix.loc[left_feature, right_feature] = residualized_spearman(
                frame,
                left_feature,
                right_feature,
                controls=controls,
            )
    return matrix
