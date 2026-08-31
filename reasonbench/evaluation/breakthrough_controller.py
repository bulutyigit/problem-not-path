"""Frozen breakthrough-aware adaptive-compute controller utilities.

Phase 5 treats MATH as development data and the external cohort as a one-shot
evaluation.  The controller predicts arm-specific correctness from two early
forecasts and the block-balanced U512 index, then maximizes a transparent
accuracy-minus-compute utility.  No terminal or continuation outcome is used
when materializing an evaluation-cohort action.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARMS = ("short", "medium", "long")
CONTROLLER_FEATURES = (
    "model_key",
    "level",
    "uncertainty_score",
    "eventual_success_probability",
    "breakthrough_probability_within_512",
)


def artifact_digest(payload: dict) -> str:
    """Hash a JSON-serializable artifact while excluding its digest field."""

    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "artifact_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_artifact_digest(payload: dict) -> bool:
    declared = str(payload.get("artifact_digest", ""))
    return bool(declared) and declared == artifact_digest(payload)


def _arm_estimator(frame: pd.DataFrame, *, seed: int) -> Pipeline:
    categorical = ["model_key"]
    numeric = [column for column in CONTROLLER_FEATURES if column not in categorical]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


@dataclass
class BreakthroughAwareController:
    """Three arm-response models plus one frozen compute penalty."""

    arm_models: dict[str, Pipeline]
    token_costs: dict[str, float]
    compute_penalty: float
    feature_columns: tuple[str, ...] = CONTROLLER_FEATURES

    def predict_arm_probabilities(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Controller input is missing columns: {sorted(missing)}")
        result = pd.DataFrame(index=frame.index)
        for arm in ARMS:
            model = self.arm_models[arm]
            values = model.predict_proba(frame[list(self.feature_columns)])[:, 1]
            result[f"predicted_correct_{arm}"] = np.clip(values, 0.0, 1.0)
        return result

    def choose(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_arm_probabilities(frame)
        maximum_cost = max(float(value) for value in self.token_costs.values())
        if maximum_cost <= 0:
            raise ValueError("At least one positive arm cost is required")
        utilities = pd.DataFrame(index=frame.index)
        for arm in ARMS:
            normalized_cost = float(self.token_costs[arm]) / maximum_cost
            utilities[f"utility_{arm}"] = (
                probabilities[f"predicted_correct_{arm}"]
                - self.compute_penalty * normalized_cost
            )
        # np.argmax is deliberately stable: exact utility ties prefer the
        # cheaper arm because ARMS is ordered short -> medium -> long.
        chosen_indices = np.argmax(utilities.to_numpy(dtype=float), axis=1)
        chosen = pd.Series([ARMS[index] for index in chosen_indices], index=frame.index)
        return pd.concat(
            [probabilities, utilities, chosen.rename("selected_arm")], axis=1
        )

    def dump(self, path) -> None:
        joblib.dump(self, path)


def fit_arm_models(frame: pd.DataFrame, *, seed: int) -> dict[str, Pipeline]:
    """Fit one simple correctness response model for each compute arm."""

    missing = set(CONTROLLER_FEATURES) - set(frame.columns)
    missing |= {f"{arm}_correct" for arm in ARMS} - set(frame.columns)
    if missing:
        raise ValueError(f"Development response table is missing: {sorted(missing)}")
    models: dict[str, Pipeline] = {}
    for index, arm in enumerate(ARMS):
        target = frame[f"{arm}_correct"].astype(int)
        if target.nunique() < 2:
            raise ValueError(f"Arm {arm!r} has only one correctness class")
        model = _arm_estimator(frame, seed=seed + index)
        model.fit(frame[list(CONTROLLER_FEATURES)], target)
        models[arm] = model
    return models


def select_compute_penalty(
    frame: pd.DataFrame,
    arm_probabilities: pd.DataFrame,
    *,
    token_costs: dict[str, float],
    maximum_accuracy_gap: float,
    penalty_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 101)),
) -> tuple[float, pd.DataFrame]:
    """Select the cheapest OOF policy within a frozen fixed-long accuracy gap."""

    if not 0 <= maximum_accuracy_gap < 1:
        raise ValueError("maximum_accuracy_gap must be in [0, 1)")
    maximum_cost = max(float(value) for value in token_costs.values())
    fixed_long_accuracy = float(frame["long_correct"].astype(float).mean())
    target_accuracy = fixed_long_accuracy - maximum_accuracy_gap
    rows: list[dict] = []
    for penalty in penalty_grid:
        utility = np.column_stack(
            [
                arm_probabilities[f"predicted_correct_{arm}"].to_numpy(float)
                - float(penalty) * float(token_costs[arm]) / maximum_cost
                for arm in ARMS
            ]
        )
        chosen_indices = np.argmax(utility, axis=1)
        actions = [ARMS[index] for index in chosen_indices]
        correctness = np.asarray(
            [bool(frame.iloc[row][f"{arm}_correct"]) for row, arm in enumerate(actions)],
            dtype=float,
        )
        costs = np.asarray([token_costs[arm] for arm in actions], dtype=float)
        rows.append(
            {
                "compute_penalty": float(penalty),
                "accuracy": float(correctness.mean()),
                "mean_token_cost": float(costs.mean()),
                "meets_accuracy_target": float(correctness.mean()) >= target_accuracy,
                **{
                    f"route_rate_{arm}": float(np.mean(np.asarray(actions) == arm))
                    for arm in ARMS
                },
            }
        )
    candidates = pd.DataFrame(rows)
    feasible = candidates[candidates["meets_accuracy_target"]]
    if feasible.empty:
        # Fail closed: freezing the cheapest infeasible policy would silently
        # abandon the frozen accuracy contract.
        raise RuntimeError(
            "no feasible compute penalty: no policy on the grid reaches the "
            f"frozen accuracy target {target_accuracy:.3f} "
            f"(fixed-long {fixed_long_accuracy:.3f} minus gap {maximum_accuracy_gap:.3f}); "
            "the policy must not be frozen"
        )
    ranked = feasible.sort_values(
        ["mean_token_cost", "accuracy", "compute_penalty"],
        ascending=[True, False, True],
    )
    return float(ranked.iloc[0]["compute_penalty"]), candidates
