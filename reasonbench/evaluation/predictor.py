"""Leakage-resistant correctness prediction models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reasonbench.evaluation.metrics import clustered_bootstrap

IDENTIFIER_COLUMNS = {
    "run_id",
    "experiment_id",
    "phase_id",
    "problem_id",
    "research_split",
    "correct",
    "normal_completion",
    "noncompletion",
    "wrong_completion",
    "needs_intervention",
    "parse_status",
    "finish_reason",
    "boundary_status",
    "prefix_length",
}

DIFFICULTY_COLUMNS = {
    "dataset",
    "level",
    "category",
    "problem_character_count",
    "problem_token_proxy_count",
    "problem_numeric_count",
    "problem_operator_count",
    "problem_equation_count",
}

LENGTH_COLUMNS = {
    "trajectory_token_count",
    "assigned_reasoning_budget",
}

PREDICTIVE_SIGNAL_PREFIXES = (
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
)

TRANSITION_SIGNAL_PREFIXES = (
    "successive_kl_divergence",
    "successive_js_divergence",
)

# Frozen low-dimensional block summary for severely label-limited fits
# (Phase 5 controller forecasts). Three summaries per signal block; the full
# early_* families remain available for the better-powered Phase 4 analyses.
EARLY_BLOCKS_SUMMARY_COLUMNS = (
    # predictive-ambiguity levels
    "normalized_entropy_mean",
    "surprisal_mean",
    "top1_top2_probability_margin_mean",
    # predictive dynamics
    "normalized_entropy_std",
    "normalized_entropy_robust_slope",
    "surprisal_max_rise",
    # successive-distribution transitions
    "successive_js_divergence_mean",
    "successive_js_divergence_std",
    "successive_kl_divergence_robust_slope",
    # hidden-state geometry
    "relative_l2_step_mean",
    "cosine_drift_mean",
    "hidden_norm_robust_slope",
    # spectral
    "spectral_successive_js_divergence_entropy",
    "spectral_successive_js_divergence_centroid",
    "spectral_successive_js_divergence_low_energy_ratio",
)


def feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    """Resolve a named ablation feature set."""

    available = set(frame.columns) - IDENTIFIER_COLUMNS
    difficulty = available & DIFFICULTY_COLUMNS
    length = available & LENGTH_COLUMNS
    distribution = {
        column
        for column in available
        if column.startswith(PREDICTIVE_SIGNAL_PREFIXES)
    }
    distribution_means = {column for column in distribution if column.endswith("_mean")}
    distribution_dynamic = distribution - distribution_means
    transition = {
        column
        for column in available
        if column.startswith(TRANSITION_SIGNAL_PREFIXES)
    }
    geometry = {
        column
        for column in available
        if column.startswith(
            (
                "geometry_",
                "hidden_norm_",
                "relative_l2_step_",
                "cosine_drift_",
            )
        )
    }
    spectral = {column for column in available if column.startswith("spectral_")}
    early_known = {
        column
        for column in (
            "model_key",
            "assigned_reasoning_budget",
            "observed_token_count",
            "forecast_token",
            "forecast_time_bin",
            "time_log1p",
            "hazard_interval_width",
        )
        if column in available
    }
    early_baseline = difficulty | early_known
    sets = {
        "constant": set(),
        "difficulty": difficulty,
        "length": difficulty | length,
        # Preserve the original ablation semantics: mean confidence contains
        # only level summaries, while the dynamic block contains slopes,
        # dispersion, quantiles, and extrema.
        "mean_confidence": difficulty | length | distribution_means,
        "dynamic_uncertainty": difficulty | length | distribution_dynamic,
        "geometry": difficulty | length | geometry,
        "spectral": difficulty | length | spectral,
        "full_without_spectral": difficulty | length | distribution | transition | geometry,
        "full": difficulty | length | distribution | transition | geometry | spectral,
        "early_baseline": early_baseline,
        "early_confidence": early_baseline | distribution_means,
        "early_dynamic_uncertainty": early_baseline | distribution_dynamic,
        "early_transition": early_baseline | transition,
        "early_geometry": early_baseline | geometry,
        "early_spectral": early_baseline | spectral,
        "early_full_without_spectral": early_baseline | distribution | transition | geometry,
        "early_full": early_baseline | distribution | transition | geometry | spectral,
        # Backward-compatible Phase 6 definition.
        "early": early_baseline | distribution | transition | geometry,
    }
    if feature_set == "early_blocks":
        missing = sorted(set(EARLY_BLOCKS_SUMMARY_COLUMNS) - available)
        if missing:
            raise ValueError(
                f"early_blocks requires frozen summary columns absent from the frame: {missing}"
            )
        return sorted(early_baseline | set(EARLY_BLOCKS_SUMMARY_COLUMNS))
    transfer = sets["full"] - {
        column
        for column in sets["full"]
        if column in {"assigned_reasoning_budget", "level"}
        or pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
    }
    sets["transfer"] = transfer
    if feature_set not in sets:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {sorted(sets)}")
    return sorted(sets[feature_set])


def _preprocessor(frame: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    categorical = [
        column
        for column in columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _estimator(
    frame: pd.DataFrame,
    columns: list[str],
    model_name: str,
    seed: int,
) -> Any:
    if model_name == "constant":
        return DummyClassifier(strategy="prior")
    if model_name == "logistic_regression":
        classifier: Any = LogisticRegression(
            # Fixed-prefix, model-specific analyses can have relatively few
            # observations and many correlated trajectory summaries.  Stronger
            # L2 regularization prevents quasi-separation from turning a
            # handful of bootstrap draws into numerically degenerate scores.
            C=0.1,
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        )
    elif model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return Pipeline(
        [
            ("preprocess", _preprocessor(frame, columns)),
            ("classifier", classifier),
        ]
    )


@dataclass
class FittedPredictor:
    """A train-fitted estimator with optional validation-only calibration."""

    estimator: Any
    feature_columns: list[str]
    calibrator: LogisticRegression | None
    calibration_nan_to_num_interventions: int = 0
    prediction_nan_to_num_interventions: dict[str, int] = field(default_factory=dict)

    def _matrix(self, frame: pd.DataFrame) -> Any:
        if self.feature_columns:
            return frame[self.feature_columns]
        return np.ones((len(frame), 1), dtype=np.float64)

    @staticmethod
    def _sanitize(probabilities: np.ndarray) -> tuple[np.ndarray, int]:
        values = np.asarray(probabilities, dtype=float)
        interventions = int((~np.isfinite(values)).sum())
        return np.clip(np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0), interventions

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities, count = self._sanitize(self.estimator.predict_proba(self._matrix(frame))[:, 1])
        self.prediction_nan_to_num_interventions["raw_probability"] = (
            self.prediction_nan_to_num_interventions.get("raw_probability", 0) + count
        )
        if self.calibrator is None:
            return probabilities
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        calibrated, count = self._sanitize(self.calibrator.predict_proba(logits)[:, 1])
        self.prediction_nan_to_num_interventions["calibrated_probability"] = (
            self.prediction_nan_to_num_interventions.get("calibrated_probability", 0) + count
        )
        return calibrated


def fit_predictor(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_set: str,
    model_name: str,
    seed: int,
    target_column: str = "correct",
) -> FittedPredictor:
    """Fit preprocessing on train and probability calibration on validation only."""

    columns = feature_columns(frame, feature_set)
    effective_model = "constant" if feature_set == "constant" else model_name
    if target_column not in frame.columns:
        raise ValueError(f"Unknown target column: {target_column}")
    if effective_model != "constant" and train[target_column].nunique() < 2:
        raise ValueError(f"Training data must contain both {target_column} classes")
    estimator = _estimator(frame, columns, model_name=effective_model, seed=seed)
    train_matrix: Any = train[columns] if columns else np.ones((len(train), 1), dtype=np.float64)
    estimator.fit(train_matrix, train[target_column].astype(int))
    calibrator: LogisticRegression | None = None
    calibration_interventions = 0
    if (
        not validation.empty
        and validation[target_column].nunique() == 2
        and effective_model != "constant"
    ):
        validation_matrix: Any = (
            validation[columns] if columns else np.ones((len(validation), 1), dtype=np.float64)
        )
        raw, calibration_interventions = FittedPredictor._sanitize(
            estimator.predict_proba(validation_matrix)[:, 1]
        )
        raw = np.clip(raw, 1e-6, 1 - 1e-6)
        logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        calibrator = LogisticRegression(C=0.1, random_state=seed, max_iter=2000)
        calibrator.fit(logits, validation[target_column].astype(int))
    return FittedPredictor(
        estimator=estimator,
        feature_columns=columns,
        calibrator=calibrator,
        calibration_nan_to_num_interventions=calibration_interventions,
    )


@dataclass
class EvaluationResult:
    feature_set: str
    model_name: str
    feature_columns: list[str]
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    pipeline: FittedPredictor
    calibration_applied: bool
    target_column: str
    nan_to_num_interventions: dict[str, int]


def evaluate_one(
    frame: pd.DataFrame,
    feature_set: str,
    model_name: str = "logistic_regression",
    bootstrap_repetitions: int = 2000,
    seed: int = 20260728,
    target_column: str = "correct",
) -> EvaluationResult:
    """Fit on train, reserve validation for inspection, and evaluate on test."""

    train = frame[frame["research_split"] == "train"].copy()
    validation = frame[frame["research_split"] == "validation"].copy()
    test = frame[frame["research_split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Both train and test splits must contain trajectories")
    if train["problem_id"].isin(test["problem_id"]).any():
        raise ValueError("Problem leakage detected between train and test")
    columns = feature_columns(frame, feature_set)
    pipeline = fit_predictor(
        frame,
        train=train,
        validation=validation,
        feature_set=feature_set,
        model_name=model_name,
        seed=seed,
        target_column=target_column,
    )
    probability = pipeline.predict_proba(test)
    columns_to_save = ["run_id", "problem_id", "dataset", "model_key", "correct"]
    if target_column != "correct":
        columns_to_save.append(target_column)
    predictions = test[columns_to_save].copy()
    predictions["probability"] = probability
    metrics = clustered_bootstrap(
        predictions,
        repetitions=bootstrap_repetitions,
        seed=seed,
        target_column=target_column,
    )
    return EvaluationResult(
        feature_set=feature_set,
        model_name=model_name,
        feature_columns=columns,
        metrics=metrics,
        predictions=predictions,
        pipeline=pipeline,
        calibration_applied=pipeline.calibrator is not None,
        target_column=target_column,
        nan_to_num_interventions={
            "calibration": pipeline.calibration_nan_to_num_interventions,
            **pipeline.prediction_nan_to_num_interventions,
        },
    )


def evaluate_feature_sets(
    frame: pd.DataFrame,
    feature_sets: Iterable[str],
    model_names: Iterable[str] = (
        "logistic_regression",
        "hist_gradient_boosting",
        "random_forest",
    ),
    bootstrap_repetitions: int = 2000,
    seed: int = 20260728,
    target_column: str = "correct",
) -> list[EvaluationResult]:
    """Evaluate a complete ablation grid."""

    results: list[EvaluationResult] = []
    for feature_set in feature_sets:
        names = ("constant",) if feature_set == "constant" else tuple(model_names)
        for model_name in names:
            results.append(
                evaluate_one(
                    frame,
                    feature_set=feature_set,
                    model_name=model_name,
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=seed,
                    target_column=target_column,
                )
            )
    return results
