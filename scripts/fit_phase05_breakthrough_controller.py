#!/usr/bin/env python
"""Fit and freeze the Phase 5 breakthrough-aware compute controller.

MATH is development-only here.  Forecast and response predictions used for
policy selection are problem-grouped out-of-fold predictions.  The final
estimators are then refit on all development rows and may be applied once to
an untouched external cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from reasonbench.evaluation.breakthrough_controller import (
    ARMS,
    CONTROLLER_FEATURES,
    BreakthroughAwareController,
    _arm_estimator,
    artifact_digest,
    fit_arm_models,
    select_compute_penalty,
)
from reasonbench.evaluation.metrics import classification_metrics
from reasonbench.evaluation.predictor import fit_predictor
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic

MODEL_KEYS = ("gemma4_e4b_mlx_4bit", "ministral3_3b_mlx_4bit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-validation", type=Path, required=True)
    parser.add_argument("--horizon-table", type=Path, required=True)
    parser.add_argument("--eventual-success-table", type=Path, required=True)
    parser.add_argument("--prefix-features", type=Path, required=True)
    parser.add_argument("--development-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forecast-token", type=int, default=512)
    parser.add_argument("--forecast-horizon", type=int, default=512)
    parser.add_argument(
        "--feature-set",
        default="early_blocks",
        help=(
            "Forecast feature set. The frozen default is the low-dimensional "
            "early_blocks summary; the 300+-column early_full set is retained "
            "only for explicitly requested sensitivity runs."
        ),
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--minimum-labeled-trajectories", type=int, default=30)
    parser.add_argument("--maximum-accuracy-gap", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def _intersection_power_gate(
    tables: dict[str, tuple[pd.DataFrame, str]],
    *,
    folds: int,
    output_dir: Path,
) -> dict:
    """Authoritative power gate on the post-intersection fitting tables.

    The labeling-cohort count upstream is only a sanity check; the quantity
    that determines identifiability is the per-class problem-group support of
    each target after the frozen filters and the response-pair intersection.
    """

    report: dict[str, dict] = {}
    underpowered: list[str] = []
    for name, (frame, target) in tables.items():
        groups_per_class = (
            frame.groupby(target)["problem_id"].nunique().astype(int).to_dict()
            if not frame.empty
            else {}
        )
        entry = {
            "rows": int(len(frame)),
            "problems": int(frame["problem_id"].nunique()) if not frame.empty else 0,
            "target": target,
            "positives": int(frame[target].astype(bool).sum()) if not frame.empty else 0,
            "problem_groups_per_class": {str(k): v for k, v in groups_per_class.items()},
            "required_groups_per_class": folds,
        }
        if len(groups_per_class) < 2 or min(groups_per_class.values()) < folds:
            underpowered.append(name)
        report[name] = entry
    status = {
        "status": (
            "underpowered_after_intersection" if underpowered else "power_gate_passed"
        ),
        "underpowered_tables": underpowered,
        "tables": report,
    }
    write_json_atomic(ensure_directory(output_dir) / "phase05_fit_status.json", status)
    if underpowered:
        raise RuntimeError(
            "underpowered_after_intersection: "
            + "; ".join(
                f"{name} target={report[name]['target']} rows={report[name]['rows']} "
                f"problem_groups_per_class={report[name]['problem_groups_per_class']} "
                f"(need >= {folds} per class)"
                for name in underpowered
            )
            + " — see phase05_fit_status.json"
        )
    return status


def _check_groups(frame: pd.DataFrame, target: str, folds: int) -> None:
    if frame[target].nunique() < 2:
        raise RuntimeError(f"{target} has only one class")
    groups_per_class = frame.groupby(target)["problem_id"].nunique()
    if int(groups_per_class.min()) < folds:
        raise RuntimeError(
            f"{target} needs at least {folds} problem groups in each class; "
            f"observed {groups_per_class.to_dict()}"
        )


def _crossfit_forecast(
    label_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
    *,
    target: str,
    feature_set: str,
    folds: int,
    seed: int,
) -> tuple[object, pd.Series, dict]:
    _check_groups(label_rows, target, folds)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    raw_predictions = pd.Series(np.nan, index=score_rows.index, dtype=float)
    fold_assignments = pd.Series(-1, index=score_rows.index, dtype=int)
    audit: list[dict] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(label_rows, label_rows[target], groups=label_rows["problem_id"])
    ):
        train = label_rows.iloc[train_index].copy()
        heldout_labels = label_rows.iloc[test_index].copy()
        heldout_problems = set(heldout_labels["problem_id"].astype(str))
        heldout_scores = score_rows[
            score_rows["problem_id"].astype(str).isin(heldout_problems)
        ]
        if train[target].nunique() < 2:
            raise RuntimeError(f"Fold {fold} training rows have one {target} class")
        predictor = fit_predictor(
            label_rows,
            train=train,
            validation=label_rows.iloc[:0],
            feature_set=feature_set,
            model_name="logistic_regression",
            seed=seed + fold,
            target_column=target,
        )
        raw_predictions.loc[heldout_scores.index] = predictor.predict_proba(heldout_scores)
        fold_assignments.loc[heldout_scores.index] = fold
        audit.append(
            {
                "fold": fold,
                "train_problems": int(train["problem_id"].nunique()),
                "heldout_problems": int(len(heldout_problems)),
                "heldout_scored_rows": int(len(heldout_scores)),
            }
        )
    if raw_predictions.isna().any() or (fold_assignments < 0).any():
        missing = score_rows.loc[raw_predictions.isna(), "problem_id"].astype(str).unique()
        raise RuntimeError(f"Cross-fit forecast did not cover score rows: {missing[:10].tolist()}")

    # Cross-fit the one-dimensional Platt calibrator as a second layer. A
    # row's outcome is used neither by its base predictor nor by its calibrator.
    raw_by_run = dict(zip(score_rows["run_id"].astype(str), raw_predictions, strict=True))
    fold_by_run = dict(zip(score_rows["run_id"].astype(str), fold_assignments, strict=True))
    label_raw = label_rows["run_id"].astype(str).map(raw_by_run).astype(float)
    label_folds = label_rows["run_id"].astype(str).map(fold_by_run).astype(int)
    calibrated_predictions = pd.Series(np.nan, index=score_rows.index, dtype=float)
    for fold in range(folds):
        calibration_train = label_folds.ne(fold)
        if label_rows.loc[calibration_train, target].nunique() < 2:
            raise RuntimeError(f"Fold {fold} calibration training has one {target} class")
        train_raw = np.clip(label_raw[calibration_train].to_numpy(float), 1e-6, 1 - 1e-6)
        train_logits = np.log(train_raw / (1.0 - train_raw)).reshape(-1, 1)
        calibrator = LogisticRegression(C=0.1, max_iter=2000, random_state=seed + fold)
        calibrator.fit(train_logits, label_rows.loc[calibration_train, target].astype(int))
        heldout = fold_assignments.eq(fold)
        heldout_raw = np.clip(raw_predictions[heldout].to_numpy(float), 1e-6, 1 - 1e-6)
        heldout_logits = np.log(heldout_raw / (1.0 - heldout_raw)).reshape(-1, 1)
        calibrated_predictions.loc[heldout] = calibrator.predict_proba(heldout_logits)[:, 1]
    if calibrated_predictions.isna().any():
        raise RuntimeError("Cross-fitted calibration did not cover every score row")
    final = fit_predictor(
        label_rows,
        train=label_rows,
        validation=label_rows.iloc[:0],
        feature_set=feature_set,
        model_name="logistic_regression",
        seed=seed + 10_000,
        target_column=target,
    )
    all_raw = np.clip(label_raw.to_numpy(float), 1e-6, 1 - 1e-6)
    all_logits = np.log(all_raw / (1.0 - all_raw)).reshape(-1, 1)
    final_calibrator = LogisticRegression(C=0.1, max_iter=2000, random_state=seed + 20_000)
    final_calibrator.fit(all_logits, label_rows[target].astype(int))
    final.calibrator = final_calibrator
    labeled_predictions = label_rows[["run_id", "problem_id", target]].merge(
        score_rows[["run_id"]].assign(probability=calibrated_predictions.to_numpy()),
        on="run_id",
        how="inner",
        validate="one_to_one",
    )
    metrics = classification_metrics(
        labeled_predictions[target].astype(int).to_numpy(),
        labeled_predictions["probability"].to_numpy(float),
    )
    return final, calibrated_predictions, {
        "folds": audit,
        "oof_metrics": metrics,
        "calibration": "nested_cross_fitted_platt_scaling",
    }


def _crossfit_arm_response(
    frame: pd.DataFrame,
    *,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    # A stable per-source target stratifies the split without looking at any
    # one branch in isolation. All branches and arms of a problem remain in a
    # single fold.
    source_target = frame.groupby(["problem_id", "source_run_id"], as_index=False).agg(
        any_long_correct=("long_correct", "max")
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    output = pd.DataFrame(index=frame.index)
    for fold, (train_source_index, test_source_index) in enumerate(
        splitter.split(
            source_target,
            source_target["any_long_correct"],
            groups=source_target["problem_id"],
        )
    ):
        train_sources = set(source_target.iloc[train_source_index]["source_run_id"].astype(str))
        test_sources = set(source_target.iloc[test_source_index]["source_run_id"].astype(str))
        train = frame[frame["source_run_id"].astype(str).isin(train_sources)]
        test = frame[frame["source_run_id"].astype(str).isin(test_sources)]
        for arm_index, arm in enumerate(ARMS):
            target = train[f"{arm}_correct"].astype(int)
            if target.nunique() < 2:
                raise RuntimeError(f"Fold {fold}, arm {arm} has one training class")
            estimator = _arm_estimator(train, seed=seed + fold * 10 + arm_index)
            estimator.fit(train[list(CONTROLLER_FEATURES)], target)
            output.loc[test.index, f"predicted_correct_{arm}"] = estimator.predict_proba(
                test[list(CONTROLLER_FEATURES)]
            )[:, 1]
    if output.isna().any().any():
        raise RuntimeError("OOF arm-response predictions are incomplete")
    return output


def main() -> None:
    args = parse_args()
    validation = read_json(args.probe_validation)
    if not validation.get("valid") or validation.get("stage") != "labeling_cohort":
        raise RuntimeError("Phase 5 requires a valid full labeling_cohort, not pilot labels")
    if int(validation.get("completed_trajectories", 0)) < args.minimum_labeled_trajectories:
        raise RuntimeError("Full breakthrough cohort is below the pre-specified power gate")

    horizon = pd.read_parquet(args.horizon_table)
    eventual = pd.read_parquet(args.eventual_success_table)
    features = pd.read_parquet(args.prefix_features)
    pairs = pd.read_parquet(args.development_pairs)
    horizon = horizon[
        horizon["model_key"].isin(MODEL_KEYS)
        & horizon["forecast_token"].eq(args.forecast_token)
        & horizon["forecast_horizon"].eq(args.forecast_horizon)
    ].copy()
    eventual = eventual[
        eventual["model_key"].isin(MODEL_KEYS)
        & eventual["forecast_token"].eq(args.forecast_token)
    ].copy()
    features = features[
        features["model_key"].isin(MODEL_KEYS)
        & features["observed_token_count"].ge(args.forecast_token)
    ].copy()
    pairs = pairs[pairs["model_key"].isin(MODEL_KEYS)].copy()
    # The controller-development cohort is the intersection for which we have
    # both continuation-derived breakthrough supervision and matched compute
    # responses. Other Phase 4B rows must not appear as unlabelled pseudo-OOF
    # observations.
    development_run_ids = set(pairs["source_run_id"].astype(str))
    features = features[features["run_id"].astype(str).isin(development_run_ids)].copy()
    eventual = eventual[eventual["run_id"].astype(str).isin(development_run_ids)].copy()
    horizon = horizon[horizon["run_id"].astype(str).isin(development_run_ids)].copy()
    for name, frame in {
        "horizon": horizon,
        "eventual": eventual,
        "features": features,
        "development pairs": pairs,
    }.items():
        if frame.empty:
            raise RuntimeError(f"Phase 5 {name} table is empty after the frozen filters")
    if features["run_id"].duplicated().any():
        raise RuntimeError("Prefix feature table has duplicate run_id values")
    power_status = _intersection_power_gate(
        {
            "horizon": (horizon, "breakthrough_within_horizon"),
            "eventual": (eventual, "correct"),
        },
        folds=args.folds,
        output_dir=args.output_dir,
    )

    breakthrough_model, breakthrough_oof, breakthrough_audit = _crossfit_forecast(
        horizon,
        features,
        target="breakthrough_within_horizon",
        feature_set=args.feature_set,
        folds=args.folds,
        seed=args.seed,
    )
    success_model, success_oof, success_audit = _crossfit_forecast(
        eventual,
        features,
        target="correct",
        feature_set=args.feature_set,
        folds=args.folds,
        seed=args.seed + 1_000,
    )
    scores = features[["run_id", "problem_id", "model_key", "level"]].copy()
    scores["eventual_success_probability"] = success_oof.to_numpy()
    scores["breakthrough_probability_within_512"] = breakthrough_oof.to_numpy()
    if "uncertainty_score" not in pairs.columns:
        raise RuntimeError("Development pairs must preserve the frozen U512 score")
    development = pairs.merge(
        scores.rename(columns={"run_id": "source_run_id"}),
        on=["source_run_id", "problem_id", "model_key"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_score"),
    )
    if development["source_run_id"].nunique() != pairs["source_run_id"].nunique():
        raise RuntimeError("OOF forecasts do not cover every development trajectory")
    if "level_score" in development:
        development["level"] = development["level"].fillna(development["level_score"])
    arm_oof = _crossfit_arm_response(development, folds=args.folds, seed=args.seed + 2_000)
    token_costs = {
        arm: float(development[f"{arm}_total_generated_tokens"].mean())
        for arm in ARMS
    }
    penalty, candidates = select_compute_penalty(
        development,
        arm_oof,
        token_costs=token_costs,
        maximum_accuracy_gap=args.maximum_accuracy_gap,
    )
    arm_models = fit_arm_models(development, seed=args.seed + 3_000)
    u512_thresholds = {
        "short_max": float(development["uncertainty_score"].quantile(1 / 3)),
        "medium_max": float(development["uncertainty_score"].quantile(2 / 3)),
    }
    controller = BreakthroughAwareController(
        arm_models=arm_models,
        token_costs=token_costs,
        compute_penalty=penalty,
    )

    output = ensure_directory(args.output_dir)
    forecasters_path = output / "phase05_forecasters.joblib"
    controller_path = output / "phase05_controller.joblib"
    joblib.dump(
        {
            "breakthrough": breakthrough_model,
            "eventual_success": success_model,
            "forecast_token": args.forecast_token,
            "forecast_horizon": args.forecast_horizon,
        },
        forecasters_path,
    )
    controller.dump(controller_path)
    scores.to_parquet(output / "development_oof_forecasts.parquet", index=False)
    pd.concat([development.reset_index(drop=True), arm_oof.reset_index(drop=True)], axis=1).to_parquet(
        output / "development_oof_controller_rows.parquet", index=False
    )
    candidates.to_csv(output / "compute_penalty_grid.csv", index=False)
    payload = {
        "schema_version": "phase05_breakthrough_controller_v1",
        "status": "frozen_for_external_evaluation",
        "models": list(MODEL_KEYS),
        "development_dataset": "MATH",
        "external_evaluation_dataset": "HARP",
        "forecast_token": args.forecast_token,
        "forecast_horizon": args.forecast_horizon,
        "feature_set": args.feature_set,
        "controller_features": list(CONTROLLER_FEATURES),
        "compute_penalty": penalty,
        "maximum_accuracy_gap": args.maximum_accuracy_gap,
        "token_costs": token_costs,
        "u512_only_ablation_thresholds": u512_thresholds,
        "crossfit_folds": args.folds,
        "intersection_power": power_status,
        "breakthrough_forecast": breakthrough_audit,
        "eventual_success_forecast": success_audit,
        "input_sha256": {
            "probe_validation": sha256_file(args.probe_validation),
            "horizon_table": sha256_file(args.horizon_table),
            "eventual_success_table": sha256_file(args.eventual_success_table),
            "prefix_features": sha256_file(args.prefix_features),
            "development_pairs": sha256_file(args.development_pairs),
        },
        "forecasters_sha256": sha256_file(forecasters_path),
        "controller_sha256": sha256_file(controller_path),
        "outcome_blind_external_freeze": True,
        "harp_outcomes_opened": False,
        "notes": [
            "MATH OOF estimates are development diagnostics, not confirmatory results.",
            "HARP must not tune features, thresholds, calibration, or compute penalty.",
            "The controller predicts compute response; U512 alone is only an ablation.",
        ],
    }
    payload["artifact_digest"] = artifact_digest(payload)
    write_json_atomic(output / "phase05_frozen_policy.json", payload)
    print(json.dumps({"policy": str(output / "phase05_frozen_policy.json"), "penalty": penalty}, indent=2))


if __name__ == "__main__":
    main()
