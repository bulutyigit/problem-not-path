"""Breakthrough labels and leakage-safe longitudinal prediction tables.

The expensive experiment observes continuation success only at sparse token
anchors.  The resulting event time is therefore interval censored.  This
module keeps that uncertainty explicit instead of silently turning every
dense prefix into an exact event-time observation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnchorProbe:
    """Continuation outcomes from one exact generated prefix."""

    anchor: int
    successes: int
    continuations: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.continuations if self.continuations else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "success_rate": self.success_rate}


@dataclass(frozen=True)
class BreakthroughLabel:
    """Stable-success event label derived from ordered sparse probes."""

    event_observed: bool
    interval_lower: int
    interval_upper: int | None
    event_time_proxy: int | None
    censoring_time: int
    stable_anchor: int | None
    stability_anchor: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_breakthrough_label(
    probes: Iterable[AnchorProbe],
    *,
    threshold: float = 0.75,
) -> BreakthroughLabel:
    """Return the earliest threshold crossing stable at the next probe.

    A crossing at anchor ``a_j`` implies only that the transition occurred in
    ``(a_{j-1}, a_j]``.  The next observed anchor must also pass the threshold.
    If no stable pair exists, the trajectory is right-censored at the final
    probed anchor.
    """

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    ordered = sorted(probes, key=lambda probe: probe.anchor)
    if not ordered:
        raise ValueError("At least one completed anchor probe is required")
    if len({probe.anchor for probe in ordered}) != len(ordered):
        raise ValueError("Anchor probes must have unique token positions")
    if any(probe.continuations < 1 for probe in ordered):
        raise ValueError("Every anchor must contain at least one continuation")
    if any(not 0 <= probe.successes <= probe.continuations for probe in ordered):
        raise ValueError("Anchor successes must lie in [0, continuations]")

    for index in range(len(ordered) - 1):
        current = ordered[index]
        following = ordered[index + 1]
        if current.success_rate >= threshold and following.success_rate >= threshold:
            lower = ordered[index - 1].anchor if index else 0
            return BreakthroughLabel(
                event_observed=True,
                interval_lower=lower,
                interval_upper=current.anchor,
                event_time_proxy=current.anchor,
                censoring_time=following.anchor,
                stable_anchor=current.anchor,
                stability_anchor=following.anchor,
            )
    final_anchor = ordered[-1].anchor
    return BreakthroughLabel(
        event_observed=False,
        interval_lower=final_anchor,
        interval_upper=None,
        event_time_proxy=None,
        censoring_time=final_anchor,
        stable_anchor=None,
        stability_anchor=None,
    )


def horizon_outcome(
    label: BreakthroughLabel,
    *,
    prefix: int,
    horizon: int,
) -> int | None:
    """Label a forecast horizon only when interval censoring permits it.

    ``None`` means that the observed interval cannot distinguish a positive
    from a negative outcome at this prefix and horizon.
    """

    if prefix < 0 or horizon <= 0:
        raise ValueError("prefix must be non-negative and horizon must be positive")
    horizon_end = prefix + horizon
    if label.event_observed:
        assert label.interval_upper is not None
        # The event is known to have happened before or at the prefix, so this
        # row is no longer in the future-event risk set.
        if label.interval_upper <= prefix:
            return None
        if label.interval_lower >= prefix and label.interval_upper <= horizon_end:
            return 1
        if label.interval_lower >= horizon_end:
            return 0
        return None
    return 0 if label.censoring_time >= horizon_end else None


def _label_from_row(row: pd.Series) -> BreakthroughLabel:
    upper = row.get("interval_upper")
    event_time = row.get("event_time_proxy")
    stable_anchor = row.get("stable_anchor")
    stability_anchor = row.get("stability_anchor")
    return BreakthroughLabel(
        event_observed=bool(row["event_observed"]),
        interval_lower=int(row["interval_lower"]),
        interval_upper=None if pd.isna(upper) else int(upper),
        event_time_proxy=None if pd.isna(event_time) else int(event_time),
        censoring_time=int(row["censoring_time"]),
        stable_anchor=None if pd.isna(stable_anchor) else int(stable_anchor),
        stability_anchor=None if pd.isna(stability_anchor) else int(stability_anchor),
    )


def build_longitudinal_tables(
    features_directory: str | Path,
    labels: pd.DataFrame,
    *,
    horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join dense prefix features to sparse, interval-censored event labels.

    Returns a horizon-classification table and a discrete-time hazard table.
    Rows remain keyed by ``run_id`` and retain the immutable problem-level
    research split. No random row-level split is performed here.
    """

    required = {
        "run_id",
        "event_observed",
        "interval_lower",
        "interval_upper",
        "event_time_proxy",
        "censoring_time",
        "stable_anchor",
        "stability_anchor",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Breakthrough labels are missing columns: {sorted(missing)}")
    if labels["run_id"].duplicated().any():
        raise ValueError("Breakthrough labels must contain one row per run_id")

    root = Path(features_directory)
    feature_paths = sorted(root.glob("features_prefix_*.parquet"))
    if not feature_paths:
        raise ValueError(f"No dense prefix feature tables were found in {root}")
    joined_rows: list[pd.DataFrame] = []
    label_columns = sorted(required - {"run_id"})
    for path in feature_paths:
        try:
            prefix = int(path.stem.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid prefix feature filename: {path.name}") from exc
        frame = pd.read_parquet(path)
        if frame["run_id"].duplicated().any():
            raise ValueError(f"Duplicate run_id values in {path}")
        merged = frame.merge(
            labels[["run_id", *label_columns]],
            on="run_id",
            how="inner",
            validate="one_to_one",
        )
        merged["forecast_token"] = prefix
        # Only prefixes that were actually observed and remained in the event
        # risk set are eligible. This prevents padding a short trajectory into
        # a later pseudo-observation.
        merged = merged[merged["observed_token_count"] >= prefix].copy()
        joined_rows.append(merged)
    if not joined_rows:
        raise ValueError("No feature rows matched breakthrough labels")
    panel = pd.concat(joined_rows, ignore_index=True).sort_values(
        ["run_id", "forecast_token"]
    ).reset_index(drop=True)
    panel["forecast_time_bin"] = panel["forecast_token"].map(lambda value: f"t_{int(value)}")

    horizon_rows: list[pd.DataFrame] = []
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon <= 0:
            raise ValueError("Forecast horizons must be positive")
        block = panel.copy()
        outcomes = [
            horizon_outcome(_label_from_row(row), prefix=int(row["forecast_token"]), horizon=horizon)
            for _, row in block.iterrows()
        ]
        block["forecast_horizon"] = horizon
        block["breakthrough_within_horizon"] = pd.array(outcomes, dtype="Int64")
        block = block[block["breakthrough_within_horizon"].notna()].copy()
        block["breakthrough_within_horizon"] = block[
            "breakthrough_within_horizon"
        ].astype(int)
        horizon_rows.append(block)
    horizon_table = pd.concat(horizon_rows, ignore_index=True) if horizon_rows else pd.DataFrame()

    hazard_rows: list[pd.DataFrame] = []
    for _, group in panel.groupby("run_id", sort=False):
        group = group.sort_values("forecast_token").copy()
        label = _label_from_row(group.iloc[0])
        if label.event_observed:
            assert label.interval_upper is not None
            # Only prefixes at or before the interval's left boundary are
            # certainly pre-breakthrough. The final such row predicts an event
            # over (prefix, interval_upper]; later dense rows are ambiguous and
            # are excluded instead of receiving a post-event positive label.
            eligible = group[group["forecast_token"] <= label.interval_lower].copy()
            if eligible.empty:
                continue
            eligible["breakthrough_in_bin"] = 0
            eligible["hazard_interval_end"] = eligible["forecast_token"].shift(-1)
            final_index = eligible.index[-1]
            eligible.loc[final_index, "breakthrough_in_bin"] = 1
            eligible.loc[final_index, "hazard_interval_end"] = label.interval_upper
        else:
            # Every retained interval is fully observed without an event. The
            # final interval ends at the right-censoring time.
            eligible = group[group["forecast_token"] < label.censoring_time].copy()
            if eligible.empty:
                continue
            eligible["breakthrough_in_bin"] = 0
            eligible["hazard_interval_end"] = eligible["forecast_token"].shift(-1)
            eligible.loc[eligible.index[-1], "hazard_interval_end"] = label.censoring_time
        eligible["hazard_interval_width"] = (
            eligible["hazard_interval_end"] - eligible["forecast_token"]
        )
        eligible = eligible[eligible["hazard_interval_width"] > 0].copy()
        hazard_rows.append(eligible)
    hazard = pd.concat(hazard_rows, ignore_index=True) if hazard_rows else panel.iloc[:0].copy()
    hazard["time_log1p"] = np.log1p(hazard["forecast_token"].astype(float))

    keys = ["problem_id", "run_id", "forecast_token"]
    return (
        horizon_table.sort_values(keys + ["forecast_horizon"]).reset_index(drop=True),
        hazard.sort_values(keys).reset_index(drop=True),
    )
