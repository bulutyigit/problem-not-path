#!/usr/bin/env python
"""Visualize hidden-state reasoning paths in a shared, train-fitted PCA space.

The visualizations intentionally encode *path progress* with colour.  Terminal
correctness is represented by separate panels rather than competing with the
temporal colour scale.  This makes temporal direction, start states, and end
states legible without smoothing or reordering the observed hidden states.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from reasonbench.storage import ensure_directory, read_json, write_json_atomic

CORRECT_COLOUR = "#2878b5"
INCORRECT_COLOUR = "#b44a4a"
PROGRESS_CMAP = mpl.colormaps["viridis"]


@dataclass(frozen=True)
class HiddenTrajectory:
    """One captured thinking trajectory and the metadata needed to audit it."""

    directory: Path
    run_id: str
    split: str
    correct: bool
    level: int | None
    category: str
    token_indices: np.ndarray
    hidden_states: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-training-states", type=int, default=5000)
    parser.add_argument("--maximum-plotted-trajectories", type=int, default=12)
    parser.add_argument("--landmarks", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _load_trajectory(directory: Path) -> HiddenTrajectory | None:
    hidden_path = directory / "hidden_states.npz"
    metrics_path = directory / "token_metrics.parquet"
    if not hidden_path.exists() or not metrics_path.exists():
        return None
    metadata = read_json(directory / "metadata.json")
    token_frame = pd.read_parquet(metrics_path)
    segment = "thinking" if (token_frame["segment"] == "thinking").any() else "solution"
    included = set(
        token_frame.loc[token_frame["segment"] == segment, "token_index"].astype(int)
    )
    with np.load(hidden_path) as payload:
        token_indices = payload["token_indices"].astype(int)
        hidden_states = payload["hidden_states"].astype(np.float32)
    mask = np.asarray([index in included for index in token_indices], dtype=bool)
    if mask.sum() < 2:
        return None
    return HiddenTrajectory(
        directory=directory,
        run_id=str(metadata["run_id"]),
        split=str(metadata["research_split"]),
        correct=bool(metadata["verification"]["correct"]),
        level=int(metadata["level"]) if metadata.get("level") is not None else None,
        category=str(metadata.get("category", "unknown")),
        token_indices=token_indices[mask],
        hidden_states=hidden_states[mask],
    )


def _training_matrix(
    trajectories: list[HiddenTrajectory], maximum_states: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Sample training states without letting early filesystem rows dominate PCA."""

    if not trajectories:
        raise ValueError("No training hidden states were available for PCA")
    batches: list[np.ndarray] = []
    quota = max(1, maximum_states // len(trajectories))
    remaining = maximum_states
    contributed = 0
    for trajectory in sorted(trajectories, key=lambda item: item.run_id):
        hidden = _normalise_rows(trajectory.hidden_states)
        take = min(len(hidden), quota, remaining)
        if take:
            if len(hidden) > take:
                hidden = hidden[rng.choice(len(hidden), size=take, replace=False)]
            batches.append(hidden)
            contributed += 1
            remaining -= take
    # Spend unused capacity fairly: each additional pass gives at most one
    # further equal-sized slice per trajectory before moving to the next.
    while remaining > 0:
        added = False
        for trajectory in sorted(trajectories, key=lambda item: item.run_id):
            hidden = _normalise_rows(trajectory.hidden_states)
            take = min(len(hidden), quota, remaining)
            if not take:
                continue
            if len(hidden) > take:
                hidden = hidden[rng.choice(len(hidden), size=take, replace=False)]
            batches.append(hidden)
            remaining -= take
            added = True
            if remaining == 0:
                break
        if not added:
            break
    if not batches:
        raise ValueError("No training hidden states were available for PCA")
    # PCA's randomized solver is numerically more stable on CPU in float64.
    return np.concatenate(batches, axis=0).astype(np.float64, copy=False), contributed


def _relative_progress(token_indices: np.ndarray) -> np.ndarray:
    span = int(token_indices[-1] - token_indices[0])
    if span <= 0:
        return np.linspace(0.0, 1.0, len(token_indices))
    return (token_indices - token_indices[0]) / span


def _resample_path(coordinates: np.ndarray, progress: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Linearly interpolate captured states; no spline geometry is invented."""
    unique, unique_indices = np.unique(progress, return_index=True)
    coordinates = coordinates[unique_indices]
    if len(unique) == 1:
        return np.repeat(coordinates, len(landmarks), axis=0)
    return np.column_stack(
        [np.interp(landmarks, unique, coordinates[:, component]) for component in range(2)]
    )


def _selected_trajectories(
    trajectories: list[HiddenTrajectory], maximum: int
) -> list[HiddenTrajectory]:
    """Deterministically spread representative held-out paths across outcomes and levels."""
    selected: list[HiddenTrajectory] = []
    by_outcome: dict[bool, list[HiddenTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.split == "test":
            by_outcome[trajectory.correct].append(trajectory)
    quota = max(1, maximum // 2)
    for outcome in (True, False):
        candidates = sorted(
            by_outcome[outcome],
            key=lambda item: (
                item.level is None,
                item.level if item.level is not None else 999,
                len(item.token_indices),
                item.run_id,
            ),
        )
        if not candidates:
            continue
        # Pick evenly across the sorted level/length ordering rather than the first runs.
        locations = np.linspace(0, len(candidates) - 1, min(quota, len(candidates)), dtype=int)
        selected.extend(candidates[int(location)] for location in locations)
    return selected


def _add_coloured_path(axis: plt.Axes, coordinates: np.ndarray, progress: np.ndarray, alpha: float) -> None:
    points = coordinates.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap=PROGRESS_CMAP,
        norm=mpl.colors.Normalize(0.0, 1.0),
        array=(progress[:-1] + progress[1:]) / 2,
        linewidth=1.8,
        alpha=alpha,
        zorder=2,
    )
    axis.add_collection(collection)
    axis.scatter(*coordinates[0], marker="P", s=52, color="#222222", zorder=4)
    axis.scatter(
        *coordinates[-1], marker="*", s=98, color="#222222", edgecolor="white", linewidth=0.45, zorder=5
    )


def _set_shared_limits(axes: np.ndarray, coordinate_sets: list[np.ndarray]) -> None:
    all_coordinates = np.concatenate(coordinate_sets, axis=0)
    lower = all_coordinates.min(axis=0)
    upper = all_coordinates.max(axis=0)
    padding = np.maximum((upper - lower) * 0.07, 0.05)
    for axis in axes.flat:
        axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
        axis.set_xlabel("PCA component 1")
        axis.set_ylabel("PCA component 2")


def _draw_representative_paths(
    model_key: str,
    trajectories: list[tuple[HiddenTrajectory, np.ndarray]],
    output_path: Path,
) -> dict[str, int]:
    # A grid prevents temporal paths from becoming an unreadable spaghetti plot.
    columns = min(4, max(1, len(trajectories)))
    rows = int(np.ceil(len(trajectories) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.05 * columns, 3.5 * rows), sharex=True, sharey=True, squeeze=False
    )
    counts = {
        "correct": sum(trajectory.correct for trajectory, _ in trajectories),
        "incorrect": sum(not trajectory.correct for trajectory, _ in trajectories),
    }
    display_paths = sorted(
        trajectories,
        key=lambda item: (
            not item[0].correct,
            item[0].level is None,
            item[0].level if item[0].level is not None else 999,
            item[0].run_id,
        ),
    )
    grid = np.linspace(0.0, 1.0, 49)
    for axis, (trajectory, coordinates) in zip(axes.flat, display_paths, strict=False):
        path = _resample_path(coordinates, _relative_progress(trajectory.token_indices), grid)
        _add_coloured_path(axis, path, grid, alpha=0.92)
        outcome = "correct" if trajectory.correct else "incorrect"
        colour = CORRECT_COLOUR if trajectory.correct else INCORRECT_COLOUR
        axis.set_title(
            f"{outcome}; L{trajectory.level}; {int(trajectory.token_indices[-1]) + 1} tokens",
            fontsize=9,
            color=colour,
        )
    for axis in axes.flat[len(display_paths) :]:
        axis.set_visible(False)
    _set_shared_limits(axes, [coordinates for _, coordinates in trajectories])
    figure.subplots_adjust(top=0.84, bottom=0.15, hspace=0.42, wspace=0.26)
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=PROGRESS_CMAP),
        cax=figure.add_axes((0.2, 0.06, 0.6, 0.018)),
        orientation="horizontal",
    )
    colourbar.set_label("Relative thinking progress (early → late)")
    figure.suptitle(
        f"{model_key}: held-out hidden-state paths, one trajectory per panel\n"
        "Train-fitted PCA; start = P; end = star; colour = temporal order",
        y=0.99,
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)
    return counts


def _draw_aggregate_flow(
    model_key: str,
    trajectories: list[tuple[HiddenTrajectory, np.ndarray]],
    landmarks: int,
    output_path: Path,
) -> None:
    grid = np.linspace(0.0, 1.0, landmarks)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    all_coordinate_sets = [coordinates for _, coordinates in trajectories]
    for axis, outcome in zip(axes, (True, False), strict=True):
        subset = [(trajectory, coordinates) for trajectory, coordinates in trajectories if trajectory.correct == outcome]
        label = "Correct final answer" if outcome else "Incorrect final answer"
        axis.set_title(f"{label} aggregate flow (n={len(subset)})", color=CORRECT_COLOUR if outcome else INCORRECT_COLOUR)
        if not subset:
            axis.text(0.5, 0.5, "No held-out trajectories in this outcome", ha="center", va="center", transform=axis.transAxes)
            continue
        resampled = np.stack(
            [_resample_path(coordinates, _relative_progress(trajectory.token_indices), grid) for trajectory, coordinates in subset]
        )
        for path in resampled:
            axis.plot(path[:, 0], path[:, 1], color="#7a7a7a", alpha=0.13, linewidth=0.7, zorder=1)
        median_path = np.median(resampled, axis=0)
        _add_coloured_path(axis, median_path, grid, alpha=1.0)
        # Ellipses at quartiles show spread without inventing a continuous density.
        for landmark_index in (0, landmarks // 4, landmarks // 2, 3 * landmarks // 4, landmarks - 1):
            points = resampled[:, landmark_index, :]
            if len(points) < 3:
                continue
            covariance = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            ellipse = mpl.patches.Ellipse(
                xy=median_path[landmark_index],
                width=2 * np.sqrt(eigenvalues[0]),
                height=2 * np.sqrt(eigenvalues[1]),
                angle=angle,
                edgecolor=PROGRESS_CMAP(grid[landmark_index]),
                facecolor="none",
                linewidth=0.9,
                alpha=0.72,
                zorder=3,
            )
            axis.add_patch(ellipse)
    _set_shared_limits(axes, all_coordinate_sets)
    figure.subplots_adjust(top=0.80, bottom=0.16, wspace=0.14)
    colourbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=PROGRESS_CMAP),
        cax=figure.add_axes((0.25, 0.065, 0.5, 0.023)),
        orientation="horizontal",
    )
    colourbar.set_label("Relative thinking progress (early → late)")
    figure.suptitle(
        f"{model_key}: aggregate temporal flow (train-fitted PCA)\n"
        "Grey = individual resampled paths; coloured = coordinate-wise median; ellipses = 1 SD at five time points",
        y=0.99,
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def _draw_path_kinematics(
    model_key: str,
    trajectories: list[tuple[HiddenTrajectory, np.ndarray]],
    landmarks: int,
    output_path: Path,
) -> None:
    grid = np.linspace(0.0, 1.0, landmarks)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    for axis, outcome in zip(axes, (True, False), strict=True):
        subset = [(trajectory, coordinates) for trajectory, coordinates in trajectories if trajectory.correct == outcome]
        label = "Correct final answer" if outcome else "Incorrect final answer"
        colour = CORRECT_COLOUR if outcome else INCORRECT_COLOUR
        axis.set_title(f"{label} (n={len(subset)})", color=colour)
        if not subset:
            axis.text(0.5, 0.5, "No held-out trajectories in this outcome", ha="center", va="center", transform=axis.transAxes)
            continue
        resampled = np.stack(
            [_resample_path(coordinates, _relative_progress(trajectory.token_indices), grid) for trajectory, coordinates in subset]
        )
        radial = np.linalg.norm(resampled - resampled[:, :1, :], axis=2)
        steps = np.linalg.norm(np.diff(resampled, axis=1), axis=2)
        cumulative = np.concatenate([np.zeros((len(subset), 1)), np.cumsum(steps, axis=1)], axis=1)
        for values, name, style in ((radial, "displacement from start", "-"), (cumulative, "cumulative path length", "--")):
            median = np.median(values, axis=0)
            lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
            axis.plot(grid, median, label=name, color=colour, linestyle=style, linewidth=2)
            axis.fill_between(grid, lower, upper, color=colour, alpha=0.14)
        axis.set_xlabel("Relative thinking progress")
        axis.set_ylabel("Distance in two-PC projection")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, loc="upper left")
    figure.suptitle(
        f"{model_key}: hidden-path movement over time\n"
        "Projection-space descriptors; compare only within a model because PCA axes are model-specific",
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def _coordinate_table(
    model_key: str, trajectories: list[tuple[HiddenTrajectory, np.ndarray]]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trajectory, coordinates in trajectories:
        rows.append(
            pd.DataFrame(
                {
                    "model_key": model_key,
                    "run_id": trajectory.run_id,
                    "split": trajectory.split,
                    "correct": trajectory.correct,
                    "level": trajectory.level,
                    "category": trajectory.category,
                    "token_index": trajectory.token_indices,
                    "relative_thinking_progress": _relative_progress(trajectory.token_indices),
                    "pca_component_1": coordinates[:, 0],
                    "pca_component_2": coordinates[:, 1],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    by_model: dict[str, list[HiddenTrajectory]] = defaultdict(list)
    markers = sorted({marker for run_directory in args.run_dir for marker in run_directory.rglob("complete.json")})
    for marker in markers:
        trajectory = _load_trajectory(marker.parent)
        if trajectory is not None:
            metadata = read_json(marker.parent / "metadata.json")
            by_model[str(metadata["model_key"])].append(trajectory)
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {"models": {}, "temporal_encoding": {"line_colour": "relative thinking progress", "start_marker": "P", "end_marker": "star", "aggregate": "coordinate-wise median over linearly resampled captured states"}}
    coordinate_tables: list[pd.DataFrame] = []
    for model_key, trajectories in sorted(by_model.items()):
        training = [trajectory for trajectory in trajectories if trajectory.split == "train"]
        testing = [trajectory for trajectory in trajectories if trajectory.split == "test"]
        matrix, training_contributors = _training_matrix(
            training, args.maximum_training_states, rng
        )
        pca = PCA(n_components=2, svd_solver="randomized", random_state=args.seed)
        pca.fit(matrix)
        transformed = [
            (trajectory, pca.transform(_normalise_rows(trajectory.hidden_states).astype(np.float64)))
            for trajectory in testing
        ]
        if not transformed:
            continue
        selected_ids = {item.run_id for item in _selected_trajectories(trajectories, args.maximum_plotted_trajectories)}
        selected = [(trajectory, coordinates) for trajectory, coordinates in transformed if trajectory.run_id in selected_ids]
        # Retain the established filename as the temporal representative-path view.
        representative_path = output_dir / f"{model_key}_hidden_pca.png"
        selected_counts = _draw_representative_paths(model_key, selected, representative_path)
        flow_path = output_dir / f"{model_key}_hidden_pca_aggregate_flow.png"
        _draw_aggregate_flow(model_key, transformed, args.landmarks, flow_path)
        kinematics_path = output_dir / f"{model_key}_hidden_pca_kinematics.png"
        _draw_path_kinematics(model_key, transformed, args.landmarks, kinematics_path)
        coordinate_tables.append(_coordinate_table(model_key, transformed))
        model_summary = {
            "training_states": len(matrix),
            "training_trajectories_available": len(training),
            "training_trajectories_contributed": training_contributors,
            "held_out_test_trajectories": len(testing),
            "plotted_test_trajectories": len(selected),
            "plotted_test_outcomes": selected_counts,
            "held_out_test_outcomes": {
                "correct": sum(trajectory.correct for trajectory in testing),
                "incorrect": sum(not trajectory.correct for trajectory in testing),
            },
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "figures": {
                "representative_temporal_paths": str(representative_path),
                "aggregate_temporal_flow": str(flow_path),
                "projection_kinematics": str(kinematics_path),
            },
            "cross_model_raw_coordinates_compared": False,
            "interpretation_caveat": "The first two PCA components are a low-variance projection and are descriptive; temporal paths should be checked against quantitative features and repeated embeddings.",
        }
        summary["models"][model_key] = model_summary
    if not summary["models"]:
        raise ValueError("No complete model trajectories were found")
    coordinates = pd.concat(coordinate_tables, ignore_index=True)
    coordinates_path = output_dir / "hidden_pca_test_path_coordinates.parquet"
    coordinates.to_parquet(coordinates_path, index=False)
    summary["held_out_coordinate_table"] = str(coordinates_path)
    write_json_atomic(output_dir / "hidden_pca_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
