#!/usr/bin/env python
"""Run post-generation CPU analysis locally, without a GPU or Hugging Face token."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from reasonbench.constants import PHASE4_DENSE_PREFIXES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("phase_02", "phase_03", "phase_04", "phase_04b", "phase_04d")
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--expected-trajectories",
        type=int,
        help="Override the phase default when running a registered reduced-seed profile.",
    )
    parser.add_argument("--breakthrough-labels", type=Path)
    parser.add_argument("--phase04b-features-dir", type=Path)
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="CPU workers for independent trajectory feature extraction.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    artifacts = args.artifacts_root.resolve()
    phase_root = artifacts / args.phase
    if args.phase == "phase_04d":
        tables = phase_root / "tables"
        analysis = phase_root / "analysis"
        labels = args.breakthrough_labels or artifacts / "phase_04c" / "breakthrough_labels.parquet"
        phase04b_features = args.phase04b_features_dir or artifacts / "phase_04b" / "features"
        commands = [
            [
                "build_breakthrough_tables.py",
                "--features-dir",
                phase04b_features,
                "--labels",
                labels,
                "--output-dir",
                tables,
            ],
            [
                "evaluate_breakthrough_forecasts.py",
                "--horizon-table",
                tables / "breakthrough_horizon_table.parquet",
                "--hazard-table",
                tables / "breakthrough_hazard_table.parquet",
                "--eventual-success-table",
                tables / "eventual_success_table.parquet",
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ],
        ]
        for command in commands:
            subprocess.run(
                [sys.executable, "-u", root / "scripts" / command[0], *map(str, command[1:])],
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-u",
                root / "scripts" / "create_phase_report.py",
                "--phase",
                args.phase,
                "--run-dir",
                phase_root,
                "--summary-json",
                analysis / "breakthrough_forecast_summary.json",
            ],
            check=True,
        )
        return
    generation_root = phase_root / "generation"
    run_dir = generation_root / "gemma4" if args.phase == "phase_02" else generation_root
    features = phase_root / "features"
    analysis = phase_root / "analysis"
    expected_trajectories = args.expected_trajectories or (
        400
        if args.phase == "phase_02"
        else 600
        if args.phase == "phase_04b"
        else 300
    )
    commands = [
        [
            "validate_generation.py",
            "--run-dir",
            run_dir,
            "--output-dir",
            phase_root,
            "--expected-trajectories",
            str(expected_trajectories),
            "--minimum-completion-rate",
            "0.98",
        ],
        [
            "extract_features.py",
            "--run-dir",
            run_dir,
            "--output-dir",
            features,
            "--workers",
            str(args.feature_workers),
        ],
    ]
    if args.phase in {"phase_02", "phase_03", "phase_04", "phase_04b"}:
        prefix_lengths = (
            PHASE4_DENSE_PREFIXES
            if args.phase == "phase_04b"
            else (16, 32, 64, 128, 256, 512)
            if args.phase == "phase_04"
            else (16, 32, 64, 128, 256, 512, 1024, 2048)
        )
        commands[-1].extend(
            value for prefix in prefix_lengths for value in ("--prefix-length", str(prefix))
        )
    if args.phase == "phase_04b":
        commands[0].extend(
            (
                "--expected-panel-manifest",
                artifacts / "shared" / "datasets_v2" / "dataset_manifest.json",
            )
        )
    if args.phase == "phase_02":
        commands.append(
            [
                "analyze_difficulty_dynamics.py",
                "--features",
                features / "features_full.parquet",
                "--features-dir",
                features,
                "--run-dir",
                run_dir,
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
        )
    elif args.phase == "phase_03":
        commands.append(
            [
                "analyze_cross_model_difficulty.py",
                "--features",
                features / "features_full.parquet",
                "--features-dir",
                features,
                "--run-dir",
                generation_root,
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
                "--seeds-per-problem",
                "1",
            ]
        )
        commands.append(
            [
                "analyze_hidden_pca.py",
                "--run-dir",
                generation_root,
                "--output-dir",
                analysis / "hidden_pca",
            ]
        )
    elif args.phase == "phase_04":
        commands.append(
            [
                "analyze_cap_extension.py",
                "--baseline-features",
                artifacts / "phase_03" / "features" / "features_full.parquet",
                "--extended-features",
                features / "features_full.parquet",
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
        )
        early_command = [
            "evaluate_early_prediction.py",
            "--features-dir",
            features,
            "--output-dir",
            analysis,
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
            "--workers",
            str(min(2, args.feature_workers)),
        ]
        early_command.extend(
            value
            for prefix in (16, 32, 64, 128, 256, 512)
            for value in ("--prefix-length", str(prefix))
        )
        commands.append(early_command)
        commands.append(
            [
                "evaluate_early_length.py",
                "--features-dir",
                features,
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
        )
        commands.append(
            [
                "analyze_phase04_dynamics.py",
                "--features-dir",
                features,
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
        )
        commands.append(
            [
                "analyze_phase04_relative_spectral.py",
                "--generation-dir",
                generation_root,
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
        )
        commands.append(
            [
                "finalize_phase_04.py",
                "--cap-summary",
                analysis / "cap_extension_summary.json",
                "--early-summary",
                analysis / "early_summary.json",
                "--length-summary",
                analysis / "length_prediction_summary.json",
                "--dynamics-summary",
                analysis / "dynamics_summary.json",
                "--output",
                analysis / "phase_summary.json",
            ]
        )
    else:
        # Phase 4b is a descriptive diagnostic panel.  The count-only audit
        # records whether a future classifier would be viable, but sparse
        # failures never suppress the pre-registered trajectory analyses.
        commands.append(
            [
                "audit_phase04b_power.py",
                "--features",
                features / "features_full.parquet",
                "--output-dir",
                analysis,
                "--target-column",
                "correct",
            ]
        )
    for command in commands:
        subprocess.run(
            [sys.executable, "-u", root / "scripts" / command[0], *map(str, command[1:])],
            check=True,
        )
    if args.phase == "phase_04b":
        power_audit = analysis / "phase04b_power_audit.json"
        phase04b_commands = [
            [
                "analyze_phase04_dynamics.py",
                "--features-dir",
                features,
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ],
            [
                "analyze_phase04_relative_spectral.py",
                "--generation-dir",
                generation_root,
                "--output-dir",
                analysis,
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ],
            [
                "analyze_hidden_pca.py",
                "--run-dir",
                generation_root,
                "--output-dir",
                analysis / "hidden_pca",
            ],
        ]
        for command in phase04b_commands:
            subprocess.run(
                [sys.executable, "-u", root / "scripts" / command[0], *map(str, command[1:])],
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-u",
                root / "scripts" / "finalize_phase_04b.py",
                "--power-audit",
                power_audit,
                "--dynamics-summary",
                analysis / "dynamics_summary.json",
                "--output",
                analysis / "phase_summary.json",
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-u",
            root / "scripts" / "create_phase_report.py",
            "--phase",
            args.phase,
            "--run-dir",
            phase_root,
            "--summary-json",
            analysis / "phase_summary.json",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
