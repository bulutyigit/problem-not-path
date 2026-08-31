"""Phase report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reasonbench.phases import PhaseStatus, build_artifact_manifest
from reasonbench.storage import read_json, sha256_file, write_text_atomic


def _markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return f"`{json.dumps(value, sort_keys=True)}`"
    return str(value)


def _automatic_sections(root: Path) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    resolved_configs = [path for path in all_files if path.name == "resolved_config.json"]
    if resolved_configs:
        rows = [
            "| Experiment | Model | Revision | Resolved config SHA-256 | Shards |",
            "|---|---|---|---|---:|",
        ]
        unique: dict[str, tuple[dict[str, Any], int]] = {}
        for path in resolved_configs:
            config_hash = sha256_file(path)
            data = read_json(path)
            previous = unique.get(config_hash)
            unique[config_hash] = (data, 1 if previous is None else previous[1] + 1)
        for config_hash, (data, shard_count) in sorted(unique.items()):
            rows.append(
                f"| {data.get('experiment_id')} | {data.get('model', {}).get('model_id')} | "
                f"`{data.get('model', {}).get('revision')}` | `{config_hash}` | {shard_count} |"
            )
        sections.append(("Accepted configurations", "\n".join(rows)))
    selections = [path for path in all_files if path.name == "problem_selection_manifest.json"]
    if selections:
        rows = [
            "| Experiment | Config hash | Dataset bundle hash | Seeds | Expected trajectories |",
            "|---|---|---|---|---:|",
        ]
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for path in selections:
            data = read_json(path)
            key = (str(data.get("experiment_id")), str(data.get("dataset_bundle_sha256")))
            unique.setdefault(key, data)
        for _, data in sorted(unique.items()):
            rows.append(
                "| "
                f"{data.get('experiment_id')} | `{data.get('config_hash', 'not recorded')}` | "
                f"`{data.get('dataset_bundle_sha256', 'not recorded')}` | "
                f"`{data.get('seeds', [])}` | {data.get('expected_trajectories')} |"
            )
        runtime_manifests = [path for path in all_files if path.name == "runtime_manifest.json"]
        source_revisions = sorted(
            {
                str(read_json(path).get("source_tree_sha256"))
                for path in runtime_manifests
                if read_json(path).get("source_tree_sha256")
            }
        )
        if source_revisions:
            rows.append(
                "| Source tree revision(s) | "
                + " | ".join(f"`{revision}`" for revision in source_revisions)
                + " |  |  |"
            )
        sections.append(("Generation provenance", "\n".join(rows)))
    validation_path = root / "generation_validation.json"
    if validation_path.exists():
        validation = read_json(validation_path)
        sections.append(
            (
                "Execution diagnostics",
                "\n".join(
                    [
                        f"- Expected trajectories: {validation['expected_trajectories']}",
                        f"- Completed trajectories: {validation['completed_trajectories']}",
                        f"- Completion rate: {validation['completion_rate']:.3%}",
                        f"- Unique problems: {validation['problem_count']}",
                        f"- Duplicate run IDs: {validation['duplicate_run_ids']}",
                        f"- Aligned signal rows: {validation['signal_rows']}",
                    ]
                ),
            )
        )
    input_manifests = [
        path
        for path in all_files
        if path.name.endswith("input_manifest.json")
        if path.name != "artifacts_manifest.json"
    ]
    if input_manifests:
        lines = [f"- `{path.relative_to(root)}`: `{sha256_file(path)}`" for path in input_manifests]
        sections.append(("Input manifests", "\n".join(lines)))
    figures = [
        path.relative_to(root) for path in all_files if path.suffix.lower() in {".png", ".pdf"}
    ]
    if figures:
        sections.append(
            (
                "Figures",
                "\n".join(f"- `{path}`" for path in figures),
            )
        )
    error_files = [path for path in all_files if path.suffix == ".json" and "errors" in path.parts]
    sections.append(
        (
            "Anomalies and limitations",
            "\n".join(
                [
                    f"- Recorded generation errors: {len(error_files)}",
                    "- Generated reasoning traces are observable outputs, not guaranteed faithful internal reasoning.",
                    "- Primary predictors exclude final-answer tokens.",
                    "- Confidence intervals cluster complete trajectories by problem.",
                    "- Cross-model conclusions use normalized summaries and never raw hidden coordinates.",
                ]
            ),
        )
    )
    return sections


def render_phase_report(
    phase_directory: str | Path,
    status: PhaseStatus,
    sections: list[tuple[str, str]] | None = None,
) -> Path:
    """Write a compact, reviewable Markdown phase report."""

    status.validate()
    root = Path(phase_directory)
    lines = [
        f"# {status.phase_id.replace('_', ' ').title()} Report",
        "",
        "## Decision summary",
        "",
        f"- Technical status: **{status.technical_status}**",
        f"- Scientific outcome: **{status.scientific_outcome}**",
        f"- Next decision: **{status.next_decision}**",
        f"- Completed at: `{status.completed_at}`",
        "",
        status.summary or "No summary was provided.",
        "",
    ]
    if status.metrics:
        lines.extend(["## Key metrics", "", "| Metric | Value |", "|---|---:|"])
        for key, value in sorted(status.metrics.items()):
            lines.append(f"| {key} | {_markdown_value(value)} |")
        lines.append("")
    if status.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in status.warnings)
        lines.append("")
    combined_sections = _automatic_sections(root) + list(sections or [])
    for title, body in combined_sections:
        lines.extend([f"## {title}", "", body.strip(), ""])
    report_path = write_text_atomic(root / "phase_report.md", "\n".join(lines).rstrip() + "\n")
    status.write(root)
    build_artifact_manifest(root)
    return report_path
