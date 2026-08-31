#!/usr/bin/env python
"""Download, sample, split, and persist the benchmark problems."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from reasonbench.constants import DEFAULT_SPLIT_SEED
from reasonbench.datasets import (
    assign_research_splits,
    build_problem_sample,
    load_problem_records,
    write_problem_bundle,
)
from reasonbench.datasets.loader import DATASET_SOURCES
from reasonbench.storage import ensure_directory, write_json_atomic
from reasonbench.verification import verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gsm8k-size", type=int, default=200)
    parser.add_argument("--math-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--gsm8k-revision")
    parser.add_argument("--math-revision")
    return parser.parse_args()


def _resolve_revision(
    repository: str,
    requested_revision: str | None,
) -> tuple[str, str | None]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to pin immutable dataset revisions"
        ) from exc
    information = HfApi().dataset_info(repo_id=repository, revision=requested_revision)
    if not information.sha:
        raise RuntimeError(f"Could not resolve an immutable revision for {repository}")
    card_data = information.card_data
    license_name = getattr(card_data, "license", None) if card_data is not None else None
    return str(information.sha), license_name


def _audit_references(records: list, dataset: str) -> tuple[dict, list[dict]]:
    failures: list[str] = []
    methods: Counter[str] = Counter()
    rows: list[dict] = []
    for record in records:
        generated = f"Final answer: \\boxed{{{record.reference_answer}}}"
        result = verify_answer(generated, record.reference_answer, dataset)
        methods[result.verification_method] += 1
        if not result.correct:
            failures.append(record.problem_id)
        rows.append(
            {
                "dataset": dataset,
                "problem_id": record.problem_id,
                "reference_answer": record.reference_answer,
                "correct": result.correct,
                "extraction_status": result.extraction_status,
                "verification_method": result.verification_method,
                "error": result.error,
            }
        )
    return (
        {
            "records": len(records),
            "self_verification_failures": len(failures),
            "failed_problem_ids": failures,
            "verification_methods": dict(methods),
        },
        rows,
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    specifications = [
        ("gsm8k", args.gsm8k_size, args.gsm8k_revision),
        ("math", args.math_size, args.math_revision),
    ]
    manifest: dict = {"seed": args.seed, "datasets": {}}
    audit_rows: list[dict] = []
    for dataset, sample_size, requested_revision in specifications:
        repository = DATASET_SOURCES[dataset]["repository"]
        resolved_revision, license_name = _resolve_revision(
            repository,
            requested_revision,
        )
        all_records = load_problem_records(dataset, revision=resolved_revision)
        sample = build_problem_sample(all_records, sample_size=sample_size, seed=args.seed)
        assigned = assign_research_splits(sample, seed=args.seed)
        data_path, split_path = write_problem_bundle(
            assigned,
            output_directory=output_dir,
            name=f"{dataset}_sample",
        )
        split_counts = Counter(record.research_split for record in assigned)
        level_counts = Counter(record.level for record in assigned if record.level is not None)
        audit_summary, dataset_audit_rows = _audit_references(assigned, dataset)
        audit_rows.extend(dataset_audit_rows)
        manifest["datasets"][dataset] = {
            "source_rows": len(all_records),
            "sample_rows": len(assigned),
            "source_repository": assigned[0].source_repository,
            "source_split": assigned[0].source_split,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "license": license_name,
            "data_path": str(data_path),
            "split_path": str(split_path),
            "split_counts": dict(split_counts),
            "level_counts": {str(key): value for key, value in level_counts.items()},
            "reference_audit": audit_summary,
        }
    pd.DataFrame(audit_rows).to_parquet(
        output_dir / "verifier_audit.parquet",
        index=False,
    )
    write_json_atomic(output_dir / "dataset_manifest.json", manifest)
    failed = sum(
        details["reference_audit"]["self_verification_failures"]
        for details in manifest["datasets"].values()
    )
    if failed:
        raise SystemExit(
            f"Dataset preparation completed, but {failed} reference answers failed self-verification."
        )
    print(f"Dataset bundle written to {output_dir}")


if __name__ == "__main__":
    main()
