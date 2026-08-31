#!/usr/bin/env python
"""Canonicalize, deduplicate, sample, and hash the untouched HARP cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from reasonbench.datasets.loader import ProblemRecord
from reasonbench.datasets.splits import read_problem_bundle, write_problem_bundle
from reasonbench.storage import ensure_directory, sha256_file, write_json_atomic

HARP_REPOSITORY = "https://github.com/aadityasingh/HARP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harp-jsonl-or-zip", type=Path, required=True)
    parser.add_argument("--math-bundle", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--selection-seed", type=int, default=20260822)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".jsonl")]
            if len(names) != 1:
                raise ValueError("HARP zip must contain exactly one JSONL file")
            text = archive.read(names[0]).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _shingles(text: str, width: int = 5) -> frozenset[tuple[str, ...]]:
    tokens = _normalized(text).split()
    if len(tokens) < width:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _solution(row: dict) -> str:
    keys = sorted(key for key in row if re.fullmatch(r"solution_\d+", str(key)))
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _canonicalize(rows: list[dict]) -> list[ProblemRecord]:
    records: list[ProblemRecord] = []
    for index, row in enumerate(rows):
        problem = str(row.get("problem") or "").strip()
        answer = str(row.get("answer") or "").strip()
        level = int(row.get("level"))
        subject = str(row.get("subject") or "").strip()
        if not problem or not answer or level not in range(1, 7) or not subject:
            raise ValueError(f"Invalid HARP row {index}: missing problem/answer/level/subject")
        source_identity = f"{row.get('year')}|{row.get('contest')}|{row.get('number')}|{problem}"
        digest = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:12]
        records.append(
            ProblemRecord(
                problem_id=f"harp_{index:05d}_{digest}",
                dataset="harp",
                source_repository=HARP_REPOSITORY,
                source_split="HARP.jsonl",
                source_index=index,
                problem=problem,
                reference_answer=answer,
                reference_solution=_solution(row),
                level=level,
                category=subject,
                research_split="external_test",
            )
        )
    return records


def _deduplicate(
    harp: list[ProblemRecord],
    math: list[ProblemRecord],
    *,
    threshold: float,
) -> tuple[list[ProblemRecord], list[dict]]:
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate-threshold must be in (0, 1]")
    math_exact = {_normalized(record.problem): record.problem_id for record in math}
    math_shingles = {record.problem_id: _shingles(record.problem) for record in math}
    inverted: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for problem_id, shingles in math_shingles.items():
        for shingle in shingles:
            inverted[shingle].add(problem_id)
    retained: list[ProblemRecord] = []
    excluded: list[dict] = []
    for record in harp:
        normalized = _normalized(record.problem)
        if normalized in math_exact:
            excluded.append(
                {"harp_problem_id": record.problem_id, "math_problem_id": math_exact[normalized], "reason": "exact", "similarity": 1.0}
            )
            continue
        shingles = _shingles(record.problem)
        candidates: set[str] = set()
        for shingle in shingles:
            candidates.update(inverted.get(shingle, set()))
        best_id = None
        best_similarity = 0.0
        for problem_id in candidates:
            similarity = _jaccard(shingles, math_shingles[problem_id])
            if similarity > best_similarity:
                best_id, best_similarity = problem_id, similarity
        if best_similarity >= threshold:
            excluded.append(
                {"harp_problem_id": record.problem_id, "math_problem_id": best_id, "reason": "near", "similarity": best_similarity}
            )
        else:
            retained.append(record)
    return retained, excluded


def main() -> None:
    args = parse_args()
    if args.sample_size < 12 or args.sample_size % 6:
        raise ValueError("HARP sample-size must be >=12 and divisible by six levels")
    source_rows = _rows(args.harp_jsonl_or_zip)
    harp = _canonicalize(source_rows)
    math = [record for path in args.math_bundle for record in read_problem_bundle(path)]
    if not math or any(record.dataset != "math" for record in math):
        raise ValueError("--math-bundle inputs must contain canonical MATH records")
    deduplicated, excluded = _deduplicate(
        harp,
        math,
        threshold=args.near_duplicate_threshold,
    )
    rng = random.Random(args.selection_seed)
    per_level = args.sample_size // 6
    selected: list[ProblemRecord] = []
    for level in range(1, 7):
        candidates = [record for record in deduplicated if record.level == level]
        rng.shuffle(candidates)
        if len(candidates) < per_level:
            raise RuntimeError(f"HARP level {level} has only {len(candidates)} eligible rows")
        selected.extend(candidates[:per_level])
    rng.shuffle(selected)
    selected = [replace(record, research_split="external_test") for record in selected]
    output = ensure_directory(args.output_dir)
    data_path, split_path = write_problem_bundle(selected, output, "harp_sample")
    audit_path = output / "math_overlap_audit.json"
    write_json_atomic(
        audit_path,
        {
            "exact_or_near_duplicates_removed": len(excluded),
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "excluded": excluded,
        },
    )
    manifest = {
        "schema_version": "phase05_harp_external_cohort_v1",
        "dataset": "harp",
        "source": HARP_REPOSITORY,
        "source_file_sha256": sha256_file(args.harp_jsonl_or_zip),
        "source_rows": len(source_rows),
        "selection_seed": args.selection_seed,
        "sample_size": len(selected),
        "levels": {str(level): sum(record.level == level for record in selected) for level in range(1, 7)},
        "subjects": {
            str(subject): sum(record.category == subject for record in selected)
            for subject in sorted({record.category for record in selected})
        },
        "problem_ids": [record.problem_id for record in selected],
        "data_sha256": sha256_file(data_path),
        "split_mapping_sha256": sha256_file(split_path),
        "math_bundle_sha256": {str(path): sha256_file(path) for path in args.math_bundle},
        "duplicate_audit_sha256": sha256_file(audit_path),
        "math_overlap_count": 0,
        "selection_outcome_blind": True,
        "forbidden_selection_fields": [
            "model_output",
            "correctness",
            "uncertainty",
            "reasoning_length",
            "breakthrough_probability",
        ],
    }
    write_json_atomic(output / "dataset_manifest.json", manifest)
    print(json.dumps({"selected": len(selected), "removed_duplicates": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
