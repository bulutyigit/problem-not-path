"""Hugging Face dataset adapters for GSM8K and MATH."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from reasonbench.exceptions import ConfigurationError
from reasonbench.verification.extract import extract_boxed_answers

DATASET_SOURCES = {
    "gsm8k": {
        "repository": "openai/gsm8k",
        "configuration": "main",
        "split": "test",
    },
    "math": {
        "repository": "DigitalLearningGmbH/MATH-lighteval",
        "configuration": "default",
        "split": "test",
    },
}


@dataclass(frozen=True)
class ProblemRecord:
    """Canonical benchmark problem."""

    problem_id: str
    dataset: str
    source_repository: str
    source_split: str
    source_index: int
    problem: str
    reference_answer: str
    reference_solution: str
    level: int | None = None
    category: str | None = None
    research_split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_problem_id(dataset: str, source_index: int, problem: str) -> str:
    digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}_{source_index:05d}_{digest}"


def _parse_math_level(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"([1-5])", str(value))
    return int(match.group(1)) if match else None


def _gsm8k_reference(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ConfigurationError("GSM8K reference answer is missing the #### marker")
    return answer.rsplit(marker, maxsplit=1)[1].strip()


def _math_reference(row: dict[str, Any]) -> str:
    if row.get("answer") not in {None, ""}:
        return str(row["answer"]).strip()
    solution = str(row.get("solution", ""))
    boxed = extract_boxed_answers(solution)
    if not boxed:
        raise ConfigurationError("MATH reference solution does not contain a boxed answer")
    return boxed[-1]


def _to_problem_record(dataset: str, row: dict[str, Any], index: int) -> ProblemRecord:
    source = DATASET_SOURCES[dataset]
    if dataset == "gsm8k":
        problem = str(row["question"]).strip()
        solution = str(row["answer"]).strip()
        reference = _gsm8k_reference(solution)
        level = None
        category = None
    elif dataset == "math":
        problem = str(row["problem"]).strip()
        solution = str(row["solution"]).strip()
        reference = _math_reference(row)
        level = _parse_math_level(row.get("level"))
        category = str(row.get("type") or row.get("category") or "").strip() or None
    else:
        raise ConfigurationError(f"Unsupported dataset: {dataset}")
    return ProblemRecord(
        problem_id=_stable_problem_id(dataset, index, problem),
        dataset=dataset,
        source_repository=source["repository"],
        source_split=source["split"],
        source_index=index,
        problem=problem,
        reference_answer=reference,
        reference_solution=solution,
        level=level,
        category=category,
    )


def load_problem_records(
    dataset: str,
    revision: str | None = None,
) -> list[ProblemRecord]:
    """Download and canonicalize one benchmark dataset."""

    if dataset not in DATASET_SOURCES:
        raise ConfigurationError(f"Unsupported dataset: {dataset}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ConfigurationError(
            "The datasets package is required. Install the colab extra."
        ) from exc
    source = DATASET_SOURCES[dataset]
    kwargs: dict[str, Any] = {"split": source["split"]}
    if revision:
        kwargs["revision"] = revision
    configuration = source["configuration"]
    try:
        loaded = load_dataset(source["repository"], configuration, **kwargs)
    except ValueError:
        if configuration != "default":
            raise
        loaded = load_dataset(source["repository"], **kwargs)
    return [_to_problem_record(dataset, dict(row), index) for index, row in enumerate(loaded)]


def _balanced_math_sample(
    records: list[ProblemRecord],
    sample_size: int,
    seed: int,
) -> list[ProblemRecord]:
    rng = random.Random(seed)
    by_level: dict[int | None, list[ProblemRecord]] = {}
    for record in records:
        by_level.setdefault(record.level, []).append(record)
    for level_records in by_level.values():
        rng.shuffle(level_records)
    target_levels = [level for level in range(1, 6) if level in by_level]
    if not target_levels:
        rng.shuffle(records)
        return records[:sample_size]
    selected: list[ProblemRecord] = []
    per_level = sample_size // len(target_levels)
    for level in target_levels:
        selected.extend(by_level[level][:per_level])
    selected_ids = {record.problem_id for record in selected}
    remaining = [record for record in records if record.problem_id not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: sample_size - len(selected)])
    rng.shuffle(selected)
    return selected


def build_problem_sample(
    records: Iterable[ProblemRecord],
    sample_size: int,
    seed: int,
    levels: Iterable[int] = (),
    nested_base_sample_size: int | None = None,
) -> list[ProblemRecord]:
    """Create a deterministic benchmark sample."""

    materialized = list(records)
    if sample_size > len(materialized):
        raise ConfigurationError(
            f"Requested {sample_size} problems from a dataset with {len(materialized)} rows"
        )
    datasets = {record.dataset for record in materialized}
    if len(datasets) != 1:
        raise ConfigurationError("build_problem_sample expects records from one dataset")
    dataset = next(iter(datasets))
    requested_levels = tuple(sorted(set(int(level) for level in levels)))
    if nested_base_sample_size is not None:
        if dataset != "math" or not requested_levels:
            raise ConfigurationError("Nested sampling requires level-balanced MATH sampling")
        if not 0 < nested_base_sample_size < sample_size:
            raise ConfigurationError(
                "nested_base_sample_size must be positive and smaller than sample_size"
            )
        if nested_base_sample_size % len(requested_levels):
            raise ConfigurationError(
                "nested_base_sample_size must be divisible by the requested levels"
            )
        base = _balanced_math_sample(materialized.copy(), nested_base_sample_size, seed)
        target_per_level = sample_size // len(requested_levels)
        base_ids = {record.problem_id for record in base}
        base_counts = {
            level: sum(record.level == level for record in base) for level in requested_levels
        }
        expected_base_per_level = nested_base_sample_size // len(requested_levels)
        if any(count != expected_base_per_level for count in base_counts.values()):
            raise ConfigurationError(
                "The nested base sample is not exactly balanced across requested levels"
            )
        rng = random.Random(seed)
        selected = list(base)
        for level in requested_levels:
            candidates = [
                record
                for record in materialized
                if record.level == level and record.problem_id not in base_ids
            ]
            rng.shuffle(candidates)
            needed = target_per_level - base_counts[level]
            if len(candidates) < needed:
                raise ConfigurationError(
                    f"Nested sample needs {needed} additional level-{level} problems, "
                    f"but only {len(candidates)} are available"
                )
            selected.extend(candidates[:needed])
        rng.shuffle(selected)
        return selected
    if requested_levels:
        if dataset not in {"math", "harp"}:
            raise ConfigurationError("Difficulty-level sampling is supported only for MATH/HARP")
        if sample_size % len(requested_levels):
            raise ConfigurationError(
                "A level-balanced sample size must be divisible by the number of levels"
            )
        per_level = sample_size // len(requested_levels)
        rng = random.Random(seed)
        selected: list[ProblemRecord] = []
        for level in requested_levels:
            candidates = [record for record in materialized if record.level == level]
            if len(candidates) < per_level:
                raise ConfigurationError(
                    f"Requested {per_level} MATH level-{level} problems, "
                    f"but only {len(candidates)} are available"
                )
            rng.shuffle(candidates)
            selected.extend(candidates[:per_level])
        rng.shuffle(selected)
        return selected
    if dataset == "math":
        return _balanced_math_sample(materialized, sample_size, seed)
    rng = random.Random(seed)
    rng.shuffle(materialized)
    return materialized[:sample_size]
