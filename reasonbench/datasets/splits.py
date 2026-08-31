"""Problem-level research split creation."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from reasonbench.datasets.loader import ProblemRecord
from reasonbench.exceptions import ConfigurationError
from reasonbench.storage import ensure_directory, write_json_atomic, write_text_atomic


def assign_research_splits(
    records: Iterable[ProblemRecord],
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> list[ProblemRecord]:
    """Assign immutable problem-level train, validation, and test labels."""

    if not 0 < train_fraction < 1:
        raise ConfigurationError("train_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ConfigurationError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ConfigurationError("train and validation fractions must sum to less than 1")
    materialized = list(records)
    if len({record.problem_id for record in materialized}) != len(materialized):
        raise ConfigurationError("Problem IDs must be unique before split assignment")
    rng = random.Random(seed)
    rng.shuffle(materialized)
    train_end = round(len(materialized) * train_fraction)
    validation_end = train_end + round(len(materialized) * validation_fraction)
    assigned: list[ProblemRecord] = []
    for index, record in enumerate(materialized):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        assigned.append(replace(record, research_split=split))
    return sorted(assigned, key=lambda record: record.problem_id)


def assign_stratified_research_splits(
    records: Iterable[ProblemRecord],
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> list[ProblemRecord]:
    """Assign immutable research splits independently within MATH level.

    The assignment uses only the pre-run level label and deterministic record
    identity; it never observes model outcomes.  Each level receives the same
    rounded train/validation/test allocation when its size permits it.
    """

    materialized = list(records)
    if any(record.level is None for record in materialized):
        raise ConfigurationError("Stratified MATH splits require a level for every record")
    if len({record.problem_id for record in materialized}) != len(materialized):
        raise ConfigurationError("Problem IDs must be unique before split assignment")
    grouped: dict[int, list[ProblemRecord]] = {}
    for record in materialized:
        assert record.level is not None
        grouped.setdefault(record.level, []).append(record)
    assigned: list[ProblemRecord] = []
    for level, group in sorted(grouped.items()):
        # Level-specific RNG makes the split stable if another level is later
        # extended in an explicitly versioned bundle.
        rng = random.Random(f"{seed}:level:{level}")
        shuffled = sorted(group, key=lambda record: record.problem_id)
        rng.shuffle(shuffled)
        train_end = round(len(shuffled) * train_fraction)
        validation_end = train_end + round(len(shuffled) * validation_fraction)
        for index, record in enumerate(shuffled):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            assigned.append(replace(record, research_split=split))
    return sorted(assigned, key=lambda record: record.problem_id)


def write_problem_bundle(
    records: Iterable[ProblemRecord],
    output_directory: str | Path,
    name: str,
) -> tuple[Path, Path]:
    """Write canonical JSONL records and a compact split mapping."""

    root = ensure_directory(output_directory)
    materialized = list(records)
    data_path = root / f"{name}.jsonl"
    lines = [
        json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) for record in materialized
    ]
    write_text_atomic(data_path, "\n".join(lines) + "\n")
    split_path = root / f"{name}_splits.json"
    split_mapping = {record.problem_id: record.research_split for record in materialized}
    write_json_atomic(split_path, split_mapping)
    return data_path, split_path


def read_problem_bundle(path: str | Path) -> list[ProblemRecord]:
    """Read canonical JSONL problem records."""

    records: list[ProblemRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ProblemRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"Invalid problem record at {path}:{line_number}") from exc
    return records
