from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from reasonbench.datasets.loader import ProblemRecord
from reasonbench.datasets.splits import assign_research_splits, assign_stratified_research_splits
from reasonbench.exceptions import PhaseGateError
from reasonbench.phases import PhaseStatus, require_phase_gate


def _records(count: int = 20) -> list[ProblemRecord]:
    return [
        ProblemRecord(
            problem_id=f"gsm8k_{index}",
            dataset="gsm8k",
            source_repository="fixture",
            source_split="test",
            source_index=index,
            problem=f"Problem {index}",
            reference_answer=str(index),
            reference_solution=str(index),
        )
        for index in range(count)
    ]


def test_problem_splits_are_deterministic_and_disjoint() -> None:
    first = assign_research_splits(_records(), seed=7)
    second = assign_research_splits(_records(), seed=7)
    assert [(row.problem_id, row.research_split) for row in first] == [
        (row.problem_id, row.research_split) for row in second
    ]
    split_sets = {
        split: {row.problem_id for row in first if row.research_split == split}
        for split in ("train", "validation", "test")
    }
    assert not (split_sets["train"] & split_sets["validation"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["validation"] & split_sets["test"])


def test_duplicate_problem_ids_are_rejected() -> None:
    records = _records(2)
    records[1] = replace(records[1], problem_id=records[0].problem_id)
    with pytest.raises(Exception, match="unique"):
        assign_research_splits(records, seed=7)


def test_phase04b_level_stratified_split_is_deterministic_and_balanced() -> None:
    records = [
        replace(record, dataset="math", level=index // 20 + 1)
        for index, record in enumerate(_records(100))
    ]
    first = assign_stratified_research_splits(records, seed=11)
    second = assign_stratified_research_splits(records, seed=11)
    assert [(item.problem_id, item.research_split) for item in first] == [
        (item.problem_id, item.research_split) for item in second
    ]
    for level in range(1, 6):
        assigned = [item for item in first if item.level == level]
        assert {item.research_split for item in assigned} == {"train", "validation", "test"}
        assert sum(item.research_split == "train" for item in assigned) == 12
        assert sum(item.research_split == "validation" for item in assigned) == 4
        assert sum(item.research_split == "test" for item in assigned) == 4


def test_phase_gate_accepts_only_passed_allowed_decision(tmp_path: Path) -> None:
    phase_root = tmp_path / "phase_00"
    PhaseStatus(
        phase_id="phase_00",
        technical_status="passed",
        scientific_outcome="not_applicable",
        next_decision="continue",
    ).write(phase_root)
    loaded = require_phase_gate(tmp_path, "phase_00", {"continue"})
    assert loaded.next_decision == "continue"
    with pytest.raises(PhaseGateError, match="not one of"):
        require_phase_gate(tmp_path, "phase_00", {"stop"})


def test_passed_phase_requires_next_decision() -> None:
    with pytest.raises(PhaseGateError, match="next_decision"):
        PhaseStatus(
            phase_id="phase_01",
            technical_status="passed",
            scientific_outcome="limited",
        ).validate()
