from __future__ import annotations

import pandas as pd

from reasonbench.datasets.loader import ProblemRecord, build_problem_sample
from reasonbench.evaluation.difficulty import (
    difficulty_metric_summary,
    level_trends,
    validate_difficulty_design,
)


def _math_records(per_level: int = 25) -> list[ProblemRecord]:
    records = []
    for level in range(1, 6):
        for index in range(per_level):
            records.append(
                ProblemRecord(
                    problem_id=f"math_{level}_{index}",
                    dataset="math",
                    source_repository="fixture",
                    source_split="test",
                    source_index=len(records),
                    problem=f"Level {level} problem {index}",
                    reference_answer=str(index),
                    reference_solution=str(index),
                    level=level,
                    category="algebra",
                )
            )
    return records


def test_level_sampling_is_exact_balanced_and_deterministic() -> None:
    first = build_problem_sample(
        _math_records(),
        sample_size=100,
        seed=17,
        levels=(1, 2, 3, 4, 5),
    )
    second = build_problem_sample(
        _math_records(),
        sample_size=100,
        seed=17,
        levels=(1, 2, 3, 4, 5),
    )
    assert [record.problem_id for record in first] == [record.problem_id for record in second]
    assert pd.Series([record.level for record in first]).value_counts().to_dict() == {
        level: 20 for level in range(1, 6)
    }


def test_nested_level_sampling_contains_the_smaller_balanced_sample() -> None:
    records = _math_records(per_level=40)
    base = build_problem_sample(records, sample_size=50, seed=17)
    expanded = build_problem_sample(
        records,
        sample_size=100,
        seed=17,
        levels=(1, 2, 3, 4, 5),
        nested_base_sample_size=50,
    )

    assert {record.problem_id for record in base} <= {record.problem_id for record in expanded}
    assert pd.Series([record.level for record in expanded]).value_counts().to_dict() == {
        level: 20 for level in range(1, 6)
    }


def _feature_frame(models: tuple[str, ...] = ("qwen35_4b",)) -> pd.DataFrame:
    rows = []
    for model in models:
        for level in range(1, 6):
            for problem_index in range(2):
                problem_id = f"math_{level}_{problem_index}"
                for seed in (11, 23):
                    rows.append(
                        {
                            "dataset": "math",
                            "model_key": model,
                            "problem_id": problem_id,
                            "seed": seed,
                            "level": level,
                            "category": "algebra",
                            "correct": float(level <= 3),
                            "trajectory_token_count": float(level * 100 + seed),
                            "normalized_entropy_mean": float(level) / 10,
                        }
                    )
    return pd.DataFrame(rows)


def test_difficulty_design_and_trends_use_problem_clusters() -> None:
    frame = _feature_frame()
    design = validate_difficulty_design(
        frame,
        expected_models=["qwen35_4b"],
        problems_per_level=2,
        seeds_per_problem=2,
    )
    assert design["problems"] == 10
    summary = difficulty_metric_summary(
        frame,
        ["normalized_entropy_mean"],
        repetitions=20,
        seed=3,
    )
    assert len(summary) == 5
    trends = level_trends(
        frame,
        ["normalized_entropy_mean"],
        repetitions=20,
        seed=3,
    )
    assert trends["models"]["qwen35_4b"]["normalized_entropy_mean"]["slope_per_level"] > 0


def test_cross_model_design_requires_same_problem_seed_pairs() -> None:
    frame = _feature_frame(("gemma4_e4b", "qwen35_4b"))
    validate_difficulty_design(
        frame,
        expected_models=["gemma4_e4b", "qwen35_4b"],
        problems_per_level=2,
        seeds_per_problem=2,
    )
    broken = frame.drop(frame.index[-1])
    try:
        validate_difficulty_design(
            broken,
            expected_models=["gemma4_e4b", "qwen35_4b"],
            problems_per_level=2,
            seeds_per_problem=2,
        )
    except ValueError as exc:
        assert "same problem/seed pairs" in str(exc)
    else:
        raise AssertionError("Mismatched cross-model pairs were accepted")
