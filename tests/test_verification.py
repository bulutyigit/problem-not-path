from __future__ import annotations

from reasonbench.verification.extract import (
    extract_boxed_answers,
    extract_final_answer,
    split_reasoning_and_answer,
)
from reasonbench.verification.grader import verify_answer


def test_balanced_boxed_answer_extraction() -> None:
    text = r"First \boxed{\frac{1}{2}}, then \boxed{x^{2}+1}."
    assert extract_boxed_answers(text) == [r"\frac{1}{2}", r"x^{2}+1"]


def test_last_boxed_answer_is_final() -> None:
    answer, status = extract_final_answer(r"Draft \boxed{3}. Final \boxed{4}.")
    assert answer == "4"
    assert status == "boxed"


def test_numeric_normalization() -> None:
    result = verify_answer(r"The result is \boxed{1,250}.", "1250", "gsm8k")
    assert result.correct
    assert result.verification_method == "numeric"


def test_percent_and_fraction_normalization() -> None:
    result = verify_answer(r"\boxed{50\%}", "1/2", "gsm8k")
    assert result.correct


def test_missing_answer_is_separate_failure() -> None:
    result = verify_answer("No final marker is present.", "7", "gsm8k")
    assert not result.correct
    assert result.extraction_status == "missing"
    assert result.verification_method == "missing_answer"


def test_reasoning_boundary_split() -> None:
    reasoning, final, status = split_reasoning_and_answer(
        "<think>Compute carefully.</think> Final: \\boxed{9}"
    )
    assert reasoning == "Compute carefully."
    assert final == "Final: \\boxed{9}"
    assert status == "think_tag"
