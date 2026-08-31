"""Dataset-aware correctness grading."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from reasonbench.verification.extract import extract_final_answer


@dataclass(frozen=True)
class VerificationResult:
    """Structured correctness result."""

    correct: bool
    predicted_answer: str | None
    reference_answer: str
    extraction_status: str
    verification_method: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_wrappers(value: str) -> str:
    normalized = value.strip().strip("$")
    normalized = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", normalized)
    normalized = normalized.replace("\\,", "").replace(" ", "")
    normalized = normalized.rstrip(".,;")
    return normalized


def _decimal_value(value: str) -> Decimal | None:
    normalized = _strip_wrappers(value)
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("\\%", "%")
    is_percent = normalized.endswith("%")
    if is_percent:
        normalized = normalized[:-1]
    fraction_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", normalized)
    try:
        if fraction_match:
            numerator = Decimal(fraction_match.group(1))
            denominator = Decimal(fraction_match.group(2))
            if denominator == 0:
                return None
            result = numerator / denominator
        else:
            result = Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None
    return result / Decimal(100) if is_percent else result


def _numeric_equal(prediction: str, reference: str) -> bool | None:
    predicted_value = _decimal_value(prediction)
    reference_value = _decimal_value(reference)
    if predicted_value is None or reference_value is None:
        return None
    return math.isclose(
        float(predicted_value),
        float(reference_value),
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _math_verify_equal(prediction: str, reference: str) -> tuple[bool | None, str | None]:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None, "math_verify_not_installed"
    try:
        gold = parse(f"${reference}$")
        answer = parse(f"${prediction}$")
        return bool(verify(gold, answer)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def verify_answer(
    generated_text: str,
    reference_answer: str,
    dataset: str,
) -> VerificationResult:
    """Extract and verify a generated final answer."""

    prediction, extraction_status = extract_final_answer(generated_text)
    if prediction is None:
        return VerificationResult(
            correct=False,
            predicted_answer=None,
            reference_answer=reference_answer,
            extraction_status=extraction_status,
            verification_method="missing_answer",
        )
    numeric_result = _numeric_equal(prediction, reference_answer)
    if numeric_result is not None:
        return VerificationResult(
            correct=numeric_result,
            predicted_answer=prediction,
            reference_answer=reference_answer,
            extraction_status=extraction_status,
            verification_method="numeric",
        )
    normalized_prediction = _strip_wrappers(prediction)
    normalized_reference = _strip_wrappers(reference_answer)
    if normalized_prediction == normalized_reference:
        return VerificationResult(
            correct=True,
            predicted_answer=prediction,
            reference_answer=reference_answer,
            extraction_status=extraction_status,
            verification_method="normalized_string",
        )
    if dataset in {"math", "harp"}:
        verified, error = _math_verify_equal(prediction, reference_answer)
        if verified is not None:
            return VerificationResult(
                correct=verified,
                predicted_answer=prediction,
                reference_answer=reference_answer,
                extraction_status=extraction_status,
                verification_method="math_verify",
                error=error,
            )
    return VerificationResult(
        correct=False,
        predicted_answer=prediction,
        reference_answer=reference_answer,
        extraction_status=extraction_status,
        verification_method="normalized_string",
    )
