from reasonbench.generation.engine import _infer_finish_reason


def test_finish_reason_uses_realized_length_when_model_eos_id_differs() -> None:
    assert (
        _infer_finish_reason(
            [7, 8, 99],
            token_limit=8,
            eos_token_id=1,
            limit_reason="max_new_tokens",
        )
        == "eos"
    )


def test_finish_reason_reports_limit_only_when_the_limit_is_exhausted() -> None:
    assert (
        _infer_finish_reason(
            [7, 8, 9],
            token_limit=3,
            eos_token_id=1,
            limit_reason="max_new_tokens",
        )
        == "max_new_tokens"
    )
    assert (
        _infer_finish_reason(
            [7, 8, 9],
            token_limit=3,
            eos_token_id=1,
            limit_reason="answer_reserve",
        )
        == "answer_reserve"
    )
