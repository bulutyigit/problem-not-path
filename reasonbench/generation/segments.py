"""Token-sequence segmentation from decoded token text spans.

Marker matching previously encoded each marker string and searched for that
token-id subsequence. That silently fails whenever a marker tokenizes
differently in context than in isolation, which was observed in practice for
Ministral's ``[/THINK]`` boundary and for ``\\boxed{`` inside generated text.
Matching on the concatenation of per-token decoded texts is robust to such
merges: a marker is found wherever its characters appear, and every token
whose character span overlaps the marker is labeled ``special``.
"""

from __future__ import annotations

from typing import Any

DEFAULT_REASONING_CLOSE_MARKERS = ("</think>", "[/THINK]", "<channel|>")
DEFAULT_REASONING_OPEN_MARKERS = ("<think>", "[THINK]", "<|channel>thought")
FINAL_ANSWER_MARKERS = ("\\boxed{", "\\fbox{")


def _first_occurrence(
    text: str,
    markers: tuple[str, ...],
    start: int = 0,
) -> tuple[int, int] | None:
    """Return the earliest ``(start, end)`` character span among the markers."""

    best: tuple[int, int] | None = None
    for marker in markers:
        if not marker:
            continue
        position = text.find(marker, start)
        if position >= 0 and (best is None or position < best[0]):
            best = (position, position + len(marker))
    return best


def segment_token_texts(
    token_texts: list[str],
    mode: str,
    reasoning_close_markers: tuple[str, ...] | None = None,
    reasoning_open_markers: tuple[str, ...] | None = None,
) -> list[str]:
    """Label thinking, solution, final-answer, and special tokens."""

    close_markers = reasoning_close_markers or DEFAULT_REASONING_CLOSE_MARKERS
    open_markers = reasoning_open_markers or DEFAULT_REASONING_OPEN_MARKERS
    token_count = len(token_texts)
    segments = ["solution"] * token_count
    starts: list[int] = []
    total = 0
    for text in token_texts:
        starts.append(total)
        total += len(text)
    ends = [start + len(text) for start, text in zip(starts, token_texts, strict=True)]
    joined = "".join(token_texts)

    def mark_special(span: tuple[int, int]) -> None:
        for index in range(token_count):
            if starts[index] < span[1] and ends[index] > span[0]:
                segments[index] = "special"

    close_span = _first_occurrence(joined, close_markers)
    answer_search_start = 0
    if mode == "reasoning":
        open_span = _first_occurrence(joined, open_markers)
        thinking_start = open_span[1] if open_span is not None else 0
        thinking_end = total
        if close_span is not None:
            thinking_end = close_span[0]
            answer_search_start = close_span[1]
        for index in range(token_count):
            if thinking_start <= starts[index] < thinking_end:
                segments[index] = "thinking"
        if open_span is not None:
            mark_special(open_span)
        if close_span is not None:
            mark_special(close_span)
    box_span = _first_occurrence(joined, FINAL_ANSWER_MARKERS, start=answer_search_start)
    if box_span is not None:
        first_box_token = next(
            (index for index in range(token_count) if ends[index] > box_span[0]),
            None,
        )
        if first_box_token is not None:
            for index in range(first_box_token, token_count):
                segments[index] = "final_answer"
    return segments


def segment_generated_tokens(
    token_ids: list[int],
    tokenizer: Any,
    mode: str,
    reasoning_close_markers: tuple[str, ...] | None = None,
    reasoning_open_markers: tuple[str, ...] | None = None,
) -> list[str]:
    """Label generated tokens by decoding each one exactly as the recorder does."""

    token_texts = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    return segment_token_texts(
        token_texts,
        mode,
        reasoning_close_markers=reasoning_close_markers,
        reasoning_open_markers=reasoning_open_markers,
    )
