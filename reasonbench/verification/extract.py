"""Final-answer and reasoning-segment extraction."""

from __future__ import annotations

import re


def extract_boxed_answers(text: str) -> list[str]:
    """Extract balanced contents from every LaTeX boxed expression."""

    answers: list[str] = []
    cursor = 0
    markers = ("\\boxed{", "\\fbox{")
    while cursor < len(text):
        positions = [(text.find(marker, cursor), marker) for marker in markers]
        positions = [(position, marker) for position, marker in positions if position >= 0]
        if not positions:
            break
        start, marker = min(positions, key=lambda item: item[0])
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            answers.append(text[content_start : index - 1].strip())
            cursor = index
        else:
            cursor = content_start
    return answers


def extract_final_answer(text: str) -> tuple[str | None, str]:
    """Extract a final answer and return an extraction status."""

    boxed = extract_boxed_answers(text)
    if boxed:
        return boxed[-1], "boxed"
    gsm_match = re.findall(r"####\s*([^\n]+)", text)
    if gsm_match:
        return gsm_match[-1].strip(), "gsm_marker"
    final_patterns = [
        r"(?i)final\s+answer\s*(?:is|:)\s*([^\n]+)",
        r"(?i)answer\s*(?:is|:)\s*([^\n]+)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip(), "text_marker"
    return None, "missing"


def split_reasoning_and_answer(text: str) -> tuple[str, str, str]:
    """Split visible reasoning from final response without claiming faithfulness."""

    markers = [
        (text.rfind("</think>"), "<think>", "</think>", "think_tag"),
        (text.rfind("[/THINK]"), "[THINK]", "[/THINK]", "mistral_think_tag"),
        (
            text.rfind("<channel|>"),
            "<|channel>thought",
            "<channel|>",
            "gemma_thought_channel",
        ),
    ]
    close_index, open_marker, close_marker, status = max(
        markers,
        key=lambda item: item[0],
    )
    if close_index >= 0:
        return (
            text[:close_index].replace(open_marker, "", 1).strip(),
            text[close_index + len(close_marker) :].strip(),
            status,
        )
    boxed_positions = [text.rfind("\\boxed{"), text.rfind("\\fbox{")]
    box_index = max(boxed_positions)
    if box_index >= 0:
        return text[:box_index].strip(), text[box_index:].strip(), "boxed_boundary"
    return text.strip(), "", "unknown"
