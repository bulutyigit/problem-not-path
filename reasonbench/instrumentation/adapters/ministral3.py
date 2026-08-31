"""Ministral 3 8B Reasoning text-generation adapter.

The reasoning-specific checkpoint is used through its official chat template;
no undocumented switch is injected. The shared recorder hooks the exposed
language-model head. Phase 0 validates that text-only generation, the final
normalized head input, and cached one-token alignment are all available through
the checkpoint's official architecture class.
"""

from __future__ import annotations

from reasonbench.instrumentation.adapters.base import (
    TASK_INSTRUCTION,
    ModelAdapter,
    PromptPlan,
)


class Ministral3Adapter(ModelAdapter):
    """Text-only prompt behavior for the reasoning-specific Ministral checkpoint."""

    key = "ministral3"

    def __init__(self) -> None:
        self._system_content: str | list[dict[str, object]] | None = None
        self.prompt_source: str | None = None

    def configure_system_prompt(self, text: str, source: str) -> None:
        """Parse the checkpoint's official structured reasoning system prompt."""

        begin_marker = "[THINK]"
        end_marker = "[/THINK]"
        begin = text.find(begin_marker)
        end = text.find(end_marker)
        if begin < 0 or end < begin:
            self._system_content = text
        else:
            self._system_content = [
                {"type": "text", "text": text[:begin]},
                {
                    "type": "thinking",
                    "thinking": text[begin + len(begin_marker) : end],
                    "closed": True,
                },
                {"type": "text", "text": text[end + len(end_marker) :]},
            ]
        self.prompt_source = source

    def build_prompt(self, problem: str, mode: str) -> PromptPlan:
        if mode != "reasoning":
            return super().build_prompt(problem, mode)
        if self._system_content is None:
            raise RuntimeError("The official Ministral SYSTEM_PROMPT.txt was not configured")
        return PromptPlan(
            messages=[
                {"role": "system", "content": self._system_content},
                {"role": "user", "content": f"{problem}\n\n{TASK_INSTRUCTION}"},
            ],
            template_kwargs={},
        )

    def reasoning_close_markers(self) -> tuple[str, ...]:
        return ("[/THINK]",)

    def reasoning_open_markers(self) -> tuple[str, ...]:
        return ("[THINK]",)

    def forced_reasoning_close_text(self) -> str:
        return "[/THINK]\n\n"
