"""Google Gemma 4 thinking controls and optional assigned-budget boundaries.

Gemma 4 E4B exposes a strict thinking on/off switch through
``enable_thinking`` in its official processor chat template. It does not expose
native low, medium, and high reasoning levels. The adapter retains an explicit
external boundary intervention for diagnostics and future experiments, but the
current Phase 1 and Phase 3 protocols use the model's native stopping behavior
under a common 8,192-token generation limit.

The shared recorder captures the last text position passed into the exposed
language-model head. Phase 0 validates the exact normalized capture point and
cached one-token alignment for the installed Transformers version.
"""

from __future__ import annotations

from reasonbench.exceptions import ConfigurationError
from reasonbench.instrumentation.adapters.base import (
    TASK_INSTRUCTION,
    ModelAdapter,
    PromptPlan,
)


class Gemma4Adapter(ModelAdapter):
    """Official Gemma 4 chat-template and thought-channel controls."""

    key = "gemma4"
    supports_reasoning_off = True
    supports_assigned_reasoning_budget = True

    def build_prompt(self, problem: str, mode: str) -> PromptPlan:
        if mode not in {"reasoning", "non_reasoning"}:
            raise ConfigurationError(f"Unsupported Gemma 4 mode: {mode}")
        return PromptPlan(
            messages=[
                {"role": "system", "content": "You are a careful mathematical assistant."},
                {"role": "user", "content": f"{problem}\n\n{TASK_INSTRUCTION}"},
            ],
            template_kwargs={"enable_thinking": mode == "reasoning"},
        )

    def reasoning_close_markers(self) -> tuple[str, ...]:
        return ("<channel|>",)

    def reasoning_open_markers(self) -> tuple[str, ...]:
        return ("<|channel>thought",)

    def forced_reasoning_close_text(self) -> str:
        return "<channel|>"
