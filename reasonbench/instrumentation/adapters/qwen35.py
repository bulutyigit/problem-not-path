"""Qwen3.5 thinking-mode controls.

Qwen3.5 exposes thinking control through ``enable_thinking`` in its official
chat template. The shared recorder captures the last position passed into the
language-model head. Phase 0 verifies the exact normalized capture point and
cached token/logit alignment for the installed Transformers version.
"""

from __future__ import annotations

from reasonbench.exceptions import ConfigurationError
from reasonbench.instrumentation.adapters.base import (
    TASK_INSTRUCTION,
    ModelAdapter,
    PromptPlan,
)


class Qwen35Adapter(ModelAdapter):
    """Official chat-template control for Qwen3.5."""

    key = "qwen35"
    supports_reasoning_off = True

    def build_prompt(self, problem: str, mode: str) -> PromptPlan:
        if mode not in {"reasoning", "non_reasoning"}:
            raise ConfigurationError(f"Unsupported Qwen3.5 mode: {mode}")
        return PromptPlan(
            messages=[
                {"role": "system", "content": "You are a careful mathematical assistant."},
                {"role": "user", "content": f"{problem}\n\n{TASK_INSTRUCTION}"},
            ],
            template_kwargs={"enable_thinking": mode == "reasoning"},
        )
