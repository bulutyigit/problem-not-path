"""Shared model-adapter interfaces and prompt structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reasonbench.exceptions import ConfigurationError

TASK_INSTRUCTION = "Solve the problem carefully. Put only the final answer inside \\boxed{}."


@dataclass(frozen=True)
class PromptPlan:
    """Messages and template options for one generation."""

    messages: list[dict[str, Any]]
    template_kwargs: dict[str, Any]


class ModelAdapter:
    """Base prompt and segmentation behavior."""

    key = "base"
    supports_reasoning_off = False
    supports_assigned_reasoning_budget = False

    def build_prompt(self, problem: str, mode: str) -> PromptPlan:
        if mode == "non_reasoning" and not self.supports_reasoning_off:
            raise ConfigurationError(f"{self.key} does not support non_reasoning mode")
        return PromptPlan(
            messages=[
                {"role": "system", "content": "You are a careful mathematical assistant."},
                {"role": "user", "content": f"{problem}\n\n{TASK_INSTRUCTION}"},
            ],
            template_kwargs={},
        )

    def reasoning_close_markers(self) -> tuple[str, ...]:
        """Return model-native strings that end a visible reasoning channel."""

        return ("</think>",)

    def reasoning_open_markers(self) -> tuple[str, ...]:
        """Return model-native strings that begin a visible reasoning channel."""

        return ("<think>",)

    def forced_reasoning_close_text(self) -> str:
        """Return the boundary injected after an assigned reasoning-token cap."""

        return ".\n</think>\n\n"


def get_model_adapter(key: str) -> ModelAdapter:
    """Return a registered model adapter."""

    from reasonbench.instrumentation.adapters.gemma4 import Gemma4Adapter
    from reasonbench.instrumentation.adapters.ministral3 import Ministral3Adapter
    from reasonbench.instrumentation.adapters.qwen35 import Qwen35Adapter

    adapters = {
        "gemma4": Gemma4Adapter,
        "qwen35": Qwen35Adapter,
        "ministral3": Ministral3Adapter,
    }
    try:
        return adapters[key]()
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown model adapter {key!r}; expected one of {sorted(adapters)}"
        ) from exc
