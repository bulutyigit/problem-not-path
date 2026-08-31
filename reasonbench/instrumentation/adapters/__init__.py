"""Model-specific prompt and reasoning controls."""

from reasonbench.instrumentation.adapters.base import ModelAdapter, get_model_adapter
from reasonbench.instrumentation.adapters.gemma4 import Gemma4Adapter
from reasonbench.instrumentation.adapters.ministral3 import Ministral3Adapter
from reasonbench.instrumentation.adapters.qwen35 import Qwen35Adapter

__all__ = [
    "Gemma4Adapter",
    "Ministral3Adapter",
    "ModelAdapter",
    "Qwen35Adapter",
    "get_model_adapter",
]
