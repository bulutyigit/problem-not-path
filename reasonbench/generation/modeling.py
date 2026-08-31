"""Official-checkpoint model and tokenizer loading."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasonbench.config import ModelConfig
from reasonbench.exceptions import InstrumentationError
from reasonbench.instrumentation.adapters import ModelAdapter, get_model_adapter


@dataclass
class ModelBundle:
    """Loaded model components."""

    model: Any
    tokenizer: Any
    processor: Any | None
    adapter: ModelAdapter
    model_config: ModelConfig
    resolved_revision: str | None
    architecture: str
    bf16_parameter_fraction: float


def _load_tokenizer_and_processor(config: ModelConfig) -> tuple[Any, Any | None]:
    import transformers
    from transformers import AutoProcessor, AutoTokenizer

    common = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
    }
    common = {key: value for key, value in common.items() if value is not None}
    if config.adapter == "ministral3" and hasattr(transformers, "MistralCommonBackend"):
        backend = transformers.MistralCommonBackend.from_pretrained(
            config.model_id,
            **common,
        )
        return backend, backend
    if config.adapter in {"gemma4", "qwen35", "ministral3"}:
        try:
            processor = AutoProcessor.from_pretrained(config.model_id, **common)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None and hasattr(processor, "decode"):
                tokenizer = processor
            if tokenizer is None:
                raise InstrumentationError(
                    f"Processor for {config.model_id} does not expose token decoding"
                )
            return tokenizer, processor
        except (ValueError, TypeError, OSError):
            if config.adapter == "ministral3":
                raise
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
        return tokenizer, None
    except (ValueError, TypeError, OSError):
        processor = AutoProcessor.from_pretrained(config.model_id, **common)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise InstrumentationError(
                f"Processor for {config.model_id} does not expose a tokenizer"
            ) from None
        return tokenizer, processor


def _load_official_model(
    config: ModelConfig,
) -> tuple[Any, str, str | None, float]:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    common: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "dtype": torch.bfloat16,
        "device_map": {"": "cuda:0"},
        "low_cpu_mem_usage": True,
    }
    common = {key: value for key, value in common.items() if value is not None}
    architecture = ""
    resolved_revision: str | None = config.revision
    auto_config = AutoConfig.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    architectures = list(getattr(auto_config, "architectures", None) or [])
    if architectures and hasattr(transformers, architectures[0]):
        architecture = architectures[0]
        model_class = getattr(transformers, architecture)
        model = model_class.from_pretrained(config.model_id, **common)
    else:
        architecture = "AutoModelForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **common)
    model.eval()
    floating_parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    floating_elements = sum(parameter.numel() for parameter in floating_parameters)
    bf16_elements = sum(
        parameter.numel() for parameter in floating_parameters if parameter.dtype == torch.bfloat16
    )
    bf16_fraction = bf16_elements / floating_elements if floating_elements else 0.0
    if bf16_fraction < 0.95:
        raise InstrumentationError(
            f"{config.model_id} did not load predominantly in BF16; "
            f"BF16 parameter fraction={bf16_fraction:.6f}"
        )
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash:
        resolved_revision = str(commit_hash)
    return model, architecture, resolved_revision, bf16_fraction


def load_model_bundle(config: ModelConfig) -> ModelBundle:
    """Load one official model checkpoint on CUDA."""

    config.validate()
    try:
        import torch
    except ImportError as exc:
        raise InstrumentationError("PyTorch is required for model loading") from exc
    if not torch.cuda.is_available():
        raise InstrumentationError("CUDA is required for primary model generation")
    tokenizer, processor = _load_tokenizer_and_processor(config)
    model, architecture, resolved_revision, bf16_fraction = _load_official_model(config)
    if model.get_output_embeddings() is None:
        raise InstrumentationError(
            f"{config.model_id} does not expose output embeddings for instrumentation"
        )
    adapter = get_model_adapter(config.adapter)
    if config.adapter == "ministral3":
        from huggingface_hub import hf_hub_download

        prompt_path = hf_hub_download(
            repo_id=config.model_id,
            filename="SYSTEM_PROMPT.txt",
            revision=resolved_revision,
        )
        adapter.configure_system_prompt(
            Path(prompt_path).read_text(encoding="utf-8"),
            source=f"{config.model_id}@{resolved_revision}:SYSTEM_PROMPT.txt",
        )
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        adapter=adapter,
        model_config=config,
        resolved_revision=resolved_revision,
        architecture=architecture,
        bf16_parameter_fraction=bf16_fraction,
    )


def unload_model_bundle(bundle: ModelBundle) -> None:
    """Release a loaded checkpoint between model conditions."""

    del bundle.model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
