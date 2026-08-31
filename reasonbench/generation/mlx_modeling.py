"""Apple-Silicon MLX model loading for quantized Phase 4 experiments."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasonbench.config import ModelConfig
from reasonbench.exceptions import InstrumentationError
from reasonbench.instrumentation.adapters import ModelAdapter, get_model_adapter


@dataclass
class MLXModelBundle:
    model: Any
    tokenizer: Any
    processor: Any
    adapter: ModelAdapter
    model_config: ModelConfig
    resolved_revision: str
    architecture: str
    quantization_bits: int
    quantization_group_size: int | None
    prompt_source: str | None = None


def _quantization_metadata(config_path: str | Path) -> tuple[int, int | None]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    quantization = payload.get("quantization") or payload.get("quantization_config") or {}
    bits = quantization.get("bits")
    group_size = quantization.get("group_size")
    if int(bits or 0) != 4:
        raise InstrumentationError(
            f"The MLX checkpoint is not declared as 4-bit in {config_path}: {quantization}"
        )
    return int(bits), None if group_size is None else int(group_size)


def load_mlx_model_bundle(config: ModelConfig) -> MLXModelBundle:
    """Resolve an immutable MLX checkpoint, validate int4 metadata, then load it."""

    config.validate()
    if config.backend != "mlx_vlm":
        raise InstrumentationError("load_mlx_model_bundle requires backend='mlx_vlm'")
    try:
        from huggingface_hub import hf_hub_download, model_info
        from mlx_vlm.utils import get_model_path, load, load_model
    except ImportError as exc:
        raise InstrumentationError(
            "Install the Mac dependencies with `uv sync --extra mac --extra dev`"
        ) from exc

    info = model_info(config.model_id, revision=config.revision)
    resolved_revision = str(info.sha)
    config_path = hf_hub_download(
        repo_id=config.model_id,
        filename="config.json",
        revision=resolved_revision,
    )
    bits, group_size = _quantization_metadata(config_path)
    if config.adapter == "ministral3":
        # mlx-vlm's streaming detokenizer currently assumes a `.vocab`
        # attribute, while Transformers 5 exposes Ministral through
        # MistralCommonBackend (`get_vocab()` instead).  We do not use the
        # streaming detokenizer, so load the exact same model and official
        # tokenizer directly rather than mutating either dependency.
        from transformers import AutoTokenizer

        model_path = get_model_path(config.model_id, revision=resolved_revision)
        model = load_model(model_path, lazy=False)
        processor = AutoTokenizer.from_pretrained(model_path)
    else:
        model, processor = load(config.model_id, revision=resolved_revision, lazy=False)
    tokenizer = (
        processor
        if config.adapter == "ministral3"
        else getattr(processor, "tokenizer", processor)
    )
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        raise InstrumentationError(
            f"MLX processor for {config.model_id} does not expose a tokenizer"
        )

    adapter = get_model_adapter(config.adapter)
    prompt_source = None
    if config.adapter == "ministral3":
        source_model_id = config.source_model_id or config.model_id
        source_info = model_info(source_model_id)
        prompt_path = hf_hub_download(
            repo_id=source_model_id,
            filename="SYSTEM_PROMPT.txt",
            revision=source_info.sha,
        )
        prompt_source = f"{source_model_id}@{source_info.sha}:SYSTEM_PROMPT.txt"
        adapter.configure_system_prompt(
            Path(prompt_path).read_text(encoding="utf-8"),
            source=prompt_source,
        )

    architecture = f"{type(model).__module__}.{type(model).__name__}"
    return MLXModelBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        adapter=adapter,
        model_config=config,
        resolved_revision=resolved_revision,
        architecture=architecture,
        quantization_bits=bits,
        quantization_group_size=group_size,
        prompt_source=prompt_source,
    )


def unload_mlx_model_bundle(bundle: MLXModelBundle) -> None:
    del bundle.model
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass
