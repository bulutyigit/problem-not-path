"""Configuration loading, validation, and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from reasonbench.exceptions import ConfigurationError


@dataclass(frozen=True)
class SamplingConfig:
    """Autoregressive sampling configuration."""

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    do_sample: bool = True
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0

    def validate(self) -> None:
        if self.temperature < 0:
            raise ConfigurationError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ConfigurationError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ConfigurationError("top_k must be non-negative")
        if not 0 <= self.min_p <= 1:
            raise ConfigurationError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ConfigurationError("repetition_penalty must be positive")


@dataclass(frozen=True)
class ModelConfig:
    """Model loading and reasoning-control configuration."""

    key: str
    model_id: str
    backend: str = "transformers_cuda"
    source_model_id: str | None = None
    revision: str | None = None
    adapter: str = ""
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    mode: str = "reasoning"
    reasoning_budget: int | None = None
    reasoning_budget_policy: str = "none"
    final_answer_reserve: int = 256
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 8192
    hidden_state_stride: int = 8
    capture_hidden_states: bool = True
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def validate(self) -> None:
        if not self.key:
            raise ConfigurationError("model key is required")
        if not self.model_id:
            raise ConfigurationError("model_id is required")
        if self.backend not in {"transformers_cuda", "mlx_vlm"}:
            raise ConfigurationError(
                "backend must be 'transformers_cuda' or 'mlx_vlm'"
            )
        if self.backend == "transformers_cuda" and self.dtype != "bfloat16":
            raise ConfigurationError(
                "Transformers/CUDA model configurations must use bfloat16"
            )
        if self.backend == "mlx_vlm" and self.dtype != "int4":
            raise ConfigurationError("MLX Phase 4 model configurations must use int4")
        if self.source_model_id is not None and not self.source_model_id:
            raise ConfigurationError("source_model_id cannot be empty")
        if self.reasoning_budget is not None and self.reasoning_budget <= 0:
            raise ConfigurationError("reasoning_budget must be positive")
        if self.reasoning_budget_policy not in {"none", "external_hard_cap"}:
            raise ConfigurationError(
                "reasoning_budget_policy must be 'none' or 'external_hard_cap'"
            )
        if self.reasoning_budget is None and self.reasoning_budget_policy != "none":
            raise ConfigurationError(
                "reasoning_budget_policy must be 'none' when reasoning_budget is unset"
            )
        if self.reasoning_budget is not None:
            if self.reasoning_budget_policy != "external_hard_cap":
                raise ConfigurationError(
                    "Assigned budgets must declare reasoning_budget_policy='external_hard_cap'"
                )
            if self.mode != "reasoning":
                raise ConfigurationError("Assigned reasoning budgets require reasoning mode")
            if self.reasoning_budget + self.final_answer_reserve > self.max_new_tokens:
                raise ConfigurationError(
                    "max_new_tokens must cover reasoning_budget plus final_answer_reserve"
                )
        if self.final_answer_reserve < 0:
            raise ConfigurationError("final_answer_reserve must be non-negative")
        if self.max_prompt_tokens <= 0 or self.max_new_tokens <= 0:
            raise ConfigurationError("token limits must be positive")
        if self.hidden_state_stride <= 0:
            raise ConfigurationError("hidden_state_stride must be positive")
        self.sampling.validate()


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset sampling configuration."""

    name: str
    split: str = "test"
    sample_size: int = 100
    seed: int = 20260728
    levels: tuple[int, ...] = ()
    nested_base_sample_size: int | None = None

    def validate(self) -> None:
        if self.name not in {"gsm8k", "math", "harp"}:
            raise ConfigurationError(f"Unsupported dataset: {self.name}")
        if self.sample_size <= 0:
            raise ConfigurationError("sample_size must be positive")
        if len(set(self.levels)) != len(self.levels):
            raise ConfigurationError("Dataset levels must be unique")
        if self.levels:
            if self.name not in {"math", "harp"}:
                raise ConfigurationError("Difficulty levels are supported only for MATH/HARP")
            maximum_level = 5 if self.name == "math" else 6
            if any(level not in range(1, maximum_level + 1) for level in self.levels):
                raise ConfigurationError(
                    f"{self.name.upper()} levels must be integers from 1 through {maximum_level}"
                )
            if self.sample_size % len(self.levels):
                raise ConfigurationError(
                    "A level-balanced sample size must be divisible by the number of levels"
                )
        if self.nested_base_sample_size is not None:
            if self.name not in {"math", "harp"} or not self.levels:
                raise ConfigurationError(
                    "nested_base_sample_size requires level-balanced MATH sampling"
                )
            if not 0 < self.nested_base_sample_size < self.sample_size:
                raise ConfigurationError(
                    "nested_base_sample_size must be positive and smaller than sample_size"
                )
            if self.nested_base_sample_size % len(self.levels):
                raise ConfigurationError(
                    "nested_base_sample_size must be divisible by the number of levels"
                )


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete generation experiment configuration."""

    experiment_id: str
    phase_id: str
    model: ModelConfig
    datasets: tuple[DatasetConfig, ...]
    seeds: tuple[int, ...] = (11, 23, 37, 53)
    prompt_version: str = "v1"
    output_subdirectory: str = ""

    def validate(self) -> None:
        if not self.experiment_id:
            raise ConfigurationError("experiment_id is required")
        if not self.phase_id.startswith("phase_"):
            raise ConfigurationError("phase_id must use the phase_XX format")
        self.model.validate()
        if not self.datasets:
            raise ConfigurationError("At least one dataset is required")
        for dataset in self.datasets:
            dataset.validate()
        if not self.seeds:
            raise ConfigurationError("At least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigurationError("Seeds must be unique")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sampling_from_dict(data: dict[str, Any] | None) -> SamplingConfig:
    return SamplingConfig(**(data or {}))


def experiment_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Construct and validate an experiment configuration."""

    try:
        model_data = dict(data["model"])
        model_data["sampling"] = _sampling_from_dict(model_data.get("sampling"))
        datasets = tuple(
            DatasetConfig(
                **{
                    **item,
                    "levels": tuple(int(level) for level in item.get("levels", ())),
                }
            )
            for item in data["datasets"]
        )
        seeds = tuple(int(seed) for seed in data.get("seeds", (11, 23, 37, 53)))
        config = ExperimentConfig(
            experiment_id=data["experiment_id"],
            phase_id=data["phase_id"],
            model=ModelConfig(**model_data),
            datasets=datasets,
            seeds=seeds,
            prompt_version=data.get("prompt_version", "v1"),
            output_subdirectory=data.get("output_subdirectory", ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid experiment configuration: {exc}") from exc
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")
    return experiment_from_dict(data)


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a standalone model YAML file."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Model configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Model configuration root must be a mapping: {config_path}")
    try:
        model_data = dict(data)
        model_data["sampling"] = _sampling_from_dict(model_data.get("sampling"))
        config = ModelConfig(**model_data)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid model configuration: {exc}") from exc
    config.validate()
    return config
