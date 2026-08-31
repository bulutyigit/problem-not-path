"""Project-wide constants."""

from __future__ import annotations

PACKAGE_VERSION = "0.1.0"
DEFAULT_PROJECT_NAME = "how_models_reason"
DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/how_models_reason"
DEFAULT_ARTIFACTS_DIRECTORY = "artifacts"
DEFAULT_SEEDS = (11, 23, 37, 53)
DEFAULT_SPLIT_SEED = 20260728

PHASE_IDS = tuple(f"phase_{index:02d}" for index in range(8)) + (
    "phase_04b",
    "phase_04c",
    "phase_04d",
    "phase_04e",
    "phase_04f",
)

# Phase 4 uses two deliberately different clocks. Dense prefixes are derived
# locally from an already-generated trajectory; sparse anchors launch new,
# expensive continuations to identify the stable-success transition.
PHASE4_DENSE_PREFIXES = (
    16,
    32,
    64,
    *sorted({*range(100, 501, 20), 128, 256}),
    512,
    768,
    1024,
    1536,
    2048,
    4096,
    8192,
)
PHASE4_BREAKTHROUGH_ANCHORS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
PHASE4_FORECAST_HORIZONS = (128, 256, 512)
PHASE4_CONTINUATIONS_PER_ANCHOR = 4
PHASE4_SUCCESS_BASIN_THRESHOLD = 0.75

TECHNICAL_STATUSES = frozenset({"passed", "failed", "incomplete"})
SCIENTIFIC_OUTCOMES = frozenset(
    {"positive", "negative", "limited", "underpowered", "inconclusive", "not_applicable"}
)

MODEL_IDS = {
    "gemma4_e4b": "google/gemma-4-E4B-it",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "ministral3_3b": "mistralai/Ministral-3-3B-Reasoning-2512",
}
