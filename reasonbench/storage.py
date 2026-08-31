"""Atomic and resumable artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from reasonbench.exceptions import StorageError


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its resolved path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    ensure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise StorageError(f"Could not write artifact atomically: {path}") from exc


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Write formatted JSON atomically."""

    output_path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    _atomic_replace_bytes(output_path, payload + b"\n")
    return output_path


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Write UTF-8 text atomically."""

    output_path = Path(path)
    _atomic_replace_bytes(output_path, text.encode("utf-8"))
    return output_path


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact."""

    input_path = Path(path)
    try:
        return json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Could not read JSON artifact: {input_path}") from exc


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_run_id(
    experiment_id: str,
    model_key: str,
    dataset: str,
    problem_id: str,
    seed: int,
) -> str:
    """Create a stable trajectory identifier."""

    raw = f"{experiment_id}|{model_key}|{dataset}|{problem_id}|{seed}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
