"""Durable per-trajectory storage."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reasonbench.generation.engine import GenerationResult
from reasonbench.storage import ensure_directory, read_json, sha256_file, write_json_atomic


def trajectory_is_complete(directory: str | Path) -> bool:
    """Return whether a trajectory has a committed completion marker."""

    root = Path(directory)
    return (
        (root / "complete.json").exists()
        and (root / "metadata.json").exists()
        and (root / "token_metrics.parquet").exists()
    )


def verify_trajectory_payload(directory: str | Path) -> bool:
    """Verify every committed trajectory payload against its content hash.

    ``complete.json`` is an integrity boundary, not just a completion flag.
    A resumable job must therefore reject a trajectory when a payload has been
    truncated, replaced, or otherwise changed after the marker was written.
    """

    root = Path(directory)
    if not trajectory_is_complete(root):
        return False
    try:
        completion = read_json(root / "complete.json")
    except Exception:
        return False
    files = completion.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for filename, record in files.items():
        if not isinstance(record, dict):
            return False
        payload = root / filename
        expected_size = record.get("size_bytes")
        expected_hash = record.get("sha256")
        if (
            not payload.is_file()
            or not isinstance(expected_size, int)
            or not isinstance(expected_hash, str)
            or payload.stat().st_size != expected_size
            or sha256_file(payload) != expected_hash
        ):
            return False
    return True


def trajectory_matches_metadata(
    directory: str | Path,
    expected_metadata: dict[str, Any],
) -> bool:
    """Return whether a committed trajectory has matching immutable identity.

    This intentionally compares only the immutable provenance fields supplied
    by the caller.  Generated text, timestamps, and measured VRAM are outputs,
    so they must never participate in resume compatibility.
    """

    root = Path(directory)
    if not verify_trajectory_payload(root):
        return False
    try:
        metadata = read_json(root / "metadata.json")
    except Exception:
        return False
    return all(metadata.get(key) == value for key, value in expected_metadata.items())


def materialize_reused_trajectory(
    source_directory: str | Path,
    destination_directory: str | Path,
) -> Path:
    """Atomically expose an accepted external trajectory in a new experiment tree."""

    source = Path(source_directory)
    destination = Path(destination_directory)
    if not verify_trajectory_payload(source):
        raise ValueError(f"External reuse source is incomplete or corrupt: {source}")
    if trajectory_is_complete(destination):
        if not verify_trajectory_payload(destination):
            raise ValueError(f"External reuse destination is corrupt: {destination}")
        return destination
    if destination.exists():
        raise ValueError(f"External reuse destination exists but is incomplete: {destination}")
    ensure_directory(destination.parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        completion = read_json(source / "complete.json")
        filenames = [*completion.get("files", {}), "complete.json"]
        for filename in filenames:
            source_path = source / filename
            destination_path = temporary / filename
            try:
                os.link(source_path, destination_path)
            except OSError:
                shutil.copy2(source_path, destination_path)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    ensure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_trajectory(
    directory: str | Path,
    metadata: dict[str, Any],
    result: GenerationResult,
) -> Path:
    """Commit one trajectory atomically enough for resumable Colab execution."""

    root = ensure_directory(directory)
    (root / "complete.json").unlink(missing_ok=True)
    write_json_atomic(root / "metadata.json", metadata)
    signal_frame = pd.DataFrame([signal.to_dict() for signal in result.signals])
    _write_parquet_atomic(signal_frame, root / "token_metrics.parquet")
    if result.hidden_states:
        stacked = np.stack(
            [
                hidden.detach().float().cpu().numpy()
                if hasattr(hidden, "detach")
                else np.asarray(hidden)
                for hidden in result.hidden_states
            ]
        ).astype(np.float16, copy=False)
        _write_npz_atomic(
            root / "hidden_states.npz",
            token_indices=np.asarray(result.hidden_state_indices, dtype=np.int32),
            hidden_states=stacked,
        )
    artifact_files = ["metadata.json", "token_metrics.parquet"]
    if (root / "hidden_states.npz").exists():
        artifact_files.append("hidden_states.npz")
    write_json_atomic(
        root / "complete.json",
        {
            "run_id": metadata["run_id"],
            "complete": True,
            "token_metric_rows": len(signal_frame),
            "hidden_state_rows": len(result.hidden_states),
            "files": {
                filename: {
                    "size_bytes": (root / filename).stat().st_size,
                    "sha256": sha256_file(root / filename),
                }
                for filename in artifact_files
            },
        },
    )
    return root
