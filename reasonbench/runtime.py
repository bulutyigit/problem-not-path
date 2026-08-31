"""Runtime inspection and reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from reasonbench.storage import write_json_atomic


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed available RNGs and optionally request best-effort deterministic kernels.

    Seeding alone fixes the sampling stream but not kernel reductions, so two
    runs can diverge at high-entropy sampling steps. ``deterministic=True``
    additionally requests deterministic CUDA algorithms with ``warn_only`` so
    ops without a deterministic implementation still run. Even then,
    reproducibility holds at most within one GPU model, driver, and library
    stack; it is never guaranteed across machines.
    """

    random.seed(seed)
    np.random.seed(seed)
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except Exception:
        # MLX is optional and Metal is intentionally unavailable in some
        # headless validation environments.
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS reads this at first handle creation; setting it here is a
        # best effort for later handles, so callers should also export it
        # before any CUDA work (generate.py does this at startup).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def package_versions(names: list[str]) -> dict[str, str | None]:
    """Return installed package versions without importing packages."""

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_revision(project_root: str | Path) -> str | None:
    """Return the current Git revision when available."""

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_tree_revision(project_root: str | Path) -> str:
    """Hash source and configuration files while ignoring notebook outputs."""

    root = Path(project_root)
    ignored_directories = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "artifacts",
    }
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in ignored_directories for part in path.parts):
            continue
        relative = path.relative_to(root)
        digest.update(str(relative).encode("utf-8"))
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["execution_count"] = None
                    cell["outputs"] = []
            digest.update(
                json.dumps(
                    notebook,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            continue
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def cuda_snapshot() -> dict[str, Any]:
    """Collect CUDA and GPU information when Torch is available."""

    try:
        import torch
    except ImportError:
        return {"torch_available": False, "cuda_available": False}

    snapshot: dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        snapshot.update(
            {
                "device_name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            }
        )
    return snapshot


def write_runtime_manifest(
    output_path: str | Path,
    project_root: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a reproducibility manifest."""

    git_commit = git_revision(project_root)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(Path(project_root)),
        "git_revision": git_commit,
        "source_tree_sha256": source_tree_revision(project_root),
        "cuda": cuda_snapshot(),
        "packages": package_versions(
            [
                "accelerate",
                "datasets",
                "huggingface-hub",
                "mistral-common",
                "mlx",
                "mlx-lm",
                "mlx-vlm",
                "numpy",
                "pandas",
                "pillow",
                "pyarrow",
                "scikit-learn",
                "scipy",
                "torch",
                "torchvision",
                "transformers",
                "zarr",
            ]
        ),
        "environment": {
            "COLAB_RELEASE_TAG": os.environ.get("COLAB_RELEASE_TAG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if extra:
        manifest["extra"] = extra
    return write_json_atomic(output_path, manifest)
