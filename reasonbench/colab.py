"""Google Colab and Google Drive bootstrap utilities."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from reasonbench.constants import (
    DEFAULT_ARTIFACTS_DIRECTORY,
    DEFAULT_DRIVE_PROJECT_ROOT,
)
from reasonbench.exceptions import ReasonBenchError
from reasonbench.storage import ensure_directory


@dataclass(frozen=True)
class ColabPaths:
    """Resolved Drive-first project paths."""

    project_root: Path
    artifacts_root: Path
    phase_root: Path
    shared_root: Path


def is_colab() -> bool:
    """Return whether the current process appears to run in Google Colab."""

    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def mount_google_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive when running in Colab."""

    if not is_colab():
        return
    from google.colab import drive

    drive.mount(mount_point, force_remount=False)


def load_huggingface_token_from_colab_secret() -> bool:
    """Load HF_TOKEN from Colab Secrets without printing it."""

    if not is_colab():
        return bool(os.environ.get("HF_TOKEN"))
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
    if token:
        os.environ["HF_TOKEN"] = token
        return True
    return False


def resolve_colab_paths(
    phase_id: str,
    project_root: str | Path = DEFAULT_DRIVE_PROJECT_ROOT,
) -> ColabPaths:
    """Validate the uploaded project and create durable artifact directories."""

    root = Path(project_root)
    if not (root / "pyproject.toml").exists() or not (root / "PLAN.md").exists():
        raise ReasonBenchError(
            f"ReasonBench project not found at {root}. Upload the complete folder or "
            "change PROJECT_ROOT in the notebook configuration cell."
        )
    artifacts_root = ensure_directory(root / DEFAULT_ARTIFACTS_DIRECTORY)
    return ColabPaths(
        project_root=root,
        artifacts_root=artifacts_root,
        phase_root=ensure_directory(artifacts_root / phase_id),
        shared_root=ensure_directory(artifacts_root / "shared"),
    )


def install_project(project_root: str | Path, extra: str = "colab") -> None:
    """Install pinned dependencies and the uploaded project in editable mode."""

    root = Path(project_root)
    lock_path = root / "requirements-colab.lock"
    if extra == "colab" and lock_path.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--requirement",
                str(lock_path),
            ],
            check=True,
        )
        target = str(root)
        no_dependencies = ["--no-deps"]
    else:
        target = f"{root}[{extra}]" if extra else str(root)
        no_dependencies = []
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            *no_dependencies,
            "-e",
            target,
        ],
        check=True,
    )


def assert_supported_gpu(minimum_memory_gib: float = 39.0) -> dict[str, str | float]:
    """Require one BF16-capable A100/H100/H200-class GPU with sufficient memory."""

    try:
        import torch
    except ImportError as exc:
        raise ReasonBenchError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise ReasonBenchError("CUDA is unavailable. Select an A100, H100, or H200 GPU runtime.")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    supported_accelerators = ("A100", "H100", "H200")
    if not any(name in properties.name for name in supported_accelerators) or (
        total_gib < minimum_memory_gib
    ):
        raise ReasonBenchError(
            "Expected an A100, H100, or H200 with at least "
            f"{minimum_memory_gib:.1f} GiB, "
            f"found {properties.name} with {total_gib:.1f} GiB."
        )
    if not torch.cuda.is_bf16_supported():
        raise ReasonBenchError("The selected GPU does not report BF16 support.")
    return {"device_name": properties.name, "total_memory_gib": total_gib}


def assert_a100(minimum_memory_gib: float = 39.0) -> dict[str, str | float]:
    """Backward-compatible alias for the supported datacenter GPU requirement."""

    return assert_supported_gpu(minimum_memory_gib)
