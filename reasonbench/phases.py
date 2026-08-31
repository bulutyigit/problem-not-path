"""Phase state, gate validation, and artifact manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reasonbench.constants import (
    PHASE_IDS,
    SCIENTIFIC_OUTCOMES,
    TECHNICAL_STATUSES,
)
from reasonbench.exceptions import PhaseGateError
from reasonbench.storage import read_json, sha256_file, write_json_atomic


@dataclass
class PhaseStatus:
    """Machine-readable phase state."""

    phase_id: str
    technical_status: str
    scientific_outcome: str = "not_applicable"
    next_decision: str = ""
    completed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.phase_id not in PHASE_IDS:
            raise PhaseGateError(f"Unknown phase_id: {self.phase_id}")
        if self.technical_status not in TECHNICAL_STATUSES:
            raise PhaseGateError(f"Invalid technical_status: {self.technical_status}")
        if self.scientific_outcome not in SCIENTIFIC_OUTCOMES:
            raise PhaseGateError(f"Invalid scientific_outcome: {self.scientific_outcome}")
        if self.technical_status == "passed" and not self.next_decision:
            raise PhaseGateError("A passed phase must declare next_decision")

    def write(self, phase_directory: str | Path) -> Path:
        self.validate()
        return write_json_atomic(Path(phase_directory) / "phase_status.json", asdict(self))


def load_phase_status(phase_directory: str | Path) -> PhaseStatus:
    """Load and validate a phase status file."""

    data = read_json(Path(phase_directory) / "phase_status.json")
    status = PhaseStatus(**data)
    status.validate()
    return status


def require_phase_gate(
    artifacts_root: str | Path,
    previous_phase: str,
    allowed_decisions: set[str] | None = None,
) -> PhaseStatus:
    """Require a technically passed previous phase and an allowed decision."""

    status = load_phase_status(Path(artifacts_root) / previous_phase)
    if status.technical_status != "passed":
        raise PhaseGateError(
            f"{previous_phase} has not passed technically: {status.technical_status}"
        )
    if allowed_decisions and status.next_decision not in allowed_decisions:
        raise PhaseGateError(
            f"{previous_phase} decision {status.next_decision!r} is not one of "
            f"{sorted(allowed_decisions)}"
        )
    return status


def build_artifact_manifest(
    phase_directory: str | Path,
    exclude_names: set[str] | None = None,
) -> dict[str, Any]:
    """Build a recursive manifest of durable phase files."""

    root = Path(phase_directory)
    excluded = (exclude_names or set()) | {"artifacts_manifest.json"}
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded or path.name.startswith("."):
            continue
        if "trajectories" in path.parts and path.name != "complete.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "phase_directory": str(root),
        "created_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    write_json_atomic(root / "artifacts_manifest.json", manifest)
    return manifest
