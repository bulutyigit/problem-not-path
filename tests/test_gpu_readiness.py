from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from reasonbench.colab import assert_supported_gpu
from reasonbench.exceptions import ReasonBenchError


def _fake_torch(device_name: str, memory_gib: float = 80.0) -> SimpleNamespace:
    properties = SimpleNamespace(
        name=device_name,
        total_memory=int(memory_gib * 1024**3),
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _index: properties,
        is_bf16_supported=lambda: True,
    )
    return SimpleNamespace(cuda=cuda)


@pytest.mark.parametrize(
    ("device_name", "memory_gib"),
    [
        ("NVIDIA A100-SXM4-80GB", 80.0),
        ("NVIDIA H100 80GB HBM3", 80.0),
        ("NVIDIA H200", 141.0),
        ("NVIDIA GH200 480GB", 96.0),
    ],
)
def test_supported_datacenter_gpus_pass_readiness(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    memory_gib: float,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(device_name, memory_gib))

    result = assert_supported_gpu()

    assert result["device_name"] == device_name
    assert result["total_memory_gib"] == pytest.approx(memory_gib)


def test_unsupported_gpu_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("NVIDIA L40S", 48.0))

    with pytest.raises(ReasonBenchError, match="A100, H100, or H200"):
        assert_supported_gpu()
