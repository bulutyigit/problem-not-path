"""Spectral summaries of scalar generation trajectories."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def summarize_spectrum(
    values: Any,
    prefix: str,
    interpolation_points: int = 256,
    minimum_length: int = 63,
) -> dict[str, float]:
    """Compute normalized RFFT features after robust resampling."""

    names = (
        "low_energy_ratio",
        "mid_energy_ratio",
        "high_energy_ratio",
        "dominant_frequency",
        "centroid",
        "bandwidth",
        "entropy",
        "flatness",
    )
    # KL/JS and step-based geometry have one structural first-token gap.  A
    # 64-token prefix therefore contributes 63 finite observations; requiring
    # 64 silently erased its first valid spectral feature block.
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < minimum_length:
        return {f"{prefix}_{name}": math.nan for name in names}
    median = np.median(array)
    scale = np.median(np.abs(array - median))
    if scale < 1e-12:
        return {f"{prefix}_{name}": 0.0 for name in names}
    normalized = (array - median) / scale
    source = np.linspace(0.0, 1.0, num=len(normalized))
    target = np.linspace(0.0, 1.0, num=interpolation_points)
    resampled = np.interp(target, source, normalized)
    windowed = resampled * np.hanning(interpolation_points)
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    power[0] = 0.0
    total = float(power.sum())
    if total <= 1e-20:
        return {f"{prefix}_{name}": 0.0 for name in names}
    frequencies = np.fft.rfftfreq(interpolation_points)
    normalized_power = power / total
    low = frequencies <= 0.10
    mid = (frequencies > 0.10) & (frequencies <= 0.25)
    high = frequencies > 0.25
    centroid = float(np.sum(frequencies * normalized_power))
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * normalized_power)))
    positive = normalized_power[normalized_power > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / np.log(len(normalized_power)))
    nonzero_power = power[1:]
    flatness = float(
        np.exp(np.mean(np.log(nonzero_power + 1e-20))) / (np.mean(nonzero_power) + 1e-20)
    )
    result = {
        "low_energy_ratio": float(normalized_power[low].sum()),
        "mid_energy_ratio": float(normalized_power[mid].sum()),
        "high_energy_ratio": float(normalized_power[high].sum()),
        "dominant_frequency": float(frequencies[int(np.argmax(power))]),
        "centroid": centroid,
        "bandwidth": bandwidth,
        "entropy": entropy,
        "flatness": flatness,
    }
    return {f"{prefix}_{key}": value for key, value in result.items()}
