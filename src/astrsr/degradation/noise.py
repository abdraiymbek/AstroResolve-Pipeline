"""Shot, read, and background noise operators."""

from __future__ import annotations

from typing import Any

import numpy as np

from astrsr.config import NoiseConfig
from astrsr.utils.arrays import as_float_image


def apply_noise(
    image: np.ndarray,
    noise: NoiseConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = as_float_image(image) + float(noise.background)
    if noise.shot:
        lam = np.clip(data, 0.0, None)
        data = rng.poisson(lam).astype(np.float64)
    if noise.read_sigma > 0:
        data = data + rng.normal(0.0, noise.read_sigma, size=data.shape)
    record = {
        "shot": noise.shot,
        "read_sigma": noise.read_sigma,
        "background": noise.background,
        "units_assumption": (
            "Shot noise treats pixel values as expected counts. "
            "This is a synthetic convention, not a calibrated gain model."
        ),
    }
    return as_float_image(data), record


def expected_sigma_map(image_noiseless: np.ndarray, noise: NoiseConfig) -> np.ndarray:
    """Per-pixel noise envelope used by the data-consistency chi-square."""
    mean = as_float_image(image_noiseless)
    variance = np.full_like(mean, noise.read_sigma**2)
    if noise.shot:
        variance = variance + np.clip(mean, 0.0, None)
    return np.sqrt(np.clip(variance, 1e-12, None))
