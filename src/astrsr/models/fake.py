"""Deterministic 2x stub used in tests. Adds seeded noise so ensembles are real."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom

from astrsr.utils.arrays import as_float_image


class FakeSRModel:
    name = "fake_sr"
    scale = 2
    license = "test-only"

    def __init__(self, noise_sigma: float = 0.0, bias: float = 0.0) -> None:
        self.noise_sigma = noise_sigma
        self.bias = bias
        self._rng: np.random.Generator | None = None

    def set_rng(self, rng: np.random.Generator | None) -> None:
        self._rng = rng

    def infer(self, image: np.ndarray) -> np.ndarray:
        up = as_float_image(zoom(as_float_image(image), self.scale, order=3, prefilter=True))
        up = up + self.bias
        if self.noise_sigma > 0:
            rng = self._rng if self._rng is not None else np.random.default_rng(0)
            up = up + rng.normal(0.0, self.noise_sigma, size=up.shape)
        return as_float_image(up)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scale": self.scale,
            "kind": "test_stub",
            "noise_sigma": self.noise_sigma,
            "bias": self.bias,
        }
