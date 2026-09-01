"""Area downsample / pixel integration."""

from __future__ import annotations

import numpy as np

from astrsr.utils.arrays import as_float_image


def block_reduce(image: np.ndarray, factor: int) -> np.ndarray:
    data = as_float_image(image)
    if factor < 1:
        raise ValueError("downsample factor must be >= 1")
    if factor == 1:
        return data.copy()
    height, width = data.shape
    if height % factor or width % factor:
        raise ValueError(
            f"Image shape {data.shape} is not divisible by downsample factor {factor}"
        )
    folded = data.reshape(height // factor, factor, width // factor, factor)
    return as_float_image(folded.mean(axis=(1, 3)))
