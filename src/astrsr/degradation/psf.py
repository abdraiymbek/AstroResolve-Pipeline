"""Gaussian PSF kernels in observation or reconstruction pixels."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from astrsr.utils.arrays import as_float_image


def fwhm_to_sigma(fwhm: float) -> float:
    return float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian_psf(fwhm_pixels: float, min_radius: int = 1) -> np.ndarray:
    if fwhm_pixels <= 0:
        raise ValueError("PSF FWHM must be positive")
    sigma = fwhm_to_sigma(fwhm_pixels)
    radius = max(min_radius, int(np.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    yy, xx = np.indices((size, size), dtype=np.float64)
    cy = cx = radius
    kernel = np.exp(-0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2))
    kernel /= kernel.sum()
    return kernel


def convolve_psf(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    data = as_float_image(image)
    blurred = convolve(data, kernel, mode="nearest")
    return as_float_image(blurred)
