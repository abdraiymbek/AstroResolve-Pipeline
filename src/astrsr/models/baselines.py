"""Interpolation and classical restoration at a requested integer scale."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom
from skimage.restoration import richardson_lucy

from astrsr.degradation.psf import gaussian_psf
from astrsr.utils.arrays import as_float_image


def _lanczos_kernel(delta: np.ndarray, a: int = 3) -> np.ndarray:
    ax = np.abs(delta)
    out = np.zeros_like(ax, dtype=np.float64)
    nz = (ax > 1e-12) & (ax < a)
    out[ax <= 1e-12] = 1.0
    t = ax[nz]
    out[nz] = np.sinc(t) * np.sinc(t / a)
    return out


def lanczos_zoom(image: np.ndarray, scale: int, a: int = 3) -> np.ndarray:
    data = as_float_image(image)
    out_h = data.shape[0] * scale
    out_w = data.shape[1] * scale

    def resample_axis(arr: np.ndarray, axis: int, n_out: int) -> np.ndarray:
        n_in = arr.shape[axis]
        coords = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        left = np.floor(coords).astype(int) - a + 1
        taps = np.arange(2 * a)
        indices = left[:, None] + taps[None, :]
        weights = _lanczos_kernel(coords[:, None] - indices, a=a)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum[weight_sum == 0] = 1.0
        weights = weights / weight_sum
        clipped = np.clip(indices, 0, n_in - 1)
        if axis == 1:
            sampled = arr[:, clipped]
            return np.einsum("hok,ok->ho", sampled, weights)
        sampled = arr[clipped, :]
        return np.einsum("okw,ok->ow", sampled, weights)

    tmp = resample_axis(data, axis=1, n_out=out_w)
    return as_float_image(resample_axis(tmp, axis=0, n_out=out_h))


def interpolate(image: np.ndarray, scale: int, method: str) -> np.ndarray:
    data = as_float_image(image)
    if method == "nearest":
        return as_float_image(zoom(data, scale, order=0, prefilter=False))
    if method == "bicubic":
        return as_float_image(zoom(data, scale, order=3, prefilter=True))
    if method == "lanczos":
        return lanczos_zoom(data, scale)
    raise ValueError(f"Unknown interpolation {method}")


def _pad_psf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, -kh // 2, axis=0)
    padded = np.roll(padded, -kw // 2, axis=1)
    return padded


def wiener_deconv(image: np.ndarray, fwhm_pixels: float, k: float = 0.02) -> np.ndarray:
    data = as_float_image(image)
    psf = gaussian_psf(fwhm_pixels)
    h = np.fft.rfft2(_pad_psf(psf, data.shape))
    y = np.fft.rfft2(data)
    gain = np.conj(h) / (np.abs(h) ** 2 + k)
    restored = np.fft.irfft2(gain * y, s=data.shape)
    return as_float_image(np.real(restored))


def richardson_lucy_deconv(image: np.ndarray, fwhm_pixels: float, iterations: int = 10) -> np.ndarray:
    data = np.clip(as_float_image(image), 0.0, None)
    psf = gaussian_psf(fwhm_pixels)
    restored = richardson_lucy(data, psf, num_iter=iterations, clip=False)
    return as_float_image(restored)


class InterpolationModel:
    def __init__(self, method: str, scale: int = 2) -> None:
        self.name = method
        self.scale = scale
        self.license = "numpy/scipy"

    def infer(self, image: np.ndarray) -> np.ndarray:
        return interpolate(image, self.scale, self.name)

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "scale": self.scale, "kind": "interpolation"}


class WienerModel:
    def __init__(self, fwhm_pixels: float, scale: int = 2, k: float = 0.02) -> None:
        self.name = "wiener"
        self.scale = scale
        self.license = "scipy"
        self.fwhm_pixels = fwhm_pixels
        self.k = k

    def infer(self, image: np.ndarray) -> np.ndarray:
        restored = wiener_deconv(image, self.fwhm_pixels, k=self.k)
        return interpolate(restored, self.scale, "bicubic")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scale": self.scale,
            "kind": "classical_restoration",
            "fwhm_pixels": self.fwhm_pixels,
            "k": self.k,
            "upsample_after": "bicubic",
        }


class RichardsonLucyModel:
    def __init__(self, fwhm_pixels: float, scale: int = 2, iterations: int = 10) -> None:
        self.name = "richardson_lucy"
        self.scale = scale
        self.license = "scikit-image"
        self.fwhm_pixels = fwhm_pixels
        self.iterations = iterations

    def infer(self, image: np.ndarray) -> np.ndarray:
        restored = richardson_lucy_deconv(image, self.fwhm_pixels, iterations=self.iterations)
        return interpolate(restored, self.scale, "bicubic")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scale": self.scale,
            "kind": "classical_restoration",
            "fwhm_pixels": self.fwhm_pixels,
            "iterations": self.iterations,
            "upsample_after": "bicubic",
        }
