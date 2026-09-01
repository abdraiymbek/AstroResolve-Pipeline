"""Array helpers used across scientific I/O and display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from PIL import Image


def as_float_image(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim != 2:
        raise ValueError(f"Scientific images must be 2-D, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Image contains NaN or Inf")
    return data.astype(np.float64, copy=False)


def require_same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch {left.shape} vs {right.shape}")


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, as_float_image(array))


def save_fits(path: Path, array: np.ndarray, header: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(as_float_image(array))
    if header:
        for key, value in header.items():
            text = str(value)
            hdu.header[key.upper()[:8]] = text[:70]
    hdu.writeto(path, overwrite=True)


def load_fits_image(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data)
            if data.ndim == 2:
                return as_float_image(data)
            if data.ndim == 3:
                return as_float_image(np.mean(data, axis=0))
    raise ValueError(f"No 2-D image found in {path}")


def load_png_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("F"), dtype=np.float64)
    return as_float_image(array)


def asinh_stretch(array: np.ndarray, scale: float | None = None) -> np.ndarray:
    data = as_float_image(array)
    floor = np.percentile(data, 1.0)
    shifted = np.clip(data - floor, 0.0, None)
    if scale is None:
        finite = shifted[shifted > 0]
        scale = float(np.percentile(finite, 50.0)) if finite.size else 1.0
        scale = max(scale, 1e-12)
    stretched = np.arcsinh(shifted / scale)
    peak = stretched.max()
    if peak > 0:
        stretched = stretched / peak
    return stretched


def save_preview_png(path: Path, array: np.ndarray) -> dict[str, float]:
    stretched = asinh_stretch(array)
    pixels = np.clip(stretched * 255.0, 0.0, 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(path)
    return {"stretch": "asinh", "display_only": True}


def save_sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
