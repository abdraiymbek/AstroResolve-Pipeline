"""Synthetic astronomical-like reference scenes with known structure."""

from __future__ import annotations

from typing import Any

import numpy as np

from astrsr.utils.arrays import as_float_image


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((size, size), dtype=np.float64)
    return yy, xx


def _gaussian(yy: np.ndarray, xx: np.ndarray, y0: float, x0: float, amp: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * (((yy - y0) / sigma) ** 2 + ((xx - x0) / sigma) ** 2))


def point_sources(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = _grid(size)
    image = np.zeros((size, size), dtype=np.float64)
    n_sources = 7
    for _ in range(n_sources):
        y0 = rng.uniform(size * 0.15, size * 0.85)
        x0 = rng.uniform(size * 0.15, size * 0.85)
        amp = rng.uniform(3.0, 12.0)
        sigma = rng.uniform(1.0, 1.8)
        image += _gaussian(yy, xx, y0, x0, amp, sigma)
    return as_float_image(image)


def blob(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = _grid(size)
    y0 = size * rng.uniform(0.45, 0.55)
    x0 = size * rng.uniform(0.45, 0.55)
    image = _gaussian(yy, xx, y0, x0, amp=8.0, sigma=size * 0.12)
    image += _gaussian(yy, xx, y0, x0, amp=3.0, sigma=size * 0.22)
    return as_float_image(image)


def ring(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = _grid(size)
    y0 = size * 0.5
    x0 = size * 0.5
    radius = size * rng.uniform(0.18, 0.22)
    width = size * 0.03
    rr = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)
    image = 5.0 * np.exp(-0.5 * ((rr - radius) / width) ** 2)
    image += _gaussian(yy, xx, y0, x0, amp=1.5, sigma=size * 0.06)
    return as_float_image(image)


def galaxy(size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = _grid(size)
    y0 = size * 0.5
    x0 = size * 0.5
    dy = yy - y0
    dx = xx - x0
    angle = rng.uniform(-0.4, 0.4)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    bulge = 6.0 * np.exp(-np.sqrt((xr / (size * 0.06)) ** 2 + (yr / (size * 0.06)) ** 2))
    disk = 2.8 * np.exp(-0.5 * ((xr / (size * 0.18)) ** 2 + (yr / (size * 0.10)) ** 2))
    rr = np.sqrt(xr**2 + yr**2) + 1e-6
    theta = np.arctan2(yr, xr)
    arms = 1.2 * np.exp(-rr / (size * 0.16)) * (1.0 + np.cos(2.0 * theta + rr * 0.18))
    arms = np.clip(arms, 0.0, None)
    image = bulge + disk + 0.55 * arms
    for _ in range(4):
        image += _gaussian(
            yy,
            xx,
            y0 + rng.uniform(-size * 0.12, size * 0.12),
            x0 + rng.uniform(-size * 0.16, size * 0.16),
            amp=rng.uniform(0.8, 2.2),
            sigma=rng.uniform(1.2, 2.4),
        )
    return as_float_image(image)


_BUILDERS = {
    "point_sources": point_sources,
    "blob": blob,
    "ring": ring,
    "galaxy": galaxy,
}


def generate_fixture(name: str, size: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if size < 32 or size % 2:
        raise ValueError("fixture size must be an even integer >= 32")
    if name not in _BUILDERS:
        raise ValueError(f"Unknown fixture {name}")
    rng = np.random.default_rng(seed)
    image = _BUILDERS[name](size, rng)
    meta = {
        "kind": "synthetic_fixture",
        "fixture": name,
        "size": size,
        "seed": seed,
        "units": "adu",
        "license": "generated in-pipeline, no third-party image license",
        "notes": "Scene is synthetic and known. It is not a real observation.",
    }
    return image, meta
