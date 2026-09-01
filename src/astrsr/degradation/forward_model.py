"""Synthetic measurement model: PSF, sampling, noise."""

from __future__ import annotations

from typing import Any

import numpy as np

from astrsr.config import DegradationConfig
from astrsr.degradation.noise import apply_noise, expected_sigma_map
from astrsr.degradation.psf import convolve_psf, gaussian_psf
from astrsr.degradation.sampling import block_reduce
from astrsr.utils.arrays import as_float_image


def deterministic_forward(
    image: np.ndarray,
    config: DegradationConfig,
    scale_factor: int,
    fwhm_in_image_pixels: float,
) -> np.ndarray:
    """PSF + downsample + background, no stochastic noise draw."""
    data = as_float_image(image)
    kernel = gaussian_psf(fwhm_in_image_pixels)
    blurred = convolve_psf(data, kernel)
    sampled = block_reduce(blurred, scale_factor)
    return as_float_image(sampled + config.noise.background)


def degrade_reference(
    reference: np.ndarray,
    config: DegradationConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not config.enabled:
        data = as_float_image(reference)
        return data, {"enabled": False, "seed": seed}
    fwhm_on_reference = config.psf.fwhm_pixels * config.scale_factor
    kernel = gaussian_psf(fwhm_on_reference)
    blurred = convolve_psf(reference, kernel)
    if config.sampling.pixel_integration:
        sampled = block_reduce(blurred, config.scale_factor)
    else:
        sampled = as_float_image(blurred[:: config.scale_factor, :: config.scale_factor])
    rng = np.random.default_rng(seed)
    noisy, noise_record = apply_noise(sampled, config.noise, rng)
    record = {
        "enabled": True,
        "seed": seed,
        "scale_factor": config.scale_factor,
        "psf": {
            "type": config.psf.type,
            "fwhm_pixels_on_observation": config.psf.fwhm_pixels,
            "fwhm_pixels_on_reference": fwhm_on_reference,
        },
        "sampling": config.sampling.model_dump(mode="json"),
        "noise": noise_record,
        "artifacts": config.artifacts.model_dump(mode="json"),
        "input_shape": list(reference.shape),
        "output_shape": list(noisy.shape),
        "equation": "y = downsample(psf(x)) + background + shot + read",
    }
    return as_float_image(noisy), record


def forward_project_to_observation(
    reconstruction: np.ndarray,
    observation: np.ndarray,
    config: DegradationConfig,
    total_scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blur and shrink a reconstruction back to the original observation grid."""
    rec = as_float_image(reconstruction)
    obs = as_float_image(observation)
    expected_h = obs.shape[0] * total_scale
    expected_w = obs.shape[1] * total_scale
    if rec.shape != (expected_h, expected_w):
        raise ValueError(
            f"Reconstruction shape {rec.shape} does not match observation "
            f"{obs.shape} at total scale {total_scale}"
        )
    fwhm_on_reconstruction = config.psf.fwhm_pixels * total_scale
    projected = deterministic_forward(
        rec,
        config,
        scale_factor=total_scale,
        fwhm_in_image_pixels=fwhm_on_reconstruction,
    )
    return as_float_image(projected), expected_sigma_map(projected, config.noise)
