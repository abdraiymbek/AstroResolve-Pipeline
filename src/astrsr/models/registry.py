"""Model registry."""

from __future__ import annotations

from typing import Any

from astrsr.config import AppConfig
from astrsr.models.baselines import InterpolationModel, RichardsonLucyModel, WienerModel
from astrsr.models.fake import FakeSRModel
from astrsr.models.galaxy_restormer import GalaxyRestormerModel, weights_available
from astrsr.models.pretrained import Swin2SRModel


def build_named_model(name: str, config: AppConfig, device: str) -> Any:
    scale = 2
    fwhm = config.degradation.psf.fwhm_pixels
    if name in {"nearest", "bicubic", "lanczos"}:
        return InterpolationModel(name, scale=scale)
    if name == "wiener":
        return WienerModel(fwhm_pixels=fwhm, scale=scale)
    if name == "richardson_lucy":
        return RichardsonLucyModel(fwhm_pixels=fwhm, scale=scale)
    if name == "fake_sr":
        return FakeSRModel(
            noise_sigma=config.models.pretrained.noise_sigma,
            bias=config.models.pretrained.bias,
        )
    if name == "swin2sr_x2":
        return Swin2SRModel(checkpoint=config.models.pretrained.checkpoint, device=device)
    if name == "galaxy_restormer":
        return GalaxyRestormerModel(device=device)
    raise ValueError(f"Unknown model {name}")


def build_baseline(name: str, config: AppConfig) -> Any:
    return build_named_model(name, config, device="cpu")


def build_primary_model(config: AppConfig, device: str) -> Any:
    return build_named_model(config.models.pretrained.name, config, device)


def zoo_names(config: AppConfig) -> list[str]:
    names = list(config.ensemble.zoo) if config.ensemble.zoo else [config.models.pretrained.name]
    resolved: list[str] = []
    for name in names:
        if name == "galaxy_restormer" and not weights_available():
            print("skipping galaxy_restormer: weights not found under checkpoints/", flush=True)
            continue
        resolved.append(name)
    if not resolved:
        raise ValueError("No runnable models left in the ensemble zoo")
    return resolved


def build_zoo(config: AppConfig, device: str) -> list[Any]:
    return [build_named_model(name, config, device) for name in zoo_names(config)]
