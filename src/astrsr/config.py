"""Typed experiment configuration. Unknown keys are rejected."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(FrozenModel):
    name: str = "p0_smoke"
    seed: int = 20260831
    output_dir: str = "runs"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    deterministic: bool = True


class DataConfig(FrozenModel):
    source: Literal["synthetic_fixture", "fits", "png"] = "synthetic_fixture"
    fixture: Literal["point_sources", "blob", "ring", "galaxy"] = "galaxy"
    size: int = 128
    path: str | None = None
    channels: list[str] = Field(default_factory=lambda: ["intensity"])
    preserve_units: bool = True
    normalization: Literal["documented_linear"] = "documented_linear"
    split: Literal["train", "validation", "test"] = "test"


class PsfConfig(FrozenModel):
    type: Literal["gaussian"] = "gaussian"
    fwhm_pixels: float = 1.5


class SamplingConfig(FrozenModel):
    pixel_integration: bool = True
    kernel: Literal["area"] = "area"


class NoiseConfig(FrozenModel):
    shot: bool = True
    read_sigma: float = 0.01
    background: float = 0.02


class ArtifactConfig(FrozenModel):
    cosmic_ray_probability: float = 0.0
    saturation: bool = False


class DegradationConfig(FrozenModel):
    enabled: bool = True
    scale_factor: int = 2
    psf: PsfConfig = Field(default_factory=PsfConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)

    @field_validator("scale_factor")
    @classmethod
    def scale_must_be_positive(cls, value: int) -> int:
        if value < 2:
            raise ValueError("degradation.scale_factor must be >= 2")
        return value


class PretrainedModelConfig(FrozenModel):
    name: str = "swin2sr_x2"
    checkpoint: str = "caidas/swin2SR-classical-sr-x2-64"
    scale: int = 2
    license: str = "Apache-2.0"
    noise_sigma: float = 0.0
    bias: float = 0.0


class ModelsConfig(FrozenModel):
    baselines: list[Literal["nearest", "bicubic", "lanczos", "wiener", "richardson_lucy"]] = Field(
        default_factory=lambda: ["bicubic", "lanczos", "wiener"]
    )
    pretrained: PretrainedModelConfig = Field(default_factory=PretrainedModelConfig)


class StochasticityConfig(FrozenModel):
    source: Literal["tta_and_input_noise", "model_zoo"] = "tta_and_input_noise"
    input_sigma_rel: float = 0.02
    tta: bool = True
    seed_stride: int = 1000
    tta_per_member: int = 1


class FilteringConfig(FrozenModel):
    method: Literal["mad", "none"] = "mad"
    threshold: float = 3.5


class ConsensusConfig(FrozenModel):
    method: Literal["median", "trimmed_mean"] = "median"
    trim_fraction: float = 0.1


class UncertaintyConfig(FrozenModel):
    maps: list[Literal["std", "mad", "percentile_interval"]] = Field(
        default_factory=lambda: ["std", "mad", "percentile_interval"]
    )
    percentile_low: float = 10.0
    percentile_high: float = 90.0


class EnsembleConfig(FrozenModel):
    enabled: bool = True
    mode: Literal["stochastic_single", "model_zoo"] = "stochastic_single"
    zoo: list[str] = Field(default_factory=list)
    samples: int = 8
    stochasticity: StochasticityConfig = Field(default_factory=StochasticityConfig)
    retain_members: bool = True
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)

    @field_validator("samples")
    @classmethod
    def samples_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("ensemble.samples must be >= 1")
        return value


class GateConfig(FrozenModel):
    min_agreement: float = 0.85
    max_relative_uncertainty: float = 0.20
    require_forward_consistency: bool = True
    max_reduced_chi2: float = 2.5
    require_photometric_check: bool = True
    max_flux_rel_error: float = 0.05
    max_metric_regression: float = 0.02


class SpatialConfig(FrozenModel):
    enabled: bool = True
    retry_failed_tiles: bool = True
    min_tile: int = 16
    overlap: int = 4
    max_retries: int = 1
    min_success_fraction_to_continue: float = 0.25
    max_residual_sigma: float = 2.5

    @field_validator("min_tile")
    @classmethod
    def min_tile_positive(cls, value: int) -> int:
        if value < 2:
            raise ValueError("recursion.spatial.min_tile must be >= 2")
        return value

    @field_validator("overlap")
    @classmethod
    def overlap_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("recursion.spatial.overlap must be >= 0")
        return value

    @field_validator("max_retries")
    @classmethod
    def retries_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("recursion.spatial.max_retries must be >= 0")
        return value

    @field_validator("min_success_fraction_to_continue")
    @classmethod
    def continue_fraction_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("recursion.spatial.min_success_fraction_to_continue must be in [0, 1]")
        return value

    @field_validator("max_residual_sigma")
    @classmethod
    def residual_sigma_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("recursion.spatial.max_residual_sigma must be > 0")
        return value


class RecursionConfig(FrozenModel):
    enabled: bool = True
    unconditional: bool = False
    factors: list[int] = Field(default_factory=lambda: [2, 2])
    max_depth: int = 2
    gates: GateConfig = Field(default_factory=GateConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)

    @field_validator("factors")
    @classmethod
    def factors_are_two(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("recursion.factors must not be empty")
        for factor in value:
            if factor != 2:
                raise ValueError("P0 only supports 2x recursion factors")
        return value


class EvaluationConfig(FrozenModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["psnr", "ssim", "flux_error", "centroid_error"]
    )
    save_error_maps: bool = True
    compare_to_reference: bool = True
    ssim_win_size: int = 7


class LoggingConfig(FrozenModel):
    save_config: bool = True
    save_environment: bool = True
    export_elabftw: bool = True
    include_failure_cases: bool = True
    save_previews: bool = True


class AppConfig(FrozenModel):
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    recursion: RecursionConfig = Field(default_factory=RecursionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def config_hash(config: AppConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_yaml_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return AppConfig.model_validate(raw)


def _parse_override_value(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None
    try:
        if any(char in text for char in ".eE") and text.replace(".", "", 1).replace("e-", "", 1).replace("e+", "", 1).replace("E-", "", 1).replace("E+", "", 1).lstrip("-").isdigit():
            return float(text)
        return int(text)
    except ValueError:
        return text


def apply_overrides(config: AppConfig, overrides: list[str]) -> AppConfig:
    data = config.model_dump(mode="json")
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be path=value, got {item!r}")
        path, raw_value = item.split("=", 1)
        keys = path.split(".")
        cursor: Any = data
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                raise ValueError(f"Unknown override path {path}")
            cursor = cursor[key]
        leaf = keys[-1]
        if leaf not in cursor:
            raise ValueError(f"Unknown override path {path}")
        value = _parse_override_value(raw_value)
        if isinstance(cursor[leaf], list) and isinstance(value, str):
            cursor[leaf] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            cursor[leaf] = value
    return AppConfig.model_validate(data)
