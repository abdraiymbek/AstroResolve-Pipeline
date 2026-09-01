from pathlib import Path

import pytest

from astrsr.config import AppConfig, apply_overrides, config_hash, load_yaml_config


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("run:\n  name: x\n  not_a_field: 1\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_yaml_config(path)


def test_override_samples() -> None:
    cfg = apply_overrides(AppConfig(), ["ensemble.samples=64", "recursion.max_depth=1"])
    assert cfg.ensemble.samples == 64
    assert cfg.recursion.max_depth == 1


def test_override_unknown_path() -> None:
    with pytest.raises(ValueError, match="Unknown override"):
        apply_overrides(AppConfig(), ["ensemble.not_real=1"])


def test_config_hash_stable() -> None:
    a = AppConfig()
    b = AppConfig()
    assert config_hash(a) == config_hash(b)
    c = apply_overrides(a, ["run.seed=1"])
    assert config_hash(a) != config_hash(c)


def test_p0_smoke_yaml_loads() -> None:
    cfg = load_yaml_config(Path("configs/p0_smoke.yaml"))
    assert cfg.ensemble.samples == 8
    assert cfg.models.pretrained.name == "swin2sr_x2"


def test_p0_fake_yaml_loads() -> None:
    cfg = load_yaml_config(Path("configs/p0_fake.yaml"))
    assert cfg.models.pretrained.name == "fake_sr"
    assert cfg.recursion.spatial.enabled is True


def test_spatial_override() -> None:
    cfg = apply_overrides(AppConfig(), ["recursion.spatial.enabled=false", "recursion.spatial.min_tile=8"])
    assert cfg.recursion.spatial.enabled is False
    assert cfg.recursion.spatial.min_tile == 8
