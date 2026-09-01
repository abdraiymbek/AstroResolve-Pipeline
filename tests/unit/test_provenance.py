from pathlib import Path

import pytest

from astrsr.config import AppConfig, apply_overrides
from astrsr.provenance import create_run_dir


def test_run_dir_refuses_overwrite(tmp_path: Path) -> None:
    create_run_dir(tmp_path, "run_a")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "run_a")


def test_seed_override_roundtrip() -> None:
    cfg = apply_overrides(AppConfig(), ["run.seed=99", "run.device=cpu"])
    assert cfg.run.seed == 99
    assert cfg.run.device == "cpu"
