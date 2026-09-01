from pathlib import Path

import numpy as np

from astrsr.config import apply_overrides, load_yaml_config
from astrsr.pipeline import run_experiment


def test_model_zoo_compare_fake(tmp_path: Path) -> None:
    cfg = load_yaml_config(Path("configs/p0_zoo_fake.yaml"))
    cfg = apply_overrides(cfg, [f"run.output_dir={tmp_path}"])
    result = run_experiment(cfg, repo_root=Path.cwd(), overrides=[])
    names = {row["name"] for row in result["solely"]}
    assert names == {"fake_sr", "bicubic", "lanczos"}
    recon = np.load(Path(result["run_dir"]) / "solely" / "bicubic" / "reconstruction.npy")
    obs = np.load(Path(result["run_dir"]) / "data" / "y_observation.npy")
    assert recon.shape[0] == obs.shape[0] * 2
    assert recon.shape == tuple(result["accepted"]["shape"])
    bicubic_row = next(row for row in result["solely"] if row["name"] == "bicubic")
    assert bicubic_row["upsample_mode"] == "direct_interpolation"
    assert bicubic_row["uses_gated_algorithm"] is False
    fake_row = next(row for row in result["solely"] if row["name"] == "fake_sr")
    assert fake_row["upsample_mode"] == "chained_own_infer"
    assert fake_row["uses_gated_algorithm"] is False
    assert result["steps"]
    assert result["model"]["ensemble_mode"] == "model_zoo"
    assert result["steps"][0]["n_members"] == 3


def test_zoo_override_list() -> None:
    cfg = apply_overrides(
        load_yaml_config(Path("configs/p0_zoo_fake.yaml")),
        ["ensemble.zoo=fake_sr,bicubic"],
    )
    assert cfg.ensemble.zoo == ["fake_sr", "bicubic"]
