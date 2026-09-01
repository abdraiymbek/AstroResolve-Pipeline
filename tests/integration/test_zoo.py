from pathlib import Path

from astrsr.config import apply_overrides, load_yaml_config
from astrsr.pipeline import run_experiment


def test_model_zoo_compare_fake(tmp_path: Path) -> None:
    cfg = load_yaml_config(Path("configs/p0_zoo_fake.yaml"))
    cfg = apply_overrides(cfg, [f"run.output_dir={tmp_path}"])
    result = run_experiment(cfg, repo_root=Path.cwd(), overrides=[])
    names = {row["name"] for row in result["solely"]}
    assert names == {"fake_sr", "bicubic", "lanczos"}
    assert (Path(result["run_dir"]) / "solely" / "bicubic" / "reconstruction.npy").is_file()
    assert result["steps"]
    assert result["model"]["ensemble_mode"] == "model_zoo"
    assert result["steps"][0]["n_members"] == 3


def test_zoo_override_list() -> None:
    cfg = apply_overrides(
        load_yaml_config(Path("configs/p0_zoo_fake.yaml")),
        ["ensemble.zoo=fake_sr,bicubic"],
    )
    assert cfg.ensemble.zoo == ["fake_sr", "bicubic"]
