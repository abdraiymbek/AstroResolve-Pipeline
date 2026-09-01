from pathlib import Path

from astrsr.config import apply_overrides, load_yaml_config
from astrsr.pipeline import run_experiment


def test_pipeline_fake_sr_end_to_end(tmp_path: Path) -> None:
    cfg = load_yaml_config(Path("configs/p0_fake.yaml"))
    cfg = apply_overrides(
        cfg,
        [f"run.output_dir={tmp_path}", "ensemble.samples=3", "recursion.max_depth=1"],
    )
    result = run_experiment(cfg, repo_root=Path.cwd(), overrides=["test"])
    run_dir = Path(result["run_dir"])
    assert (run_dir / "COMPLETED").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "data" / "y_observation.npy").is_file()
    assert (run_dir / "steps" / "00" / "consensus.npy").is_file()
    assert (run_dir / "steps" / "00" / "gate.json").is_file()
    assert result["accepted"]["stop_reason"] in {
        "accepted_continue",
        "accepted_spatial",
        "max_depth_reached",
        "rejected_uncertainty",
        "rejected_forward_consistency",
        "rejected_photometry",
        "rejected_spatial",
    }
    assert (run_dir / "steps" / "00" / "success_mask.npy").is_file()
    assert (run_dir / "steps" / "00" / "mosaic.npy").is_file()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Results vs held-out reference" in report
    assert "accepted product" in report


def test_unstable_ensemble_fails_uncertainty(tmp_path: Path) -> None:
    cfg = load_yaml_config(Path("configs/p0_fake.yaml"))
    cfg = apply_overrides(
        cfg,
        [
            f"run.output_dir={tmp_path}",
            "models.pretrained.noise_sigma=3.0",
            "ensemble.samples=6",
            "ensemble.stochasticity.input_sigma_rel=0.4",
            "recursion.max_depth=1",
            "recursion.gates.min_agreement=0.99",
            "recursion.gates.max_relative_uncertainty=0.02",
            "recursion.gates.require_forward_consistency=false",
            "recursion.gates.require_photometric_check=false",
            "recursion.spatial.retry_failed_tiles=false",
        ],
    )
    result = run_experiment(cfg, repo_root=Path.cwd(), overrides=[])
    assert result["steps"][0]["decision"] == "rejected_uncertainty"
    assert result["accepted"]["depth"] == 0


def test_biased_reconstruction_fails_data_consistency(tmp_path: Path) -> None:
    cfg = load_yaml_config(Path("configs/p0_fake.yaml"))
    cfg = apply_overrides(
        cfg,
        [
            f"run.output_dir={tmp_path}",
            "models.pretrained.bias=100.0",
            "ensemble.samples=3",
            "ensemble.stochasticity.input_sigma_rel=0.0",
            "ensemble.stochasticity.tta=false",
            "recursion.max_depth=1",
            "recursion.gates.min_agreement=0.0",
            "recursion.gates.max_relative_uncertainty=10.0",
            "recursion.gates.require_forward_consistency=true",
            "recursion.gates.max_reduced_chi2=1.2",
            "recursion.gates.require_photometric_check=false",
            "recursion.spatial.retry_failed_tiles=false",
        ],
    )
    result = run_experiment(cfg, repo_root=Path.cwd(), overrides=[])
    assert result["steps"][0]["decision"] == "rejected_forward_consistency"
    assert result["steps"][0]["success_fraction"] == 0.0
    assert result["accepted"]["depth"] == 0
