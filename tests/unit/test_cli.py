from pathlib import Path

from typer.testing import CliRunner

from astrsr.cli import app

runner = CliRunner()


def test_validate_config() -> None:
    result = runner.invoke(app, ["validate-config", "--config", "configs/p0_fake.yaml"])
    assert result.exit_code == 0, result.output
    assert "config ok" in result.output


def test_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            "configs/p0_fake.yaml",
            "--dry-run",
            "--set",
            "ensemble.samples=5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run ok" in result.output
    assert "samples=5" in result.output


def test_cli_run_fake(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            "configs/p0_fake.yaml",
            "--set",
            f"run.output_dir={tmp_path}",
            "--set",
            "recursion.max_depth=1",
            "--set",
            "ensemble.samples=2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run_id=" in result.output
    run_id = [line.split("=", 1)[1] for line in result.output.splitlines() if line.startswith("run_id=")][0]
    report = runner.invoke(app, ["report", "--run-id", run_id, "--output-dir", str(tmp_path)])
    assert report.exit_code == 0, report.output
    assert "Research question" in report.output
    assert "Results vs held-out reference" in report.output
    assert "Results vs held-out reference" in result.output
    assert "| method | psnr |" in result.output
