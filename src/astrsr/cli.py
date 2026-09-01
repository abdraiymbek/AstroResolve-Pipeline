"""Terminal interface for gated recursive astronomical super-resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from astrsr.config import apply_overrides, load_yaml_config
from astrsr.logging.report import (
    render_results_table,
    render_retry_table,
    results_table_rows,
    retry_history_rows,
)
from astrsr.pipeline import run_experiment

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False, add_completion=False)


def _echo_results_table(result: dict) -> None:
    typer.echo("")
    typer.echo("Results vs held-out reference")
    typer.echo("-" * 60)
    typer.echo(render_results_table(results_table_rows(result)))
    retry_rows = retry_history_rows(result)
    if retry_rows:
        typer.echo("")
        typer.echo("Retry trajectory")
        typer.echo("-" * 60)
        typer.echo(render_retry_table(retry_rows))
    typer.echo("-" * 60)
    typer.echo("Full report: report.md in run_dir (or: astrsr report --run-id <id>)")


def _repo_root() -> Path:
    return Path.cwd()


def _build_config(
    config_path: Path,
    sets: list[str] | None,
    seed: int | None,
    device: str | None,
):
    config = load_yaml_config(config_path)
    overrides = list(sets or [])
    if seed is not None:
        overrides.append(f"run.seed={seed}")
    if device is not None:
        overrides.append(f"run.device={device}")
    if overrides:
        config = apply_overrides(config, overrides)
    return config, overrides


@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(..., "--config", exists=True, readable=True, path_type=Path),
    set: Optional[list[str]] = typer.Option(None, "--set", help="Override path=value, repeatable"),
) -> None:
    loaded, _ = _build_config(config, set, None, None)
    typer.echo(f"config ok: {config}")
    typer.echo(f"run.name={loaded.run.name} ensemble.samples={loaded.ensemble.samples}")


@app.command("run")
def run(
    config: Path = typer.Option(..., "--config", exists=True, readable=True, path_type=Path),
    set: Optional[list[str]] = typer.Option(None, "--set", help="Override path=value, repeatable"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    device: Optional[str] = typer.Option(None, "--device"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    loaded, overrides = _build_config(config, set, seed, device)
    result = run_experiment(loaded, repo_root=_repo_root(), overrides=overrides, dry_run=dry_run)
    if dry_run:
        typer.echo("dry-run ok")
        typer.echo(f"planned_steps={result['planned_steps']} samples={result['ensemble_samples']}")
        return
    typer.echo(f"run_id={result['run_id']}")
    typer.echo(f"run_dir={result['run_dir']}")
    _echo_results_table(result)
    typer.echo(f"stop_reason={result['accepted']['stop_reason']}")
    typer.echo(f"accepted_depth={result['accepted']['depth']}")
    typer.echo(f"report={Path(result['run_dir']) / 'report.md'}")


@app.command("compare")
def compare(
    config: Path = typer.Option(
        Path("configs/p0_compare.yaml"),
        "--config",
        exists=True,
        readable=True,
        path_type=Path,
        help="Runs each zoo member solely, then the gated consensus on the same observation.",
    ),
    set: Optional[list[str]] = typer.Option(None, "--set"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    device: Optional[str] = typer.Option(None, "--device"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    loaded, overrides = _build_config(config, set, seed, device)
    if loaded.ensemble.mode != "model_zoo":
        loaded = apply_overrides(loaded, ["ensemble.mode=model_zoo"])
        overrides.append("ensemble.mode=model_zoo")
    result = run_experiment(loaded, repo_root=_repo_root(), overrides=overrides, dry_run=dry_run)
    if dry_run:
        typer.echo("dry-run ok")
        return
    typer.echo(f"run_id={result['run_id']}")
    typer.echo(f"run_dir={result['run_dir']}")
    _echo_results_table(result)
    typer.echo(f"accepted_depth={result['accepted']['depth']} stop={result['accepted']['stop_reason']}")
    typer.echo(f"report={Path(result['run_dir']) / 'report.md'}")


@app.command("report")
def report(
    run_id: str = typer.Option(..., "--run-id"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", path_type=Path),
) -> None:
    run_dir = output_dir / run_id
    report_path = run_dir / "report.md"
    if not report_path.is_file():
        raise typer.BadParameter(f"No report at {report_path}")
    typer.echo(report_path.read_text(encoding="utf-8"))
