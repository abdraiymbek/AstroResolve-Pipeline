"""Run identity, environment capture, and immutable run directories."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrsr import __version__
from astrsr.config import AppConfig, config_hash


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def environment_record(device: str) -> dict[str, Any]:
    torch_version = None
    cuda = False
    mps = False
    try:
        import torch

        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(mps_backend is not None and mps_backend.is_available())
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "astrsr_version": __version__,
        "torch": torch_version,
        "cuda_available": cuda,
        "mps_available": mps,
        "selected_device": device,
        "cwd": os.getcwd(),
    }


def make_run_id(config: AppConfig, cfg_hash: str) -> str:
    stamp = utc_now()
    return f"{stamp}_{config.run.name}_{cfg_hash[:8]}"


def create_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_run(
    config: AppConfig,
    repo_root: Path,
    overrides: list[str],
) -> tuple[Path, str, str, dict[str, Any]]:
    cfg_hash = config_hash(config)
    device = select_device(config.run.device)
    run_id = make_run_id(config, cfg_hash)
    run_dir = create_run_dir(Path(config.run.output_dir), run_id)
    provenance = {
        "run_id": run_id,
        "created_utc": utc_now(),
        "config_hash": cfg_hash,
        "seed": config.run.seed,
        "overrides": overrides,
        "git_revision": git_revision(repo_root),
        "device_requested": config.run.device,
        "device_selected": device,
        "limitations": (
            "The method reconstructs or infers a plausible representation under specified "
            "assumptions; it does not bypass physical information limits. Cross-run "
            "agreement is stability under the chosen perturbations, not proof a feature is real."
        ),
    }
    if config.logging.save_config:
        write_json(run_dir / "config.json", config.model_dump(mode="json"))
    if config.logging.save_environment:
        write_json(run_dir / "environment.json", environment_record(device))
    write_json(run_dir / "provenance.json", provenance)
    (run_dir / "config_hash.txt").write_text(cfg_hash + "\n", encoding="utf-8")
    return run_dir, run_id, device, provenance


def mark_completed(run_dir: Path) -> None:
    (run_dir / "COMPLETED").write_text(utc_now() + "\n", encoding="utf-8")
