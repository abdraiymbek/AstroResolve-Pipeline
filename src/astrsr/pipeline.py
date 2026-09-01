"""End-to-end gated recursive super-resolution experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from astrsr.config import AppConfig
from astrsr.data.fixtures import generate_fixture
from astrsr.data.ingest import ingest_reference
from astrsr.degradation.forward_model import degrade_reference, forward_project_to_observation
from astrsr.ensemble.consensus import agreement_stats, build_consensus
from astrsr.ensemble.sampler import sample_ensemble
from astrsr.evaluation.metrics import evaluate_against_reference, flux_rel_error, reduced_chi2
from astrsr.logging.report import conclude, render_report
from astrsr.models.registry import build_baseline, build_primary_model, build_zoo
from astrsr.provenance import initialize_run, mark_completed, write_json
from astrsr.recursion.gates import evaluate_gates
from astrsr.recursion.spatial import apply_spatial_keep
from astrsr.utils.arrays import save_fits, save_npy, save_preview_png


def _save_image(directory: Path, name: str, array: np.ndarray, previews: bool) -> None:
    save_npy(directory / f"{name}.npy", array)
    save_fits(directory / f"{name}.fits", array, header={"prod": name})
    if previews:
        save_preview_png(directory / "previews" / f"{name}.png", array)


def _metrics_row(name: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name}
    if metrics:
        for key in ("psnr", "ssim", "flux_error", "centroid_error"):
            value = metrics.get(key)
            row[key] = None if value is None else round(float(value), 6) if np.isfinite(value) else None
    return row


def load_reference(config: AppConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if config.data.source == "synthetic_fixture":
        return generate_fixture(config.data.fixture, config.data.size, config.run.seed)
    if not config.data.path:
        raise ValueError("data.path is required when source is fits or png")
    return ingest_reference(config.data.path, config.data.source)


def seed_runtime(seed: int, deterministic: bool) -> dict[str, Any]:
    import random

    random.seed(seed)
    np.random.seed(seed)
    note = "numpy and python RNGs seeded"
    try:
        import torch

        torch.manual_seed(seed)
        note += "; torch.manual_seed set"
        if deterministic:
            note += (
                "; full determinism is not claimed on MPS/CUDA. "
                "Recorded seeds are the reproducibility handle that exists."
            )
    except ImportError:
        pass
    return {"seed": seed, "note": note}


def planned_steps(config: AppConfig) -> list[int]:
    factors = list(config.recursion.factors[: config.recursion.max_depth])
    if not config.recursion.enabled:
        return factors[:1]
    return factors


def run_experiment(
    config: AppConfig,
    repo_root: Path,
    overrides: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return {
            "dry_run": True,
            "config": config.model_dump(mode="json"),
            "planned_steps": planned_steps(config),
            "ensemble_samples": config.ensemble.samples,
        }

    run_dir, run_id, device, provenance = initialize_run(config, repo_root, overrides)
    seed_runtime(config.run.seed, config.run.deterministic)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    reference, ref_meta = load_reference(config)
    _save_image(data_dir, "x_reference", reference, config.logging.save_previews)
    write_json(data_dir / "reference_meta.json", ref_meta)

    observation, deg_record = degrade_reference(reference, config.degradation, seed=config.run.seed)
    _save_image(data_dir, "y_observation", observation, config.logging.save_previews)
    write_json(data_dir / "degradation.json", deg_record)

    if config.ensemble.mode == "model_zoo":
        models = build_zoo(config, device=device)
    else:
        models = [build_primary_model(config, device=device)]
    primary = models[0]
    baseline_rows: list[dict[str, Any]] = []
    baseline_dir = run_dir / "baselines"
    for name in config.models.baselines:
        model = build_baseline(name, config)
        output = model.infer(observation)
        target = baseline_dir / name
        _save_image(target, "reconstruction", output, config.logging.save_previews)
        write_json(target / "metadata.json", model.metadata())
        truth = None
        if config.evaluation.compare_to_reference:
            try:
                truth = evaluate_against_reference(
                    reference, output, None, win_size=config.evaluation.ssim_win_size
                )
                write_json(target / "metrics.json", truth)
            except ValueError as exc:
                write_json(target / "metrics.json", {"error": str(exc)})
        baseline_rows.append(_metrics_row(name, truth))

    current = observation
    last_accepted = observation
    accepted_depth = 0
    total_scale = 1
    step_summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    stop_reason = "no_steps"
    steps = planned_steps(config)
    disagreement_accepted = np.zeros_like(observation)

    sole_rows: list[dict[str, Any]] = []
    if config.ensemble.mode == "model_zoo":
        for zoo_model in models:
            output = zoo_model.infer(observation)
            target = run_dir / "solely" / zoo_model.name
            _save_image(target, "reconstruction", output, config.logging.save_previews)
            write_json(target / "metadata.json", zoo_model.metadata())
            truth = None
            if config.evaluation.compare_to_reference:
                try:
                    truth = evaluate_against_reference(
                        reference, output, None, win_size=config.evaluation.ssim_win_size
                    )
                    write_json(target / "metrics.json", truth)
                except ValueError as exc:
                    write_json(target / "metrics.json", {"error": str(exc)})
            sole_rows.append(_metrics_row(zoo_model.name, truth))
            print(f"sole {zoo_model.name} psnr={None if not truth else truth.get('psnr')}", flush=True)

    for step_index, factor in enumerate(steps):
        for zoo_model in models:
            if getattr(zoo_model, "scale", factor) != factor:
                raise ValueError(f"Model {getattr(zoo_model, 'name', zoo_model)} scale != {factor}")
        print(f"recursion step {step_index + 1}/{len(steps)} factor={factor}x", flush=True)
        members, member_records = sample_ensemble(
            current, models, config.ensemble, base_seed=config.run.seed + step_index * 17
        )
        consensus, maps, cons_record = build_consensus(members, config.ensemble)
        total_scale *= factor
        projected, sigma = forward_project_to_observation(
            consensus, observation, config.degradation, total_scale=total_scale
        )
        residual = projected - observation
        chi2 = reduced_chi2(residual, sigma)
        flux_err = flux_rel_error(observation, projected)
        mad_map = maps.get("mad", maps.get("std"))
        if mad_map is None:
            raise RuntimeError("Consensus did not produce a disagreement map")
        stats = agreement_stats(consensus, mad_map)
        gate = evaluate_gates(
            mean_agreement=stats["mean_agreement"],
            mean_relative_uncertainty=stats["mean_relative_uncertainty"],
            reduced_chi2=chi2,
            flux_rel_error=flux_err,
            step_index=step_index,
            n_steps=len(steps),
            config=config.recursion,
        )
        product = consensus
        product_mad = mad_map
        decision = gate.decision
        continue_recursion = gate.continue_recursion
        success_fraction = 1.0 if gate.decision == "accepted" else 0.0
        n_retry_tiles = 0
        spatial_payload: dict[str, Any] | None = None
        success_mask: np.ndarray | None = None
        rel_map: np.ndarray | None = None
        z_map: np.ndarray | None = None
        last_step = step_index >= len(steps) - 1 or (step_index + 1) >= config.recursion.max_depth

        if config.recursion.spatial.enabled and not config.recursion.unconditional:
            def _retry_fn(crop: np.ndarray, tile_index: int) -> tuple[np.ndarray, np.ndarray]:
                tile_members, _ = sample_ensemble(
                    crop,
                    models,
                    config.ensemble,
                    base_seed=config.run.seed + step_index * 17 + 10_000 + tile_index,
                )
                tile_cons, tile_maps, _ = build_consensus(tile_members, config.ensemble)
                tile_mad = tile_maps.get("mad", tile_maps.get("std"))
                if tile_mad is None:
                    raise RuntimeError("Tile consensus did not produce a disagreement map")
                return tile_cons, tile_mad

            def _reproject(mosaic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                proj, sig = forward_project_to_observation(
                    mosaic, observation, config.degradation, total_scale=total_scale
                )
                return proj - observation, sig

            spatial = apply_spatial_keep(
                consensus=consensus,
                mad_map=mad_map,
                residual=residual,
                sigma=sigma,
                fallback=current,
                input_image=current,
                total_scale=total_scale,
                scale_factor=factor,
                config=config.recursion,
                retry_fn=_retry_fn,
                reproject_fn=_reproject,
            )
            product = spatial.mosaic
            product_mad = spatial.mad_map
            projected, sigma = forward_project_to_observation(
                product, observation, config.degradation, total_scale=total_scale
            )
            residual = projected - observation
            chi2 = reduced_chi2(residual, sigma)
            flux_err = flux_rel_error(observation, projected)
            success_fraction = spatial.success_fraction
            n_retry_tiles = spatial.n_retry_tiles
            success_mask = spatial.mask
            rel_map = spatial.relative_uncertainty
            z_map = spatial.residual_sigma
            spatial_payload = {
                "success_fraction": success_fraction,
                "n_retry_tiles": n_retry_tiles,
                "retry_tiles": [[int(a), int(b), int(c), int(d)] for a, b, c, d in spatial.retry_tiles],
                "global_gate": gate.as_dict(),
            }
            if success_fraction <= 0.0:
                decision = gate.decision if gate.decision != "accepted" else "rejected_spatial"
                continue_recursion = False
            else:
                fully_kept = success_fraction >= 1.0 - 1e-12
                decision = "accepted" if fully_kept and gate.decision == "accepted" else "accepted_spatial"
                continue_recursion = (
                    not last_step
                    and success_fraction >= config.recursion.spatial.min_success_fraction_to_continue
                )
                if (
                    config.recursion.gates.require_photometric_check
                    and flux_err > config.recursion.gates.max_flux_rel_error
                ):
                    continue_recursion = False

        truth = None
        if config.evaluation.compare_to_reference:
            try:
                truth = evaluate_against_reference(
                    reference,
                    product,
                    product_mad,
                    win_size=config.evaluation.ssim_win_size,
                )
            except ValueError as exc:
                truth = {"error": str(exc)}

        step_dir = run_dir / "steps" / f"{step_index:02d}"
        if config.ensemble.retain_members:
            for idx, member in enumerate(members):
                save_npy(step_dir / "members" / f"k{idx:03d}.npy", member)
        _save_image(step_dir, "consensus", consensus, config.logging.save_previews)
        _save_image(step_dir, "mosaic", product, config.logging.save_previews)
        _save_image(step_dir, "forward_projection", projected, config.logging.save_previews)
        _save_image(step_dir, "residual", residual, config.logging.save_previews)
        if success_mask is not None and rel_map is not None and z_map is not None:
            save_npy(step_dir / "success_mask.npy", success_mask.astype(np.float64))
            _save_image(step_dir / "maps", "relative_uncertainty", rel_map, config.logging.save_previews)
            _save_image(step_dir / "maps", "residual_sigma", z_map, config.logging.save_previews)
        for map_name, map_data in maps.items():
            _save_image(step_dir / "maps", map_name, map_data, config.logging.save_previews)
            if config.evaluation.save_error_maps and truth and "error" not in truth:
                pass
        write_json(step_dir / "members.json", {"members": member_records, "consensus": cons_record})
        gate_record = gate.as_dict()
        if spatial_payload is not None:
            gate_record["spatial"] = spatial_payload
            gate_record["decision"] = decision
            gate_record["continue_recursion"] = continue_recursion
        write_json(step_dir / "gate.json", gate_record)
        print(
            f"gate={decision} agreement={stats['mean_agreement']:.4f} "
            f"chi2={chi2:.3f} flux_rel={flux_err:.4f} keep={success_fraction:.3f} "
            f"retry_tiles={n_retry_tiles} continue={continue_recursion}",
            flush=True,
        )
        if truth:
            write_json(step_dir / "metrics.json", truth)
        write_json(
            step_dir / "model.json",
            {
                "mode": config.ensemble.mode,
                "members": [m.metadata() if hasattr(m, "metadata") else {"name": m.name} for m in models],
            },
        )

        summary = {
            "index": step_index,
            "total_scale": total_scale,
            "n_members": cons_record["n_members"],
            "n_kept": cons_record["n_kept"],
            "mean_agreement": stats["mean_agreement"],
            "mean_relative_uncertainty": stats["mean_relative_uncertainty"],
            "reduced_chi2": chi2,
            "flux_rel_error": flux_err,
            "decision": decision,
            "continue_recursion": continue_recursion,
            "success_fraction": success_fraction,
            "n_retry_tiles": n_retry_tiles,
            "reasons": gate.reasons if decision.startswith("rejected") else [decision],
            "would_reject": gate.would_reject,
            "truth_metrics": truth,
        }
        step_summaries.append(summary)
        if decision.startswith("rejected"):
            failures.append(
                f"step {step_index}: {decision} agreement={stats['mean_agreement']:.4f} "
                f"chi2={chi2:.3f} flux_rel={flux_err:.4f} keep={success_fraction:.3f}"
            )
            stop_reason = decision
            break
        last_accepted = product
        disagreement_accepted = product_mad
        accepted_depth = step_index + 1
        current = product
        if last_step or not continue_recursion:
            stop_reason = "max_depth_reached" if last_step else decision
            break
        stop_reason = "accepted_continue"

    accepted_dir = run_dir / "accepted"
    _save_image(accepted_dir, "reconstruction", last_accepted, config.logging.save_previews)
    _save_image(accepted_dir, "disagreement", disagreement_accepted, config.logging.save_previews)
    accepted_truth = None
    if config.evaluation.compare_to_reference:
        try:
            accepted_truth = evaluate_against_reference(
                reference,
                last_accepted,
                disagreement_accepted if disagreement_accepted.shape == last_accepted.shape else None,
                win_size=config.evaluation.ssim_win_size,
            )
        except ValueError as exc:
            accepted_truth = {"error": str(exc)}

    report_payload: dict[str, Any] = {
        "title": f"astrsr run {run_id}",
        "run_id": run_id,
        "config_hash": provenance["config_hash"],
        "seed": config.run.seed,
        "device": device,
        "data": {
            "source": config.data.source,
            "fixture": config.data.fixture,
            "reference_shape": list(reference.shape),
            "observation_shape": list(observation.shape),
            "split": config.data.split,
        },
        "model": {
            "name": config.models.pretrained.name,
            "checkpoint": config.models.pretrained.checkpoint,
            "license": config.models.pretrained.license,
            "ensemble_mode": config.ensemble.mode,
            "ensemble_zoo": [getattr(m, "name", type(m).__name__) for m in models],
            "ensemble_samples": len(member_records) if step_summaries else config.ensemble.samples,
            "stochasticity": config.ensemble.stochasticity.source,
        },
        "baselines": baseline_rows,
        "solely": sole_rows,
        "steps": step_summaries,
        "accepted": {
            "depth": accepted_depth,
            "stop_reason": stop_reason,
            "shape": list(last_accepted.shape),
            "truth_metrics": accepted_truth,
        },
        "failures": failures,
        "conclusion": "",
        "limitations": provenance["limitations"],
        "run_dir": str(run_dir),
    }
    report_payload["conclusion"] = conclude(report_payload)
    markdown = render_report(report_payload)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    write_json(run_dir / "report.json", report_payload)
    if config.logging.export_elabftw:
        (run_dir / "elabftw.md").write_text(markdown, encoding="utf-8")
    mark_completed(run_dir)
    return report_payload
