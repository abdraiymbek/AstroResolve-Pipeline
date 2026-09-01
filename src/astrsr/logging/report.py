"""Human-readable run report with limitations and gate history."""

from __future__ import annotations

import math
from typing import Any

LIMITATIONS = (
    "The method reconstructs or infers a plausible representation under specified "
    "assumptions; it does not bypass physical information limits. Cross-run agreement "
    "is stability under the chosen perturbations, not independent evidence that a "
    "feature is real. Shared-model bias can make independent runs agree on the same "
    "wrong structure."
)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.4f}"


def results_table_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One-shot methods, gated mosaic, and the image the algorithm actually kept."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in payload.get("solely") or []:
        rows.append({**row, "note": "one-shot 2x"})
        seen.add(str(row.get("name")))
    for row in payload.get("baselines") or []:
        name = str(row.get("name"))
        if name in seen:
            continue
        rows.append({**row, "note": "baseline"})
        seen.add(name)
    for step in payload.get("steps") or []:
        tm = step.get("truth_metrics") or {}
        if "error" in tm and "psnr" not in tm:
            tm = {}
        frac = step.get("success_fraction")
        keep = "" if frac is None else f"keep {100.0 * float(frac):.0f}%"
        rows.append(
            {
                "name": f"gated mosaic {step['total_scale']}x",
                "psnr": tm.get("psnr"),
                "ssim": tm.get("ssim"),
                "flux_error": tm.get("flux_error"),
                "centroid_error": tm.get("centroid_error"),
                "note": " ".join(
                    part for part in (str(step.get("decision", "")), keep) if part
                ),
            }
        )
    acc = payload.get("accepted") or {}
    tm = acc.get("truth_metrics") or {}
    if "error" in tm and "psnr" not in tm:
        tm = {}
    rows.append(
        {
            "name": "accepted product",
            "psnr": tm.get("psnr"),
            "ssim": tm.get("ssim"),
            "flux_error": tm.get("flux_error"),
            "centroid_error": tm.get("centroid_error"),
            "note": f"depth {acc.get('depth')} {acc.get('stop_reason', '')}".strip(),
        }
    )
    return rows


def render_results_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| method | psnr | ssim | flux_error | centroid_error | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('name', '')} | {_fmt_metric(row.get('psnr'))} | "
            f"{_fmt_metric(row.get('ssim'))} | {_fmt_metric(row.get('flux_error'))} | "
            f"{_fmt_metric(row.get('centroid_error'))} | {row.get('note', '')} |"
        )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {payload['title']}")
    lines.append("")
    lines.append(f"- Run ID: `{payload['run_id']}`")
    lines.append(f"- Config hash: `{payload['config_hash']}`")
    lines.append(f"- Seed: `{payload['seed']}`")
    lines.append(f"- Device: `{payload['device']}`")
    lines.append("")
    lines.append("## Research question")
    lines.append("")
    lines.append(
        "Can consensus and data-consistency checks make recursive astronomical "
        "super-resolution more trustworthy and reduce AI hallucinations?"
    )
    lines.append("")
    lines.append("## Hypotheses under test")
    lines.append("")
    lines.append("1. Consensus: when many reconstructions agree, is that detail actually correct?")
    lines.append("2. Gating: do agreement plus shrink-back match to the original observation stop hallucinations?")
    lines.append("3. Recursive SR: can the gated loop recover more useful structure than one-shot SR or interpolation without going past the reliable point?")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(LIMITATIONS)
    lines.append("")
    lines.append("## Data")
    lines.append("")
    for key, value in payload["data"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    for key, value in payload["model"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Results vs held-out reference")
    lines.append("")
    lines.append(render_results_table(results_table_rows(payload)))
    lines.append("")
    lines.append(
        "One-shot rows are each method alone. "
        "`gated mosaic` is the spatial keep after this 2x. "
        "`accepted product` is what the algorithm returns, including a freeze at the observation if nothing passed."
    )
    lines.append("")
    lines.append("## Recursion")
    lines.append("")
    for step in payload["steps"]:
        lines.append(f"### Step {step['index']} (total scale {step['total_scale']}x)")
        lines.append("")
        lines.append(f"- Ensemble samples: `{step['n_members']}` kept `{step['n_kept']}`")
        lines.append(f"- Mean agreement: `{step['mean_agreement']}`")
        lines.append(f"- Mean relative uncertainty: `{step['mean_relative_uncertainty']}`")
        lines.append(f"- Reduced chi-square vs original y: `{step['reduced_chi2']}`")
        lines.append(f"- Flux relative error vs original y: `{step['flux_rel_error']}`")
        lines.append(f"- Gate: `{step['decision']}`")
        lines.append(f"- Continue: `{step['continue_recursion']}`")
        if step.get("success_fraction") is not None:
            lines.append(f"- Spatial keep fraction: `{step['success_fraction']}`")
        if step.get("n_retry_tiles") is not None:
            lines.append(f"- Retry tiles: `{step['n_retry_tiles']}`")
        if step.get("truth_metrics"):
            tm = step["truth_metrics"]
            lines.append(
                f"- vs reference: PSNR `{tm.get('psnr')}`, SSIM `{tm.get('ssim')}`, "
                f"flux error `{tm.get('flux_error')}`, centroid error `{tm.get('centroid_error')}`, "
                f"disagreement-error correlation `{tm.get('disagreement_error_correlation')}`"
            )
        lines.append("")
    lines.append("## Accepted result")
    lines.append("")
    acc = payload["accepted"]
    lines.append(f"- Depth accepted: `{acc['depth']}`")
    lines.append(f"- Stop reason: `{acc['stop_reason']}`")
    lines.append(f"- Reconstruction shape: `{acc['shape']}`")
    if acc.get("truth_metrics"):
        tm = acc["truth_metrics"]
        lines.append(
            f"- vs reference: PSNR `{tm.get('psnr')}`, SSIM `{tm.get('ssim')}`, "
            f"flux error `{tm.get('flux_error')}`"
        )
    lines.append("")
    lines.append("## Conclusion limited to this run")
    lines.append("")
    lines.append(payload["conclusion"])
    lines.append("")
    if payload.get("failures"):
        lines.append("## Failure cases")
        lines.append("")
        for item in payload["failures"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines) + "\n"


def conclude(payload: dict[str, Any]) -> str:
    accepted = payload["accepted"]
    steps = payload["steps"]
    if not steps:
        return "No ensemble step ran. No scientific conclusion is available."
    rejected = [step for step in steps if str(step["decision"]).startswith("rejected")]
    spatial = [step for step in steps if step["decision"] == "accepted_spatial"]
    if rejected and not spatial:
        step = rejected[0]
        return (
            f"Recursion stopped at step {step['index']} with `{step['decision']}`. "
            f"The last accepted product is depth {accepted['depth']}. "
            "This is an abstention, not a claim that extra zoom is real structure."
        )
    if spatial:
        step = spatial[-1]
        frac = step.get("success_fraction")
        kept = "unknown fraction" if frac is None else f"{100.0 * float(frac):.0f}%"
        return (
            f"Kept {kept} of the 2x field at step {step['index']}. "
            "Failed regions stayed at the previous scale. "
            f"Product depth is {accepted['depth']}. "
            "Kept pixels passed per-pixel uncertainty and shrink-back, not a claim of true structure."
        )
    return (
        f"All {len(steps)} configured steps were accepted. "
        "Acceptance means the predeclared gates passed, not that the reconstruction "
        "is astronomically true. Compare metrics against the held-out reference before any interpretation."
    )
