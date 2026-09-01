"""Human-readable run report with limitations and gate history."""

from __future__ import annotations

from typing import Any

LIMITATIONS = (
    "The method reconstructs or infers a plausible representation under specified "
    "assumptions; it does not bypass physical information limits. Cross-run agreement "
    "is stability under the chosen perturbations, not independent evidence that a "
    "feature is real. Shared-model bias can make independent runs agree on the same "
    "wrong structure."
)


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
    lines.append("## Baselines")
    lines.append("")
    if payload["baselines"]:
        lines.append("| method | psnr | ssim | flux_error | centroid_error |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in payload["baselines"]:
            lines.append(
                f"| {row['name']} | {row.get('psnr', '')} | {row.get('ssim', '')} | "
                f"{row.get('flux_error', '')} | {row.get('centroid_error', '')} |"
            )
    else:
        lines.append("No baselines were run.")
    lines.append("")
    if payload.get("solely"):
        lines.append("## Each method solely (one-shot 2x, no ensemble)")
        lines.append("")
        lines.append("| method | psnr | ssim | flux_error | centroid_error |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in payload["solely"]:
            lines.append(
                f"| {row['name']} | {row.get('psnr', '')} | {row.get('ssim', '')} | "
                f"{row.get('flux_error', '')} | {row.get('centroid_error', '')} |"
            )
        lines.append("")
        lines.append(
            "The gated consensus row is in Recursion below. "
            "If the gate rejects, the algorithm output is the last accepted lower-resolution image, not the 2x candidate."
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
    rejected = [step for step in steps if step["decision"] != "accepted"]
    if rejected:
        step = rejected[0]
        return (
            f"Recursion stopped at step {step['index']} with `{step['decision']}`. "
            f"The last accepted product is depth {accepted['depth']}. "
            "This is an abstention, not a claim that extra zoom is real structure."
        )
    return (
        f"All {len(steps)} configured steps were accepted. "
        "Acceptance means the predeclared gates passed, not that the reconstruction "
        "is astronomically true. Compare metrics against the held-out reference before any interpretation."
    )
