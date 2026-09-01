"""Recursion gates. Agreement and original-observation consistency can stop a zoom."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from astrsr.config import GateConfig, RecursionConfig


@dataclass
class GateDecision:
    decision: str
    continue_recursion: bool
    mean_agreement: float
    mean_relative_uncertainty: float
    reduced_chi2: float
    flux_rel_error: float
    reasons: list[str] = field(default_factory=list)
    would_reject: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_gates(
    *,
    mean_agreement: float,
    mean_relative_uncertainty: float,
    reduced_chi2: float,
    flux_rel_error: float,
    step_index: int,
    n_steps: int,
    config: RecursionConfig,
) -> GateDecision:
    gates: GateConfig = config.gates
    would_reject: list[str] = []
    if mean_agreement < gates.min_agreement or mean_relative_uncertainty > gates.max_relative_uncertainty:
        would_reject.append("rejected_uncertainty")
    if gates.require_forward_consistency and reduced_chi2 > gates.max_reduced_chi2:
        would_reject.append("rejected_forward_consistency")
    if gates.require_photometric_check and flux_rel_error > gates.max_flux_rel_error:
        would_reject.append("rejected_photometry")

    last_step = step_index >= n_steps - 1 or (step_index + 1) >= config.max_depth
    if config.unconditional:
        decision = "accepted"
        continue_recursion = not last_step
        reasons = ["unconditional_recursion"]
        if last_step:
            reasons.append("max_depth_reached")
        return GateDecision(
            decision=decision,
            continue_recursion=continue_recursion,
            mean_agreement=mean_agreement,
            mean_relative_uncertainty=mean_relative_uncertainty,
            reduced_chi2=reduced_chi2,
            flux_rel_error=flux_rel_error,
            reasons=reasons,
            would_reject=would_reject,
        )

    if would_reject:
        decision = would_reject[0]
        return GateDecision(
            decision=decision,
            continue_recursion=False,
            mean_agreement=mean_agreement,
            mean_relative_uncertainty=mean_relative_uncertainty,
            reduced_chi2=reduced_chi2,
            flux_rel_error=flux_rel_error,
            reasons=would_reject,
            would_reject=would_reject,
        )

    reasons = ["accepted"]
    continue_recursion = not last_step
    if last_step:
        reasons.append("max_depth_reached")
    return GateDecision(
        decision="accepted",
        continue_recursion=continue_recursion,
        mean_agreement=mean_agreement,
        mean_relative_uncertainty=mean_relative_uncertainty,
        reduced_chi2=reduced_chi2,
        flux_rel_error=flux_rel_error,
        reasons=reasons,
        would_reject=[],
    )
