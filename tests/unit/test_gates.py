from astrsr.config import RecursionConfig
from astrsr.recursion.gates import evaluate_gates


def _gate(**kwargs):
    defaults = dict(
        mean_agreement=0.95,
        mean_relative_uncertainty=0.05,
        reduced_chi2=1.1,
        flux_rel_error=0.01,
        step_index=0,
        n_steps=2,
        config=RecursionConfig(),
    )
    defaults.update(kwargs)
    return evaluate_gates(**defaults)


def test_accept_and_continue() -> None:
    decision = _gate()
    assert decision.decision == "accepted"
    assert decision.continue_recursion is True


def test_reject_uncertainty() -> None:
    decision = _gate(mean_agreement=0.1, mean_relative_uncertainty=0.9)
    assert decision.decision == "rejected_uncertainty"
    assert decision.continue_recursion is False


def test_reject_forward_consistency() -> None:
    decision = _gate(reduced_chi2=9.0)
    assert decision.decision == "rejected_forward_consistency"


def test_reject_photometry() -> None:
    decision = _gate(flux_rel_error=0.4)
    assert decision.decision == "rejected_photometry"


def test_max_depth_stops_continuation() -> None:
    decision = _gate(step_index=1, n_steps=2)
    assert decision.decision == "accepted"
    assert decision.continue_recursion is False
    assert "max_depth_reached" in decision.reasons


def test_unconditional_ignores_failed_gates() -> None:
    cfg = RecursionConfig(unconditional=True)
    decision = _gate(mean_agreement=0.0, reduced_chi2=50.0, config=cfg)
    assert decision.decision == "accepted"
    assert "rejected_uncertainty" in decision.would_reject
    assert "rejected_forward_consistency" in decision.would_reject
