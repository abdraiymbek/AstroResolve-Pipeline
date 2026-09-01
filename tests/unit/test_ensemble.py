import numpy as np

from astrsr.config import EnsembleConfig
from astrsr.ensemble.consensus import agreement_stats, build_consensus
from astrsr.models.baselines import interpolate
from astrsr.models.fake import FakeSRModel


def test_median_consensus_recovers_common_image() -> None:
    base = np.ones((8, 8), dtype=np.float64)
    members = [base, base + 0.1, base - 0.1, base + 10.0]
    cfg = EnsembleConfig()
    consensus, maps, record = build_consensus(members, cfg)
    assert record["n_kept"] == 3
    assert 3 in record["rejected_indices"]
    np.testing.assert_allclose(consensus, base, atol=0.1)
    assert "mad" in maps


def test_agreement_high_when_members_match() -> None:
    members = [np.ones((6, 6)) for _ in range(5)]
    consensus, maps, _ = build_consensus(members, EnsembleConfig())
    stats = agreement_stats(consensus, maps["mad"])
    assert stats["mean_agreement"] > 0.99
    assert stats["mean_relative_uncertainty"] < 0.01


def test_fake_sr_is_2x() -> None:
    image = np.ones((8, 8), dtype=np.float64)
    out = FakeSRModel().infer(image)
    assert out.shape == (16, 16)


def test_interpolation_shapes() -> None:
    image = np.random.default_rng(0).normal(size=(8, 8))
    for method in ("nearest", "bicubic", "lanczos"):
        out = interpolate(image, 2, method)
        assert out.shape == (16, 16)
