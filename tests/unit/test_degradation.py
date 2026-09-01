import numpy as np
import pytest

from astrsr.config import DegradationConfig
from astrsr.data.fixtures import generate_fixture
from astrsr.degradation.forward_model import degrade_reference, forward_project_to_observation
from astrsr.degradation.psf import gaussian_psf
from astrsr.degradation.sampling import block_reduce


def test_gaussian_psf_normalized() -> None:
    kernel = gaussian_psf(1.5)
    assert kernel.ndim == 2
    assert pytest.approx(kernel.sum(), rel=1e-12) == 1.0


def test_block_reduce_shape_and_mean() -> None:
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    reduced = block_reduce(image, 2)
    assert reduced.shape == (2, 2)
    assert reduced[0, 0] == pytest.approx(image[:2, :2].mean())


def test_block_reduce_rejects_indivisible() -> None:
    with pytest.raises(ValueError):
        block_reduce(np.zeros((5, 4)), 2)


def test_degrade_shape_and_seed_replay() -> None:
    reference, _ = generate_fixture("blob", 32, seed=7)
    cfg = DegradationConfig()
    y1, rec1 = degrade_reference(reference, cfg, seed=11)
    y2, rec2 = degrade_reference(reference, cfg, seed=11)
    y3, _ = degrade_reference(reference, cfg, seed=12)
    assert y1.shape == (16, 16)
    assert rec1["scale_factor"] == 2
    np.testing.assert_allclose(y1, y2)
    assert not np.allclose(y1, y3)


def test_forward_project_matches_observation_grid() -> None:
    reference, _ = generate_fixture("ring", 32, seed=3)
    cfg = DegradationConfig()
    observation, _ = degrade_reference(reference, cfg, seed=3)
    projected, sigma = forward_project_to_observation(reference, observation, cfg, total_scale=2)
    assert projected.shape == observation.shape
    assert sigma.shape == observation.shape
    assert np.isfinite(projected).all()
