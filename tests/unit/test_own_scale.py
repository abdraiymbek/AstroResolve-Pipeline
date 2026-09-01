import numpy as np

from astrsr.models.baselines import InterpolationModel, interpolate
from astrsr.models.fake import FakeSRModel
from astrsr.pipeline import own_method_to_planned_scale, planned_total_scale


def test_planned_total_scale() -> None:
    assert planned_total_scale([2, 2, 2]) == 8
    assert planned_total_scale([2]) == 2


def test_interpolation_zooms_once_with_its_own_kernel() -> None:
    model = InterpolationModel("bicubic", scale=2)
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = own_method_to_planned_scale(model, image, [2, 2, 2])
    assert out.shape == (32, 32)
    np.testing.assert_allclose(out, interpolate(image, 8, "bicubic"))


def test_fake_sr_chains_its_own_infer() -> None:
    model = FakeSRModel()
    image = np.ones((4, 4), dtype=np.float64)
    out = own_method_to_planned_scale(model, image, [2, 2])
    assert out.shape == (16, 16)
    chained = model.infer(model.infer(image))
    np.testing.assert_allclose(out, chained)
