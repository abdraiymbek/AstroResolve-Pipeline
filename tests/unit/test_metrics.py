import numpy as np
import pytest

from astrsr.evaluation.metrics import (
    centroid,
    centroid_error,
    error_vs_truth_map,
    error_vs_truth_rate,
    evaluate_against_reference,
    flux_rel_error,
    psnr,
    total_flux,
)


def test_flux_and_centroid_shift() -> None:
    image = np.zeros((21, 21), dtype=np.float64)
    image[10, 10] = 5.0
    shifted = np.zeros_like(image)
    shifted[12, 14] = 5.0
    assert total_flux(image) == 5.0
    assert flux_rel_error(image, shifted) == 0.0
    cy, cx = centroid(image)
    assert cy == 10 and cx == 10
    error = centroid_error(image, shifted)
    assert error == np.hypot(2, 4)


def test_psnr_identity_is_high() -> None:
    image = np.linspace(0, 1, 64).reshape(8, 8)
    assert psnr(image, image) > 80


def test_centroid_error_is_scale_invariant_in_obs_pixels() -> None:
    reference = np.zeros((16, 16), dtype=np.float64)
    reference[8, 8] = 5.0
    fine = np.zeros((16, 16), dtype=np.float64)
    fine[8, 10] = 5.0
    coarse = np.zeros((8, 8), dtype=np.float64)
    coarse[4, 5] = 5.0
    observation = np.zeros((4, 4), dtype=np.float64)
    observation[2, 2] = 5.0
    fine_m = evaluate_against_reference(reference, fine, None, win_size=3, observation=observation)
    coarse_m = evaluate_against_reference(reference, coarse, None, win_size=3, observation=observation)
    assert fine_m["centroid_error_grid_px"] == 2.0
    assert coarse_m["centroid_error_grid_px"] == 1.0
    assert fine_m["centroid_error"] == pytest.approx(0.5)
    assert coarse_m["centroid_error"] == pytest.approx(fine_m["centroid_error"])
    assert fine_m["centroid_error_ref_px"] == pytest.approx(2.0)


def test_evaluate_coarser_estimate_downsamples_reference() -> None:
    reference = np.arange(16, dtype=np.float64).reshape(4, 4)
    coarse = reference.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    metrics = evaluate_against_reference(reference, coarse, None, win_size=3)
    assert metrics["compared_at_shape"] == [2, 2]
    assert metrics["psnr"] == float("inf")
    assert metrics["error_vs_truth_rate"] == 0.0


def test_error_vs_truth_identity_is_zero() -> None:
    assert error_vs_truth_rate(float("inf"), 1.0, 0.0) == 0.0


def test_error_vs_truth_total_loss_is_hundred() -> None:
    assert error_vs_truth_rate(0.0, 0.0, 1.0) == 100.0


def test_error_vs_truth_map_identity_is_zero() -> None:
    image = np.linspace(0, 1, 64).reshape(8, 8)
    err_map = error_vs_truth_map(image, image, win_size=3)
    assert err_map.shape == image.shape
    assert float(err_map.max()) < 1e-6
