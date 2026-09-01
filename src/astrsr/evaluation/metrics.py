"""Image-level and astronomy-relevant fidelity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.metrics import structural_similarity

from astrsr.degradation.sampling import block_reduce
from astrsr.utils.arrays import as_float_image, require_same_shape


def _data_range(reference: np.ndarray) -> float:
    span = float(reference.max() - reference.min())
    return span if span > 0 else 1.0


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    require_same_shape(reference, estimate)
    ref = as_float_image(reference)
    est = as_float_image(estimate)
    mse = float(np.mean((ref - est) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(_data_range(ref) ** 2 / mse))


def ssim(reference: np.ndarray, estimate: np.ndarray, win_size: int = 7) -> float:
    require_same_shape(reference, estimate)
    ref = as_float_image(reference)
    est = as_float_image(estimate)
    odd = win_size if win_size % 2 == 1 else win_size - 1
    max_win = int(min(ref.shape))
    if max_win < 3:
        return float("nan")
    if max_win % 2 == 0:
        max_win -= 1
    odd = max(3, min(odd, max_win))
    return float(
        structural_similarity(ref, est, data_range=_data_range(ref), win_size=odd)
    )


def total_flux(image: np.ndarray) -> float:
    return float(as_float_image(image).sum())


def flux_rel_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref_flux = total_flux(reference)
    est_flux = total_flux(estimate)
    denom = abs(ref_flux) if abs(ref_flux) > 1e-12 else 1.0
    return abs(est_flux - ref_flux) / denom


def centroid(image: np.ndarray) -> tuple[float, float]:
    data = np.clip(as_float_image(image), 0.0, None)
    total = data.sum()
    if total <= 0:
        return float("nan"), float("nan")
    yy, xx = np.indices(data.shape, dtype=np.float64)
    return float((data * yy).sum() / total), float((data * xx).sum() / total)


def centroid_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    ry, rx = centroid(reference)
    ey, ex = centroid(estimate)
    if not np.isfinite([ry, rx, ey, ex]).all():
        return float("nan")
    return float(np.hypot(ry - ey, rx - ex))


def reduced_chi2(residual: np.ndarray, sigma: np.ndarray) -> float:
    var = np.clip(as_float_image(sigma) ** 2, 1e-12, None)
    chi = (as_float_image(residual) ** 2) / var
    return float(np.mean(chi))


def disagreement_error_correlation(disagreement: np.ndarray, error: np.ndarray) -> float:
    d = as_float_image(disagreement).ravel()
    e = as_float_image(error).ravel()
    if d.std() < 1e-12 or e.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(d, e)[0, 1])


def match_to_reference(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compare at the reference grid. Downsample the estimate if it is larger by an integer factor."""
    est = as_float_image(estimate)
    ref = as_float_image(reference)
    if est.shape == ref.shape:
        return est
    if est.shape[0] % ref.shape[0] or est.shape[1] % ref.shape[1]:
        raise ValueError(f"Cannot match estimate {est.shape} to reference {ref.shape}")
    fy = est.shape[0] // ref.shape[0]
    fx = est.shape[1] // ref.shape[1]
    if fy != fx:
        raise ValueError(f"Anisotropic scale {fy} vs {fx}")
    return block_reduce(est, fy)


def align_for_comparison(estimate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put estimate and reference on one grid. Downsample whichever array is finer."""
    est = as_float_image(estimate)
    ref = as_float_image(reference)
    if est.shape == ref.shape:
        return est, ref
    if est.shape[0] >= ref.shape[0] and est.shape[1] >= ref.shape[1]:
        return match_to_reference(est, ref), ref
    if ref.shape[0] >= est.shape[0] and ref.shape[1] >= est.shape[1]:
        return est, match_to_reference(ref, est)
    raise ValueError(f"Cannot align estimate {est.shape} to reference {ref.shape}")


def evaluate_against_reference(
    reference: np.ndarray,
    estimate: np.ndarray,
    disagreement: np.ndarray | None,
    win_size: int,
) -> dict[str, Any]:
    matched, ref = align_for_comparison(estimate, reference)
    error = np.abs(matched - ref)
    metrics: dict[str, Any] = {
        "psnr": psnr(ref, matched),
        "ssim": ssim(ref, matched, win_size=win_size),
        "flux_error": flux_rel_error(ref, matched),
        "centroid_error": centroid_error(ref, matched),
        "mean_abs_error": float(error.mean()),
        "compared_at_shape": list(ref.shape),
        "estimate_shape": list(estimate.shape),
        "reference_shape": list(reference.shape),
        "downsampled_for_truth": list(estimate.shape) != list(reference.shape),
    }
    if disagreement is not None:
        dmatch, _ = align_for_comparison(disagreement, ref)
        metrics["disagreement_error_correlation"] = disagreement_error_correlation(dmatch, error)
    return metrics
