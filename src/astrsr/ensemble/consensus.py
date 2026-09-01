"""Robust consensus, outlier filtering, and spatial disagreement maps."""

from __future__ import annotations

from typing import Any

import numpy as np

from astrsr.config import EnsembleConfig
from astrsr.utils.arrays import as_float_image


def _trimmed_mean(stack: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return stack.mean(axis=0)
    k = stack.shape[0]
    n_trim = int(np.floor(fraction * k))
    if 2 * n_trim >= k:
        return np.median(stack, axis=0)
    ordered = np.sort(stack, axis=0)
    return ordered[n_trim : k - n_trim].mean(axis=0)


def build_consensus(
    members: list[np.ndarray],
    config: EnsembleConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if not members:
        raise ValueError("Cannot build consensus from an empty ensemble")
    stack = np.stack([as_float_image(m) for m in members], axis=0)
    median_image = np.median(stack, axis=0)
    member_l1 = np.mean(np.abs(stack - median_image), axis=(1, 2))
    center = np.median(member_l1)
    mad_scores = np.median(np.abs(member_l1 - center))
    robust_z = 0.6745 * (member_l1 - center) / (mad_scores + 1e-12)
    keep = np.ones(len(members), dtype=bool)
    filter_reason = "none"
    if config.filtering.method == "mad":
        keep = np.abs(robust_z) <= config.filtering.threshold
        if not np.any(keep):
            keep[:] = True
            filter_reason = "all_members_flagged_kept_all"
        else:
            filter_reason = "mad_zscore"
    kept = stack[keep]
    if config.consensus.method == "median":
        consensus = np.median(kept, axis=0)
    elif config.consensus.method == "trimmed_mean":
        consensus = _trimmed_mean(kept, config.consensus.trim_fraction)
    else:
        raise ValueError(f"Unknown consensus method {config.consensus.method}")
    maps: dict[str, np.ndarray] = {}
    if "std" in config.uncertainty.maps:
        maps["std"] = kept.std(axis=0, ddof=0)
    if "mad" in config.uncertainty.maps:
        maps["mad"] = np.median(np.abs(kept - consensus), axis=0) * 1.4826
    if "percentile_interval" in config.uncertainty.maps:
        hi = np.percentile(kept, config.uncertainty.percentile_high, axis=0)
        lo = np.percentile(kept, config.uncertainty.percentile_low, axis=0)
        maps["percentile_interval"] = hi - lo
    record = {
        "n_members": int(len(members)),
        "n_kept": int(keep.sum()),
        "rejected_indices": [int(i) for i, flag in enumerate(keep) if not flag],
        "filter_reason": filter_reason,
        "member_l1": member_l1.tolist(),
        "robust_z": robust_z.tolist(),
        "consensus_method": config.consensus.method,
    }
    return as_float_image(consensus), maps, record


def agreement_stats(consensus: np.ndarray, mad_map: np.ndarray) -> dict[str, float]:
    scale = float(np.median(np.abs(consensus))) + 1e-8
    relative = mad_map / (np.abs(consensus) + scale * 0.05 + 1e-8)
    agreement_map = 1.0 / (1.0 + relative)
    return {
        "mean_agreement": float(agreement_map.mean()),
        "mean_relative_uncertainty": float(relative.mean()),
        "median_relative_uncertainty": float(np.median(relative)),
    }
