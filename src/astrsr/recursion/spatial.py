"""Spatial keep-and-retry: freeze passing pixels, re-run only failed tiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import zoom

from astrsr.config import RecursionConfig
from astrsr.utils.arrays import as_float_image

RetryFn = Callable[[np.ndarray, int, int], tuple[np.ndarray, np.ndarray]]
ReprojectFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def relative_uncertainty_map(consensus: np.ndarray, mad_map: np.ndarray) -> np.ndarray:
    cons = as_float_image(consensus)
    mad = as_float_image(mad_map)
    scale = float(np.median(np.abs(cons))) + 1e-8
    return mad / (np.abs(cons) + scale * 0.05 + 1e-8)


def upsample_block(image: np.ndarray, factor: int) -> np.ndarray:
    data = as_float_image(image)
    if factor <= 1:
        return data
    return np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)


def upsample_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    data = as_float_image(image)
    if data.shape == shape:
        return data
    fy = shape[0] / data.shape[0]
    fx = shape[1] / data.shape[1]
    if (
        abs(fy - fx) < 1e-9
        and abs(fy - round(fy)) < 1e-9
        and int(round(fy)) >= 1
    ):
        return upsample_block(data, int(round(fy)))
    return as_float_image(zoom(data, (fy, fx), order=3, prefilter=True))


def spatial_success_mask(
    consensus: np.ndarray,
    mad_map: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    total_scale: int,
    config: RecursionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True where models agree and shrink-back residual is inside the noise envelope."""
    rel = relative_uncertainty_map(consensus, mad_map)
    z_obs = np.abs(as_float_image(residual)) / np.clip(as_float_image(sigma), 1e-12, None)
    z_hr = upsample_block(z_obs, total_scale)
    if z_hr.shape != rel.shape:
        z_hr = upsample_to_shape(z_obs, rel.shape)
    pass_unc = rel <= config.gates.max_relative_uncertainty
    if config.gates.require_forward_consistency:
        pass_res = z_hr <= config.spatial.max_residual_sigma
    else:
        pass_res = np.ones_like(rel, dtype=bool)
    if config.unconditional:
        mask = np.ones_like(rel, dtype=bool)
    else:
        mask = pass_unc & pass_res
    return mask.astype(bool), rel, z_hr


def mosaic_from_mask(
    consensus: np.ndarray,
    fallback: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    cons = as_float_image(consensus)
    low = upsample_to_shape(as_float_image(fallback), cons.shape)
    return as_float_image(np.where(mask, cons, low))


def _expand(r0: int, r1: int, c0: int, c1: int, overlap: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    height, width = shape
    return (
        slice(max(0, r0 - overlap), min(height, r1 + overlap)),
        slice(max(0, c0 - overlap), min(width, c1 + overlap)),
    )


def failed_tiles(
    mask: np.ndarray,
    min_tile: int,
    overlap: int,
) -> list[tuple[slice, slice]]:
    fail = ~np.asarray(mask, dtype=bool)
    if not np.any(fail):
        return []
    rows, cols = np.where(fail)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    height, width = r1 - r0, c1 - c0
    if height >= 2 * min_tile and width >= 2 * min_tile:
        mid_r = r0 + height // 2
        mid_c = c0 + width // 2
        quads = [
            (r0, mid_r, c0, mid_c),
            (r0, mid_r, mid_c, c1),
            (mid_r, r1, c0, mid_c),
            (mid_r, r1, mid_c, c1),
        ]
    else:
        quads = [(r0, r1, c0, c1)]
    tiles: list[tuple[slice, slice]] = []
    for a, b, c, d in quads:
        if b <= a or d <= c:
            continue
        if mask[a:b, c:d].all():
            continue
        tiles.append(_expand(a, b, c, d, overlap, mask.shape))
    return tiles


def cosine_weight(
    height: int,
    width: int,
    overlap: int,
    *,
    row_start: int = 0,
    col_start: int = 0,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    wy = np.ones(height, dtype=np.float64)
    wx = np.ones(width, dtype=np.float64)
    if overlap > 0:
        n = min(overlap, max(1, height // 2), max(1, width // 2))
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, n)) ** 2
        taper_top = image_shape is None or row_start > 0
        taper_bottom = image_shape is None or row_start + height < image_shape[0]
        taper_left = image_shape is None or col_start > 0
        taper_right = image_shape is None or col_start + width < image_shape[1]
        if taper_top:
            wy[:n] = ramp
        if taper_bottom:
            wy[-n:] = ramp[::-1]
        if taper_left:
            wx[:n] = ramp
        if taper_right:
            wx[-n:] = ramp[::-1]
    return np.outer(wy, wx)


def align_to_factor(slc: slice, factor: int, limit: int) -> slice:
    start = int(slc.start)
    stop = int(slc.stop)
    start = (start // factor) * factor
    stop = min(limit, ((stop + factor - 1) // factor) * factor)
    if stop <= start:
        stop = min(limit, start + factor)
    return slice(start, stop)


def hr_slices_to_lr(
    row: slice, col: slice, factor: int, lr_shape: tuple[int, int]
) -> tuple[slice, slice]:
    return (
        slice(row.start // factor, min(lr_shape[0], row.stop // factor)),
        slice(col.start // factor, min(lr_shape[1], col.stop // factor)),
    )


@dataclass
class RetrySnapshot:
    retry: int
    mosaic: np.ndarray
    mask: np.ndarray
    success_fraction: float
    n_tiles: int


@dataclass
class SpatialResult:
    mosaic: np.ndarray
    mask: np.ndarray
    mad_map: np.ndarray
    relative_uncertainty: np.ndarray
    residual_sigma: np.ndarray
    success_fraction: float
    n_retry_tiles: int
    retry_tiles: list[tuple[int, int, int, int]]
    history: list[RetrySnapshot]


def retry_failed_into_mosaic(
    mosaic: np.ndarray,
    mask: np.ndarray,
    mad_map: np.ndarray,
    input_image: np.ndarray,
    scale_factor: int,
    min_tile: int,
    overlap: int,
    retry_fn: RetryFn,
    retry_pass: int,
) -> tuple[np.ndarray, np.ndarray, int, list[tuple[int, int, int, int]]]:
    """Blend retried tile consensus into failed pixels only. Success pixels stay put."""
    cons = as_float_image(mosaic).copy()
    keep = np.asarray(mask, dtype=bool)
    combined_mad = as_float_image(mad_map).copy()
    accum = np.zeros_like(cons)
    weight = np.zeros_like(cons)
    boxes: list[tuple[int, int, int, int]] = []
    n_retry = 0
    tiles = failed_tiles(keep, min_tile, overlap)
    for tile_index, (row, col) in enumerate(tiles):
        row = align_to_factor(row, scale_factor, cons.shape[0])
        col = align_to_factor(col, scale_factor, cons.shape[1])
        lr_row, lr_col = hr_slices_to_lr(row, col, scale_factor, input_image.shape)
        crop = as_float_image(input_image[lr_row, lr_col])
        if crop.size == 0 or min(crop.shape) < 2:
            continue
        tile_cons, tile_mad = retry_fn(crop, tile_index, retry_pass)
        expected = (row.stop - row.start, col.stop - col.start)
        if tile_cons.shape != expected:
            tile_cons = upsample_to_shape(tile_cons, expected)
            tile_mad = upsample_to_shape(tile_mad, expected)
        w = cosine_weight(
            expected[0],
            expected[1],
            overlap,
            row_start=row.start,
            col_start=col.start,
            image_shape=cons.shape,
        )
        fail = ~keep[row, col]
        accum[row, col] += tile_cons * w * fail
        weight[row, col] += w * fail
        combined_mad[row, col] = np.where(fail, tile_mad, combined_mad[row, col])
        boxes.append((row.start, row.stop, col.start, col.stop))
        n_retry += 1
    filled = weight > 1e-8
    if np.any(filled):
        cons[filled] = accum[filled] / weight[filled]
    return as_float_image(cons), as_float_image(combined_mad), n_retry, boxes


def apply_spatial_keep(
    *,
    consensus: np.ndarray,
    mad_map: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    fallback: np.ndarray,
    input_image: np.ndarray,
    total_scale: int,
    scale_factor: int,
    config: RecursionConfig,
    retry_fn: RetryFn | None = None,
    reproject_fn: ReprojectFn | None = None,
) -> SpatialResult:
    cons = as_float_image(consensus)
    combined_mad = as_float_image(mad_map)
    mask, rel, z_hr = spatial_success_mask(cons, combined_mad, residual, sigma, total_scale, config)
    mosaic = mosaic_from_mask(cons, fallback, mask)
    n_retry = 0
    retry_boxes: list[tuple[int, int, int, int]] = []
    history = [
        RetrySnapshot(
            retry=0,
            mosaic=as_float_image(mosaic),
            mask=mask.copy(),
            success_fraction=float(mask.mean()),
            n_tiles=0,
        )
    ]
    spatial = config.spatial
    if (
        spatial.enabled
        and spatial.retry_failed_tiles
        and retry_fn is not None
        and not config.unconditional
        and not mask.all()
        and spatial.max_retries > 0
    ):
        working_residual = residual
        working_sigma = sigma
        for retry_pass in range(1, spatial.max_retries + 1):
            if mask.all():
                break
            mosaic, combined_mad, n_this, boxes = retry_failed_into_mosaic(
                mosaic,
                mask,
                combined_mad,
                input_image,
                scale_factor,
                spatial.min_tile,
                spatial.overlap,
                retry_fn,
                retry_pass,
            )
            if n_this == 0:
                break
            n_retry += n_this
            retry_boxes.extend(boxes)
            if reproject_fn is not None:
                working_residual, working_sigma = reproject_fn(mosaic)
            new_mask, rel, z_hr = spatial_success_mask(
                mosaic, combined_mad, working_residual, working_sigma, total_scale, config
            )
            mask = mask | new_mask
            mosaic = mosaic_from_mask(mosaic, fallback, mask)
            history.append(
                RetrySnapshot(
                    retry=retry_pass,
                    mosaic=as_float_image(mosaic),
                    mask=mask.copy(),
                    success_fraction=float(mask.mean()),
                    n_tiles=n_this,
                )
            )

    return SpatialResult(
        mosaic=as_float_image(mosaic),
        mask=mask,
        mad_map=as_float_image(combined_mad),
        relative_uncertainty=rel,
        residual_sigma=z_hr,
        success_fraction=float(mask.mean()),
        n_retry_tiles=n_retry,
        retry_tiles=retry_boxes,
        history=history,
    )
