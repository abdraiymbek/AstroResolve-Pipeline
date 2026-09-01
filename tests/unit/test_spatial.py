import numpy as np

from astrsr.config import RecursionConfig
from astrsr.recursion.spatial import (
    align_to_factor,
    apply_spatial_keep,
    cosine_weight,
    failed_tiles,
    mosaic_from_mask,
    relative_uncertainty_map,
    retry_failed_into_mosaic,
    spatial_success_mask,
    upsample_block,
)


def test_upsample_block_repeats_pixels() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    up = upsample_block(image, 2)
    assert up.shape == (4, 4)
    assert up[0, 0] == 1.0
    assert up[1, 1] == 1.0
    assert up[0, 2] == 2.0


def test_relative_uncertainty_grows_with_mad() -> None:
    consensus = np.ones((4, 4))
    low = relative_uncertainty_map(consensus, np.full((4, 4), 0.01))
    high = relative_uncertainty_map(consensus, np.full((4, 4), 1.0))
    assert float(high.mean()) > float(low.mean())


def test_mask_fails_high_uncertainty() -> None:
    consensus = np.ones((4, 4))
    mad = np.ones((4, 4))
    residual = np.zeros((2, 2))
    sigma = np.ones((2, 2))
    cfg = RecursionConfig()
    mask, rel, _ = spatial_success_mask(consensus, mad, residual, sigma, 2, cfg)
    assert rel.mean() > cfg.gates.max_relative_uncertainty
    assert not mask.any()


def test_mask_fails_high_residual() -> None:
    consensus = np.ones((4, 4))
    mad = np.zeros((4, 4))
    residual = np.full((2, 2), 10.0)
    sigma = np.ones((2, 2))
    cfg = RecursionConfig()
    mask, _, z_hr = spatial_success_mask(consensus, mad, residual, sigma, 2, cfg)
    assert z_hr.shape == (4, 4)
    assert float(z_hr.mean()) > cfg.spatial.max_residual_sigma
    assert not mask.any()


def test_mask_passes_quiet_field() -> None:
    consensus = np.ones((4, 4))
    mad = np.full((4, 4), 1e-6)
    residual = np.zeros((2, 2))
    sigma = np.ones((2, 2))
    cfg = RecursionConfig()
    mask, _, _ = spatial_success_mask(consensus, mad, residual, sigma, 2, cfg)
    assert mask.all()


def test_mosaic_keeps_consensus_on_success() -> None:
    consensus = np.full((4, 4), 5.0)
    fallback = np.full((2, 2), 1.0)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    mosaic = mosaic_from_mask(consensus, fallback, mask)
    assert mosaic[0, 0] == 5.0
    assert mosaic[3, 3] == 1.0


def test_failed_tiles_empty_when_all_pass() -> None:
    mask = np.ones((32, 32), dtype=bool)
    assert failed_tiles(mask, min_tile=16, overlap=4) == []


def test_failed_tiles_quad_on_large_fail_region() -> None:
    mask = np.ones((64, 64), dtype=bool)
    mask[0:40, 0:40] = False
    tiles = failed_tiles(mask, min_tile=16, overlap=4)
    assert len(tiles) == 4


def test_align_to_factor_snaps_to_even() -> None:
    aligned = align_to_factor(slice(3, 17), factor=2, limit=32)
    assert aligned.start % 2 == 0
    assert aligned.stop % 2 == 0
    assert aligned.stop > aligned.start


def test_cosine_weight_tapers_edges() -> None:
    weight = cosine_weight(16, 16, overlap=4)
    assert weight[0, 0] < weight[8, 8]
    assert np.isclose(weight[8, 8], 1.0)


def test_cosine_weight_keeps_image_border() -> None:
    weight = cosine_weight(8, 8, overlap=4, row_start=0, col_start=0, image_shape=(8, 8))
    assert np.isclose(weight[0, 0], 1.0)
    assert np.isclose(weight[-1, -1], 1.0)


def test_retry_does_not_overwrite_success_pixels() -> None:
    mosaic = np.full((8, 8), 3.0)
    mask = np.ones((8, 8), dtype=bool)
    mask[4:, 4:] = False
    mosaic[4:, 4:] = 0.0
    mad = np.ones((8, 8))
    observation = np.ones((4, 4))

    def retry_fn(crop: np.ndarray, tile_index: int) -> tuple[np.ndarray, np.ndarray]:
        del tile_index
        height, width = crop.shape
        return np.full((height * 2, width * 2), 9.0), np.full((height * 2, width * 2), 0.1)

    out, out_mad, n_retry, boxes = retry_failed_into_mosaic(
        mosaic, mask, mad, observation, scale_factor=2, min_tile=2, overlap=1, retry_fn=retry_fn
    )
    assert n_retry >= 1
    assert boxes
    assert np.allclose(out[:4, :4], 3.0)
    assert np.allclose(out[4:, 4:], 9.0)


def test_apply_spatial_keep_freezes_failures_without_retry() -> None:
    consensus = np.ones((8, 8))
    mad = np.zeros((8, 8))
    mad[:, 4:] = 10.0
    residual = np.zeros((4, 4))
    sigma = np.ones((4, 4))
    fallback = np.full((4, 4), 2.0)
    cfg = RecursionConfig()
    result = apply_spatial_keep(
        consensus=consensus,
        mad_map=mad,
        residual=residual,
        sigma=sigma,
        fallback=fallback,
        input_image=fallback,
        total_scale=2,
        scale_factor=2,
        config=cfg,
        retry_fn=None,
    )
    assert 0.0 < result.success_fraction < 1.0
    assert np.allclose(result.mosaic[:, :4], 1.0)
    assert np.allclose(result.mosaic[:, 4:], 2.0)
    assert result.n_retry_tiles == 0
