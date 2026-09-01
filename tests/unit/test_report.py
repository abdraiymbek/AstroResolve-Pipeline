from astrsr.logging.report import render_report, render_results_table, results_table_rows


def _payload() -> dict:
    return {
        "title": "astrsr run test",
        "run_id": "x",
        "config_hash": "abc",
        "seed": 1,
        "device": "cpu",
        "data": {"source": "synthetic_fixture"},
        "model": {"name": "fake_sr"},
        "baselines": [
            {"name": "bicubic", "psnr": 24.1, "ssim": 0.46, "flux_error": 0.09, "centroid_error": 0.6}
        ],
        "solely": [
            {"name": "bicubic", "psnr": 24.1, "ssim": 0.46, "flux_error": 0.09, "centroid_error": 0.6},
            {"name": "swin2sr_x2", "psnr": 18.4, "ssim": 0.28, "flux_error": 0.05, "centroid_error": 0.65},
        ],
        "steps": [
            {
                "index": 0,
                "total_scale": 2,
                "n_members": 5,
                "n_kept": 5,
                "mean_agreement": 0.5,
                "mean_relative_uncertainty": 1.7,
                "reduced_chi2": 0.34,
                "flux_rel_error": 0.08,
                "decision": "accepted_spatial",
                "continue_recursion": False,
                "success_fraction": 0.42,
                "n_retry_tiles": 4,
                "truth_metrics": {
                    "psnr": 23.64,
                    "ssim": 0.48,
                    "flux_error": 0.064,
                    "centroid_error": 0.68,
                    "disagreement_error_correlation": 0.6,
                },
            }
        ],
        "accepted": {
            "depth": 1,
            "stop_reason": "accepted_spatial",
            "shape": [64, 64],
            "truth_metrics": {"psnr": 23.64, "ssim": 0.48, "flux_error": 0.064},
        },
        "conclusion": "Kept 42% of the 2x field.",
        "failures": [],
    }


def test_results_table_includes_mosaic_and_product() -> None:
    rows = results_table_rows(_payload())
    names = [row["name"] for row in rows]
    assert "bicubic" in names
    assert "swin2sr_x2" in names
    assert "gated mosaic 2x" in names
    assert "accepted product" in names
    table = render_results_table(rows)
    assert "| method | psnr | ssim | flux_error | centroid_error | note |" in table
    assert "keep 42%" in table


def test_report_puts_results_table_near_the_top() -> None:
    markdown = render_report(_payload())
    results_at = markdown.index("## Results vs held-out reference")
    recursion_at = markdown.index("## Recursion")
    assert results_at < recursion_at
    assert "gated mosaic 2x" in markdown
    assert "accepted product" in markdown
