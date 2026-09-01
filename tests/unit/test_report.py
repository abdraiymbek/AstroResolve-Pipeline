from astrsr.logging.report import (
    render_report,
    render_results_table,
    render_retry_table,
    results_table_rows,
    retry_history_rows,
)


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
                "error_vs_truth_rate": 18.2,
                "n_retry_tiles": 4,
                "retry_history": [
                    {
                        "retry": 0,
                        "accepted": 0.42,
                        "n_tiles": 0,
                        "psnr": 23.64,
                        "ssim": 0.48,
                        "flux_error": 0.064,
                        "error_vs_truth_rate": 18.2,
                    },
                    {
                        "retry": 1,
                        "accepted": 0.51,
                        "n_tiles": 4,
                        "psnr": 23.9,
                        "ssim": 0.49,
                        "flux_error": 0.05,
                        "error_vs_truth_rate": 16.0,
                    },
                ],
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
    assert "| method | psnr | ssim | flux_error | centroid_error | error_vs_truth | note |" in table
    assert "keep 42%" in table


def test_retry_table_lists_each_pass() -> None:
    rows = retry_history_rows(_payload())
    assert [row["retry"] for row in rows] == [0, 1]
    table = render_retry_table(rows)
    assert "42.00%" in table
    assert "51.00%" in table
    assert "18.20%" in table


def test_report_puts_results_table_near_the_top() -> None:
    markdown = render_report(_payload())
    results_at = markdown.index("## Results vs held-out reference")
    recursion_at = markdown.index("## Recursion")
    assert results_at < recursion_at
    assert "gated mosaic 2x" in markdown
    assert "accepted product" in markdown
    assert "## Retry trajectory" in markdown
