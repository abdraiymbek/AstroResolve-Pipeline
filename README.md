# Astronomical gated recursive super-resolution

Research prototype. Not a production observatory pipeline.

The question this code is built to test:

> Can consensus and data-consistency checks make recursive astronomical super-resolution more trustworthy and reduce AI hallucinations?

The system does one 2x reconstruction many times, builds a consensus and a disagreement map, blurs the result back down to the original observation, and then either allows another 2x or stops. When a step fails globally, spatial keep keeps pixels that pass per-pixel uncertainty and shrink-back checks instead of rolling back the whole field. Failed regions can be retried as overlapped tiles; whatever still fails stays at the previous scale. Stopping is a valid result. Agreement among runs is stability, not proof a feature is real. The method does not bypass telescope information limits.

The scientific contract is in [Agents/Project.MD](Agents/Project.MD).

## Run

Python 3.12 through `uv`. Apple Silicon uses MPS when PyTorch sees it.

```text
uv sync --extra dev
uv run astrsr validate-config --config configs/p0_smoke.yaml
uv run astrsr run --config configs/p0_smoke.yaml
uv run astrsr run --config configs/p0_smoke.yaml --set ensemble.samples=16 --set recursion.max_depth=1
uv run astrsr report --run-id <id>
```

`configs/p0_smoke.yaml` downloads `caidas/swin2SR-classical-sr-x2-64` (Apache-2.0) on first neural run. `configs/p0_fake.yaml` exercises the same algorithm with a test stub and no download.

Gate thresholds in the YAML are placeholders. A conservative `min_agreement` will abstain and keep the observation. That is the intended behavior, not a crash. Loosen a knob with `--set` when you want to see a 2x accepted.

`uv run astrsr compare --config configs/p0_compare.yaml` runs each zoo member once, then the gated combination on the same observation.

After a run finishes, a markdown **results table** is printed at the end of the terminal output (one-shot methods, gated mosaic, accepted product). The same table is in `runs/<run_id>/report.md` under **Results vs held-out reference**. Re-print it with `uv run astrsr report --run-id <id>`.

Spatial keep defaults live under `recursion.spatial` in config (`enabled`, `max_residual_sigma`, `retry_failed_tiles`, `min_tile`, `overlap`, `max_retries`, `min_success_fraction_to_continue`). Step artifacts include `mosaic.npy` and `success_mask.npy`.

The ensemble can be one network sampled many times (`ensemble.mode=stochastic_single`) or a zoo of different reconstructors (`ensemble.mode=model_zoo`). Galaxy Restormer is skipped if `checkpoints/galaxy_pretrained_model.pth` is missing.

Runs are written to `runs/<run_id>/` and are never overwritten. Scientific arrays are `.npy` / `.fits`. PNGs are asinh previews only.

## Tests

```text
uv run pytest
```
