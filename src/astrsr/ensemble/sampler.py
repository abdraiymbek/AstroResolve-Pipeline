"""Stochastic sampling around a deterministic super-resolution model."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from astrsr.config import EnsembleConfig
from astrsr.utils.arrays import as_float_image

TTAOp = tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]

TTA_OPS: list[TTAOp] = [
    ("identity", lambda x: x, lambda x: x),
    ("fliplr", np.fliplr, np.fliplr),
    ("flipud", np.flipud, np.flipud),
    ("rot90", lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 3)),
    ("rot180", lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 2)),
    ("rot270", lambda x: np.rot90(x, 3), lambda x: np.rot90(x, 1)),
    ("rot90_fliplr", lambda x: np.fliplr(np.rot90(x, 1)), lambda x: np.rot90(np.fliplr(x), 3)),
    ("flipud_fliplr", lambda x: np.flipud(np.fliplr(x)), lambda x: np.fliplr(np.flipud(x))),
]


def member_seed(base_seed: int, index: int, stride: int) -> int:
    return int(base_seed + index * stride)


def perturb_input(image: np.ndarray, sigma_rel: float, rng: np.random.Generator) -> np.ndarray:
    data = as_float_image(image)
    std = float(data.std())
    sigma = sigma_rel * (std if std > 0 else 1.0)
    if sigma <= 0:
        return data
    return as_float_image(data + rng.normal(0.0, sigma, size=data.shape))


def sample_one(
    observation: np.ndarray,
    model: Any,
    config: EnsembleConfig,
    index: int,
    base_seed: int,
    ops: list[TTAOp],
) -> tuple[np.ndarray, dict[str, Any]]:
    seed = member_seed(base_seed, index, config.stochasticity.seed_stride)
    rng = np.random.default_rng(seed)
    if hasattr(model, "set_rng"):
        model.set_rng(rng)
    perturbed = perturb_input(observation, config.stochasticity.input_sigma_rel, rng)
    op_name, forward, inverse = ops[index % len(ops)]
    prepared = as_float_image(np.asarray(forward(perturbed)))
    reconstruction = as_float_image(model.infer(prepared))
    restored = as_float_image(np.asarray(inverse(reconstruction)))
    if restored.shape != reconstruction.shape:
        raise RuntimeError("TTA inverse changed array shape")
    record = {
        "index": index,
        "seed": seed,
        "tta": op_name,
        "model": getattr(model, "name", type(model).__name__),
        "input_sigma_rel": config.stochasticity.input_sigma_rel,
        "shape": list(restored.shape),
    }
    return restored, record


def sample_ensemble(
    observation: np.ndarray,
    models: list[Any] | Any,
    config: EnsembleConfig,
    base_seed: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if not isinstance(models, list):
        models = [models]
    members: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    if config.mode == "model_zoo":
        tta_count = max(1, config.stochasticity.tta_per_member)
        ops = TTA_OPS[:tta_count] if config.stochasticity.tta else TTA_OPS[:1]
        index = 0
        for model in models:
            for _ in range(tta_count):
                restored, record = sample_one(observation, model, config, index, base_seed, ops)
                print(
                    f"ensemble member {index + 1} model={record['model']} tta={record['tta']}",
                    flush=True,
                )
                members.append(restored)
                records.append(record)
                index += 1
        return members, records

    ops = TTA_OPS if config.stochasticity.tta else TTA_OPS[:1]
    model = models[0]
    for index in range(config.samples):
        restored, record = sample_one(observation, model, config, index, base_seed, ops)
        print(
            f"ensemble member {index + 1}/{config.samples} seed={record['seed']} tta={record['tta']}",
            flush=True,
        )
        members.append(restored)
        records.append(record)
    return members, records
