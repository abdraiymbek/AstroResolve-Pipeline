"""Galaxy Restormer adapter. Same-grid restoration applied after a 2x upsample."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from astrsr.models.baselines import interpolate
from astrsr.models.restormer_arch import Restormer
from astrsr.utils.arrays import as_float_image

WEIGHT_NAME = "galaxy_pretrained_model.pth"
ZENODO_ZIP = "https://zenodo.org/records/11378660/files/JOYONGSIK/GalaxyRestoration-v0.0.1.zip?download=1"
GITHUB_PTH = "https://github.com/JOYONGSIK/GalaxyRestoration/raw/main/galaxy_pretrained_model.pth"

RESTORMER_KWARGS = {
    "inp_channels": 1,
    "out_channels": 1,
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "BiasFree",
    "dual_pixel_task": False,
}


def default_weight_path() -> Path:
    return Path("checkpoints") / WEIGHT_NAME


def weights_available(path: Path | None = None) -> bool:
    target = path or default_weight_path()
    return target.is_file() and target.stat().st_size > 1_000_000


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (height, width)


class GalaxyRestormerModel:
    name = "galaxy_restormer"
    scale = 2
    license = "MIT (Park/Jo/Jee Galaxy Restormer; Restormer Apache-2.0 architecture)"

    def __init__(self, device: str = "cpu", weights: Path | None = None) -> None:
        self.device = device
        self.weights = Path(weights) if weights else default_weight_path()
        self._model: Restormer | None = None
        self.last_conversion: dict[str, Any] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        if not weights_available(self.weights):
            raise FileNotFoundError(
                f"Galaxy Restormer weights not found at {self.weights}. "
                "Place galaxy_pretrained_model.pth under checkpoints/."
            )
        model = Restormer(**RESTORMER_KWARGS)
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        state = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
        model.load_state_dict(state)
        model.to(device=self.device, dtype=torch.float32)
        model.eval()
        self._model = model

    def infer(self, image: np.ndarray) -> np.ndarray:
        self._load()
        assert self._model is not None
        up = interpolate(as_float_image(image), self.scale, "bicubic")
        vmin = float(np.min(up))
        vmax = float(np.max(up))
        if vmax <= vmin:
            vmax = vmin + 1.0
        scaled = ((up - vmin) / (vmax - vmin)).astype(np.float32)
        tensor = torch.from_numpy(scaled)[None, None, :, :].to(self.device)
        tensor, (height, width) = _pad_to_multiple(tensor, 8)
        with torch.no_grad():
            restored = self._model(tensor)
        out = restored.squeeze().detach().cpu().numpy()[:height, :width]
        native = as_float_image(out) * (vmax - vmin) + vmin
        self.last_conversion = {
            "upsample_before": "bicubic_2x",
            "normalization": "min_max_linear",
            "vmin": vmin,
            "vmax": vmax,
            "weights": str(self.weights),
            "note": (
                "Galaxy Restormer is a same-grid galaxy restoration network. "
                "Here it refines a 2x bicubic upsample. It is not a native 2x SR head."
            ),
        }
        return native

    def metadata(self) -> dict[str, Any]:
        meta = {
            "name": self.name,
            "scale": self.scale,
            "kind": "astronomy_restoration",
            "paper": "Park et al. 2024 ApJ, Galaxy Restormer",
            "license": self.license,
            "device": self.device,
            "weights": str(self.weights),
        }
        meta.update(self.last_conversion)
        return meta
