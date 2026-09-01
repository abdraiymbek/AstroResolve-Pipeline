"""Swin2SR x2 adapter. Natural-image RGB model applied to grayscale intensity."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from astrsr.utils.arrays import as_float_image

SWIN2SR_ID = "caidas/swin2SR-classical-sr-x2-64"
SWIN2SR_LICENSE = "Apache-2.0"


class Swin2SRModel:
    name = "swin2sr_x2"
    scale = 2
    license = SWIN2SR_LICENSE

    def __init__(self, checkpoint: str = SWIN2SR_ID, device: str = "cpu") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        self._processor = None
        self.last_conversion: dict[str, Any] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from transformers import Swin2SRForImageSuperResolution

        try:
            from transformers import Swin2SRImageProcessorPil as Processor
        except ImportError:
            from transformers import Swin2SRImageProcessor as Processor

        self._processor = Processor.from_pretrained(self.checkpoint)
        try:
            self._model = Swin2SRForImageSuperResolution.from_pretrained(
                self.checkpoint,
                attn_implementation="eager",
            )
        except TypeError:
            self._model = Swin2SRForImageSuperResolution.from_pretrained(self.checkpoint)
        self._model.to(device=self.device, dtype=torch.float32)
        self._model.eval()

    def infer(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        self._load()
        data = as_float_image(image)
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if vmax <= vmin:
            vmax = vmin + 1.0
        scaled = (data - vmin) / (vmax - vmin)
        rgb = np.stack([scaled, scaled, scaled], axis=-1)
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        pil = Image.fromarray(rgb_u8, mode="RGB")
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            reconstruction = self._model(pixel_values=pixel_values).reconstruction
        rec = reconstruction.squeeze(0).float().clamp(0, 1).cpu().numpy()
        if rec.ndim != 3:
            raise RuntimeError(f"Unexpected Swin2SR output shape {rec.shape}")
        gray = rec.mean(axis=0)
        target_h = data.shape[0] * self.scale
        target_w = data.shape[1] * self.scale
        gray = gray[:target_h, :target_w]
        if gray.shape != (target_h, target_w):
            raise RuntimeError(
                f"Swin2SR cropped shape {gray.shape} != {(target_h, target_w)}. "
                "The checkpoint may have padded in a way P0 cannot crop safely."
            )
        native = gray * (vmax - vmin) + vmin
        self.last_conversion = {
            "vmin": vmin,
            "vmax": vmax,
            "model_input": "uint8_rgb_replicated_grayscale",
            "domain_assumption": (
                "A natural-image RGB super-resolution network is applied to a linear "
                "intensity map after 8-bit quantization. This is a documented domain mismatch."
            ),
            "checkpoint": self.checkpoint,
            "license": self.license,
            "device": self.device,
        }
        return as_float_image(native)

    def metadata(self) -> dict[str, Any]:
        meta = {
            "name": self.name,
            "scale": self.scale,
            "kind": "pretrained_sr",
            "checkpoint": self.checkpoint,
            "license": self.license,
            "device": self.device,
        }
        meta.update(self.last_conversion)
        return meta
