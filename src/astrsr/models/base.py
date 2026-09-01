"""Shared inference interface for every reconstruction method."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SuperResModel(Protocol):
    name: str
    scale: int
    license: str

    def infer(self, image: np.ndarray) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...
