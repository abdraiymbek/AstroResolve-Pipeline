"""Load a user FITS or PNG as a known-truth reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from astrsr.utils.arrays import as_float_image, load_fits_image, load_png_image


def ingest_reference(path: str | Path, kind: str) -> tuple[Any, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Reference image not found: {source}")
    if kind == "fits":
        data = load_fits_image(source)
    elif kind == "png":
        data = load_png_image(source)
    else:
        raise ValueError(f"Unsupported ingest kind {kind}")
    image = as_float_image(data)
    meta = {
        "kind": kind,
        "path": str(source.resolve()),
        "shape": list(image.shape),
        "units": "native_file_units",
        "notes": "Loaded as a reference. Provenance and instrument metadata must be supplied by the operator.",
    }
    return image, meta
