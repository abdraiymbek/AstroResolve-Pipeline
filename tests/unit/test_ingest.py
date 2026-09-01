from pathlib import Path

import numpy as np
from PIL import Image

from astrsr.data.ingest import ingest_reference
from astrsr.utils.arrays import save_fits


def test_png_ingest(tmp_path: Path) -> None:
    path = tmp_path / "ref.png"
    Image.fromarray(np.full((24, 24), 80, dtype=np.uint8), mode="L").save(path)
    image, meta = ingest_reference(path, "png")
    assert image.shape == (24, 24)
    assert meta["kind"] == "png"


def test_fits_ingest(tmp_path: Path) -> None:
    path = tmp_path / "ref.fits"
    save_fits(path, np.ones((20, 18), dtype=np.float64))
    image, meta = ingest_reference(path, "fits")
    assert image.shape == (20, 18)
    assert meta["kind"] == "fits"
