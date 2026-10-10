"""Shared test fixtures: a tiny fake PlantVillage dataset, so tests need no real data or downloads."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPARE_SCRIPT = PROJECT_ROOT / "src" / "prepare_data.py"
FAKE_CLASSES = ["Pepper__bell___healthy", "Potato___Early_blight", "Tomato__Target_Spot"]
IMAGES_PER_CLASS = 12


def _write_random_images(folder: Path, n: int, seed: int, size=(40, 48)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        pixels = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"img_{i:03d}.JPG")


@pytest.fixture
def fake_source(tmp_path) -> Path:
    """Mimics the Kaggle layout, including the duplicate nested 'PlantVillage' folder."""
    source = tmp_path / "PlantVillage"
    for i, name in enumerate(FAKE_CLASSES):
        _write_random_images(source / name, IMAGES_PER_CLASS, seed=i)
        _write_random_images(source / "PlantVillage" / name, IMAGES_PER_CLASS, seed=100 + i)
    return source


def run_prepare(source: Path, dest: Path, per_class: int = 10, seed: int = 42) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), "--source", str(source), "--dest", str(dest),
         "--per-class", str(per_class), "--seed", str(seed)],
        capture_output=True, text=True,
    )


@pytest.fixture
def fake_data_dir(fake_source, tmp_path) -> Path:
    """A prepared train/val/test split (10 images per class -> 7/2/1)."""
    dest = tmp_path / "data"
    result = run_prepare(fake_source, dest)
    assert result.returncode == 0, result.stderr
    return dest
