"""Import path + cwd helpers for calling into WeatherMesh-3."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEATHERMESH_DIR = REPO_ROOT / "WeatherMesh-3"

DATA_DIR = REPO_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
RAW_GFS_DIR = DATA_DIR / "raw" / "gfs"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def setup_weathermesh():
    if str(WEATHERMESH_DIR) not in sys.path:
        sys.path.insert(0, str(WEATHERMESH_DIR))


@contextmanager
def weathermesh_cwd():
    """WeatherMesh-3 reads constants/ relative to its repo root."""
    previous = os.getcwd()
    os.chdir(WEATHERMESH_DIR)
    try:
        yield
    finally:
        os.chdir(previous)
