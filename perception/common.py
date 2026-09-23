"""Shared paths and ordering for the perception demos."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = Path(__file__).with_name("prompt_library.json")
MODEL_PATH = ROOT / "yoloe-v8s-seg.pt"
RGBD_DATASET = ROOT / "data" / "rgbd" / "water_bottle_1"
OUTPUT_DIR = ROOT / "outputs"


def natural_key(path: Path) -> list[int | str]:
    """Sort numbered RGB-D frames in capture order."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]
