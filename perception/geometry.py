"""Metric 3D localization from aligned depth and an object mask."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float = 0.001

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0 or self.depth_scale_m <= 0:
            raise ValueError("fx, fy, and depth_scale_m must be positive")

    @classmethod
    def from_json(cls, path: Path) -> CameraIntrinsics:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{key: data[key] for key in ("fx", "fy", "cx", "cy")},
                   depth_scale_m=data.get("depth_scale_m", 0.001))

    def to_dict(self) -> dict:
        return asdict(self)


# Washington RGB-D depthToCloud.m uses a one-indexed principal point (320, 240).
# Pixel arrays here are zero-indexed, so use (319, 239). A crop's loc.txt is
# one-indexed and is converted separately below.
WASHINGTON_INTRINSICS = CameraIntrinsics(570.3, 570.3, 319.0, 239.0)


def load_crop_origin(loc_path: Path | None) -> tuple[float, float]:
    """Return the crop origin as zero-indexed full-frame pixel coordinates."""
    if loc_path is None:
        return 0.0, 0.0
    values = np.loadtxt(loc_path, delimiter=",", dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.isfinite(values).all():
        raise ValueError(f"Expected x,y in {loc_path}")
    return float(values[0] - 1), float(values[1] - 1)


def localize_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    crop_origin: tuple[float, float] = (0.0, 0.0),
    depth_outlier_m: float = 0.12,
    min_valid_pixels: int = 20,
) -> dict:
    """Estimate a representative 3D coordinate for the visible object region.

    The 2D anchor is the mean of inlier depth pixels in the mask. Z is their
    median depth. This remains defined when the exact center pixel lacks depth,
    but the resulting point is not guaranteed to be on the physical surface.
    """
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Depth and mask must be aligned two-dimensional arrays")
    if depth_outlier_m < 0 or min_valid_pixels < 1:
        raise ValueError("depth_outlier_m must be >= 0 and min_valid_pixels >= 1")
    region = mask.astype(bool)
    mask_pixels = int(np.count_nonzero(region))
    valid = region & np.isfinite(depth) & (depth > 0)
    valid_pixels = int(np.count_nonzero(valid))
    result = {
        "status": "ok",
        "coordinate_frame": "camera_optical: +X right, +Y down, +Z forward",
        "point_definition": "representative visible-region coordinate: inlier-pixel mean ray at median depth",
        "mask_pixels": mask_pixels,
        "valid_depth_pixels": valid_pixels,
        "depth_coverage": valid_pixels / mask_pixels if mask_pixels else 0.0,
        "inlier_pixels": 0,
        "xyz_m": None,
        "anchor_pixel_full_xy": None,
        "depth_median_m": None,
        "depth_mad_m": None,
        "radial_distance_m": None,
    }
    if valid_pixels < min_valid_pixels:
        result["status"] = "insufficient_depth"
        return result
    depth_m = depth.astype(np.float64) * intrinsics.depth_scale_m
    median_depth = float(np.median(depth_m[valid]))
    if depth_outlier_m > 0:
        valid &= np.abs(depth_m - median_depth) <= depth_outlier_m
    inlier_pixels = int(np.count_nonzero(valid))
    result["inlier_pixels"] = inlier_pixels
    if inlier_pixels < min_valid_pixels:
        result["status"] = "insufficient_depth_after_filter"
        return result
    rows, cols = np.nonzero(valid)
    u = float(np.mean(cols) + crop_origin[0])
    v = float(np.mean(rows) + crop_origin[1])
    z = float(np.median(depth_m[valid]))
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    result.update({
        "xyz_m": [float(x), float(y), z],
        "anchor_pixel_full_xy": [u, v],
        "depth_median_m": z,
        "depth_mad_m": float(np.median(np.abs(depth_m[valid] - z))),
        "radial_distance_m": float(np.linalg.norm([x, y, z])),
    })
    return result
