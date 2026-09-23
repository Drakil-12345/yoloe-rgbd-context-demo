"""Create a colored PLY point cloud from a Washington RGB-D crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .common import OUTPUT_DIR, RGBD_DATASET, natural_key


FOCAL_LENGTH_PX = 570.3
PRINCIPAL_POINT = (320.0, 240.0)
MM_PER_M = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=RGBD_DATASET,
    )
    parser.add_argument("--index", type=int, default=23, help="Zero-based pair index in natural frame order.")
    parser.add_argument("--all-pixels", action="store_true", help="Include background instead of the object mask.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth valid pixel.")
    parser.add_argument(
        "--depth-outlier-mm",
        type=float,
        default=120.0,
        help="For object clouds, reject depths farther than this from the masked median; 0 disables.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "point_cloud")
    parser.add_argument("--show", action="store_true", help="Open the generated three-view preview image.")
    return parser.parse_args()


def discover_frames(dataset: Path) -> list[tuple[Path, Path, Path, Path]]:
    frames = []
    for rgb_path in sorted(dataset.glob("*_crop.png"), key=natural_key):
        prefix = rgb_path.name.removesuffix("_crop.png")
        depth_path = dataset / f"{prefix}_depthcrop.png"
        mask_path = dataset / f"{prefix}_maskcrop.png"
        loc_path = dataset / f"{prefix}_loc.txt"
        if depth_path.exists() and mask_path.exists() and loc_path.exists():
            frames.append((rgb_path, depth_path, mask_path, loc_path))
    if not frames:
        raise FileNotFoundError(f"No complete RGB/depth/mask/loc frames found in {dataset}")
    return frames


def load_frame(paths: tuple[Path, Path, Path, Path]):
    rgb_path, depth_path, mask_path, loc_path = paths
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or depth_mm is None or mask is None:
        raise ValueError("Could not read the RGB-D frame")
    if depth_mm.dtype != np.uint16 or bgr.shape[:2] != depth_mm.shape or mask.shape != depth_mm.shape:
        raise ValueError("Expected aligned BGR uint8, depth uint16, and mask uint8 images")
    loc = np.loadtxt(loc_path, delimiter=",", dtype=np.float64).reshape(-1)
    if loc.size != 2:
        raise ValueError(f"Expected x,y in {loc_path}, got {loc}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth_mm, mask, loc


def backproject(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    top_left_xy: np.ndarray,
    object_only: bool,
    stride: int,
    depth_outlier_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official Washington RGB-D depthToCloud projection."""
    height, width = depth_mm.shape
    # loc.txt is one-indexed. These grids exactly match the official MATLAB:
    # (1:w) + (loc_x - 1) - center_x.
    x_grid = np.arange(width, dtype=np.float32) + top_left_xy[0] - PRINCIPAL_POINT[0]
    y_grid = np.arange(height, dtype=np.float32) + top_left_xy[1] - PRINCIPAL_POINT[1]
    pixel_x, pixel_y = np.meshgrid(x_grid, y_grid)
    z = depth_mm.astype(np.float32) / MM_PER_M
    valid = depth_mm > 0
    if object_only:
        valid &= mask > 0
        if depth_outlier_mm > 0 and np.any(valid):
            median_depth = float(np.median(depth_mm[valid]))
            valid &= np.abs(depth_mm.astype(np.float32) - median_depth) <= depth_outlier_mm
    if stride > 1:
        sampling = np.zeros_like(valid)
        sampling[::stride, ::stride] = True
        valid &= sampling

    x = pixel_x * z / FOCAL_LENGTH_PX
    y = pixel_y * z / FOCAL_LENGTH_PX
    points = np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)
    colors = rgb[valid].astype(np.uint8)
    return points, colors


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated from Washington RGB-D Object Dataset\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def set_equal_3d_axes(axis, xyz: np.ndarray) -> None:
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = max(float(np.ptp(xyz, axis=0).max()) / 2, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def save_preview(path: Path, points: np.ndarray, colors: np.ndarray, frame_name: str) -> None:
    # Flip Y only for plotting so the bottle appears upright. PLY preserves the
    # official camera convention: +X right, +Y down, +Z forward.
    plot_points = points.copy()
    plot_points[:, 1] *= -1
    plot_colors = colors.astype(np.float32) / 255.0
    views = [(18, -70, "Perspective"), (0, -90, "Front"), (90, -90, "Top")]
    figure = plt.figure(figsize=(14, 5), facecolor="#111111")
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d", facecolor="#181818")
        axis.scatter(
            plot_points[:, 0],
            plot_points[:, 2],
            plot_points[:, 1],
            c=plot_colors,
            s=2.0,
            depthshade=False,
            linewidths=0,
        )
        # Display axes are X, Z, Y-up for an intuitive upright preview.
        display_xyz = np.column_stack((plot_points[:, 0], plot_points[:, 2], plot_points[:, 1]))
        set_equal_3d_axes(axis, display_xyz)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title, color="white", pad=10)
        axis.set_xlabel("X (m)", color="white")
        axis.set_ylabel("Z (m)", color="white")
        axis.set_zlabel("Y up (m)", color="white")
        axis.tick_params(colors="#bbbbbb", labelsize=7)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.zaxis.set_major_locator(MaxNLocator(4))
        axis.grid(False)
    figure.suptitle(f"Colored point cloud — {frame_name} — {len(points):,} points", color="white", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    frames = discover_frames(args.dataset)
    if not 0 <= args.index < len(frames):
        raise IndexError(f"--index must be in [0, {len(frames) - 1}]")
    paths = frames[args.index]
    rgb, depth_mm, mask, loc = load_frame(paths)
    points, colors = backproject(
        rgb,
        depth_mm,
        mask,
        loc,
        not args.all_pixels,
        args.stride,
        args.depth_outlier_mm,
    )
    if not len(points):
        raise ValueError("The selected frame produced no valid 3D points")

    args.output.mkdir(parents=True, exist_ok=True)
    mode = "full" if args.all_pixels else "object"
    stem = paths[0].name.removesuffix("_crop.png")
    ply_path = args.output / f"{stem}_{mode}.ply"
    preview_path = args.output / f"{stem}_{mode}_preview.png"
    report_path = args.output / f"{stem}_{mode}.json"
    write_binary_ply(ply_path, points, colors)
    save_preview(preview_path, points, colors, stem)

    report = {
        "frame_index": args.index,
        "rgb": str(paths[0].resolve()),
        "depth": str(paths[1].resolve()),
        "mask": str(paths[2].resolve()),
        "loc_xy_one_indexed": loc.tolist(),
        "intrinsics_official_one_indexed": {
            "fx": FOCAL_LENGTH_PX,
            "fy": FOCAL_LENGTH_PX,
            "cx": PRINCIPAL_POINT[0],
            "cy": PRINCIPAL_POINT[1],
        },
        "intrinsics_equivalent_zero_indexed": {
            "fx": FOCAL_LENGTH_PX,
            "fy": FOCAL_LENGTH_PX,
            "cx": PRINCIPAL_POINT[0] - 1,
            "cy": PRINCIPAL_POINT[1] - 1,
        },
        "coordinate_system": "+X right, +Y down, +Z forward; meters",
        "object_mask_only": not args.all_pixels,
        "depth_outlier_threshold_mm": args.depth_outlier_mm if not args.all_pixels else None,
        "point_count": len(points),
        "bounds_m": {"min_xyz": points.min(axis=0).tolist(), "max_xyz": points.max(axis=0).tolist()},
        "centroid_m": points.mean(axis=0).tolist(),
        "median_z_m": float(np.median(points[:, 2])),
        "sensor_note": "Transparent bottle surfaces have missing/noisy Kinect depth, so the cloud is partial.",
        "ply": str(ply_path.resolve()),
        "preview": str(preview_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if args.show:
        preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
        if preview is None:
            raise RuntimeError(f"Could not open point-cloud preview: {preview_path}")
        window = "RGB-D point cloud preview"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, preview)
        print("Press Q or Esc in the preview window to close it.", flush=True)
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
