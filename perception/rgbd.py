"""Locate an object's visible surface in the RGB-D camera coordinate frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from .common import MODEL_PATH, OUTPUT_DIR, RGBD_DATASET, natural_key
from .geometry import CameraIntrinsics, WASHINGTON_INTRINSICS, load_crop_origin, localize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Washington RGB-D instance directory.")
    parser.add_argument("--rgb", type=Path, help="Aligned RGB image from another RGB-D camera.")
    parser.add_argument("--depth", type=Path, help="Aligned uint16 depth PNG for --rgb.")
    parser.add_argument("--mask", type=Path, help="Optional annotation mask for --rgb.")
    parser.add_argument("--crop-origin", type=float, nargs=2, metavar=("X", "Y"),
                        default=(0.0, 0.0), help="Zero-based full-frame crop origin; default 0 0.")
    parser.add_argument("--index", type=int, default=23, help="Zero-based RGB-D pair index.")
    parser.add_argument("--prompt", required=True, help="YOLOE open-vocabulary target.")
    parser.add_argument("--mask-source", choices=("yoloe", "dataset"), default="yoloe",
                        help="YOLOE segment/box or annotated dataset mask.")
    parser.add_argument("--intrinsics", type=Path,
                        help="JSON with fx,fy,cx,cy,depth_scale_m; default: Washington Kinect.")
    parser.add_argument("--depth-outlier-m", type=float, default=0.12)
    parser.add_argument("--min-valid-pixels", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "rgbd")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def discover_pairs(dataset: Path) -> list[tuple[Path, Path, Path | None, Path | None]]:
    pairs = []
    for rgb_path in sorted(dataset.glob("*_crop.png"), key=natural_key):
        prefix = rgb_path.name.removesuffix("_crop.png")
        depth_path = dataset / f"{prefix}_depthcrop.png"
        mask_path = dataset / f"{prefix}_maskcrop.png"
        loc_path = dataset / f"{prefix}_loc.txt"
        if depth_path.exists():
            pairs.append((rgb_path, depth_path,
                          mask_path if mask_path.exists() else None,
                          loc_path if loc_path.exists() else None))
    if not pairs:
        raise FileNotFoundError(f"No aligned RGB/depth pairs found in {dataset}")
    return pairs


def load_rgbd(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise ValueError(f"Could not read RGB-D pair: {rgb_path}, {depth_path}")
    if depth.dtype != np.uint16 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("Expected aligned RGB and single-channel uint16 depth PNG")
    return rgb, depth


def detection_mask(result, index: int, shape: tuple[int, int], xyxy: np.ndarray) -> np.ndarray:
    """Use the YOLOE segmentation polygon, falling back to its bounding box."""
    mask = np.zeros(shape, dtype=np.uint8)
    if result.masks is not None and index < len(result.masks.xy):
        polygon = np.round(result.masks.xy[index]).astype(np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 1)
            return mask.astype(bool)
    x1, y1, x2, y2 = np.round(xyxy).astype(int)
    x1, x2 = np.clip([x1, x2], 0, shape[1])
    y1, y2 = np.clip([y1, y2], 0, shape[0])
    mask[y1:y2, x1:x2] = 1
    return mask.astype(bool)


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    colored = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        normalized = np.clip((depth.astype(np.float32) - low) / max(high - low, 1), 0, 1)
        mapped = cv2.applyColorMap(((1 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored[valid] = mapped[valid]
    return colored


def render_result(
    rgb: np.ndarray,
    depth: np.ndarray,
    detections: list[dict],
    crop_origin: tuple[float, float],
) -> np.ndarray:
    """Show the camera-frame XYZ marker alongside the depth image."""
    scale = 720 / rgb.shape[0]
    size = (round(rgb.shape[1] * scale), 720)
    left = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(depth_colormap(depth), size, interpolation=cv2.INTER_NEAREST)
    labels = []
    for item in detections:
        x1, y1, x2, y2 = [round(value * scale) for value in item["bbox_xyxy"]]
        cv2.rectangle(left, (x1, y1), (x2, y2), (50, 220, 50), 3)
        position = item["position"]
        xyz = position["xyz_m"]
        coords = f"XYZ=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:.3f})m" if xyz else position["status"]
        score = f" {item['confidence']:.2f}" if item["confidence"] is not None else ""
        labels.append(f"{item['class']}{score} | {coords}")
        if position["anchor_pixel_full_xy"] is not None:
            u, v = position["anchor_pixel_full_xy"]
            marker = (round((u - crop_origin[0]) * scale),
                      round((v - crop_origin[1]) * scale))
            cv2.drawMarker(left, marker, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    display = cv2.hconcat([left, right])
    header_height = 42 + 26 * len(labels)
    display = cv2.copyMakeBorder(display, header_height, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(display, "RGB + object XYZ in camera frame", (10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(display, "Depth", (size[0] + 10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    for index, label in enumerate(labels):
        cv2.putText(display, label, (10, 61 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return display


def main() -> None:
    args = parse_args()
    if args.rgb is not None:
        if args.depth is None or args.intrinsics is None:
            raise SystemExit("--rgb requires --depth and --intrinsics for this camera")
        if args.dataset is not None:
            raise SystemExit("Choose either --dataset or --rgb/--depth")
        rgb_path, depth_path, mask_path, loc_path = args.rgb, args.depth, args.mask, None
        crop_origin = tuple(args.crop_origin)
    else:
        if args.depth is not None or args.mask is not None:
            raise SystemExit("--depth and --mask require --rgb")
        dataset = args.dataset or RGBD_DATASET
        pairs = discover_pairs(dataset)
        if not 0 <= args.index < len(pairs):
            raise IndexError(f"--index must be in [0, {len(pairs) - 1}]")
        rgb_path, depth_path, mask_path, loc_path = pairs[args.index]
        crop_origin = load_crop_origin(loc_path)
    rgb, depth = load_rgbd(rgb_path, depth_path)
    intrinsics = CameraIntrinsics.from_json(args.intrinsics) if args.intrinsics else WASHINGTON_INTRINSICS
    result = None
    if args.mask_source == "yoloe":
        model = YOLOE(str(args.model))
        model.set_classes([args.prompt])
        device: int | str = 0 if torch.cuda.is_available() else "cpu"
        result = model.predict(rgb, device=device, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        regions = []
        for index, box in enumerate(result.boxes):
            bbox = box.xyxy[0].cpu().numpy()
            regions.append((detection_mask(result, index, depth.shape, bbox),
                            result.names[int(box.cls.item())], float(box.conf.item()),
                            [float(value) for value in bbox]))
    else:
        if mask_path is None:
            raise FileNotFoundError("Dataset mode requires *_maskcrop.png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape:
            raise ValueError(f"Mask missing or not aligned: {mask_path}")
        rows, cols = np.nonzero(mask)
        regions = [(mask > 0, args.prompt, None,
                    [float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)])] if len(cols) else []
    detections = []
    for region, name, confidence, bbox in regions:
        position = localize_mask(depth, region, intrinsics, crop_origin,
                                 args.depth_outlier_m, args.min_valid_pixels)
        detections.append({"class": name, "confidence": confidence,
                           "bbox_xyxy": bbox, "position": position})
    display = render_result(rgb, depth, detections, crop_origin)
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / "localization.jpg"
    report_path = args.output / "localization.json"
    cv2.imwrite(str(image_path), display)
    report = {
        "dataset": args.dataset.name if args.dataset else (None if args.rgb else RGBD_DATASET.name),
        "frame_index": args.index if args.rgb is None else None,
        "rgb": str(rgb_path.resolve()),
        "depth": str(depth_path.resolve()),
        "mask_source": args.mask_source,
        "crop_origin_zero_indexed_xy": list(crop_origin),
        "intrinsics": intrinsics.to_dict(),
        "detections": detections,
        "speed_ms": result.speed if result is not None else None,
        "accuracy_note": "XYZ represents the observed object region in the camera frame; it is not the 3D volume center or necessarily a physical surface point. Absolute accuracy needs independently measured XYZ ground truth for the same target definition.",
        "output_image": str(image_path.resolve()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.show:
        window = "RGB-D object localization"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, display)
        print("Press Q or Esc to close the window.", flush=True)
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
