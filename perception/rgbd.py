"""Run YOLOE on an aligned RGB-D pair and attach metric depth to detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from .common import MODEL_PATH, OUTPUT_DIR, RGBD_DATASET, natural_key

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=RGBD_DATASET,
        help="Folder containing Washington RGB-D *_crop.png pairs.",
    )
    parser.add_argument("--index", type=int, default=23, help="Zero-based RGB-D pair index.")
    parser.add_argument("--prompt", default="bottle", help="YOLOE open-vocabulary prompt.")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "rgbd")
    parser.add_argument("--show", action="store_true", help="Open the RGB-D result window.")
    return parser.parse_args()


def discover_pairs(dataset: Path) -> list[tuple[Path, Path, Path | None]]:
    pairs: list[tuple[Path, Path, Path | None]] = []
    for rgb_path in sorted(dataset.glob("*_crop.png"), key=natural_key):
        prefix = rgb_path.name.removesuffix("_crop.png")
        depth_path = dataset / f"{prefix}_depthcrop.png"
        mask_path = dataset / f"{prefix}_maskcrop.png"
        if depth_path.exists():
            pairs.append((rgb_path, depth_path, mask_path if mask_path.exists() else None))
    if not pairs:
        raise FileNotFoundError(f"No aligned *_crop.png / *_depthcrop.png pairs found in {dataset}")
    return pairs


def load_rgbd(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth_mm is None:
        raise ValueError(f"Could not read RGB-D pair: {rgb_path}, {depth_path}")
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
        raise ValueError(f"Expected uint16 single-channel depth PNG, got {depth_mm.dtype} {depth_mm.shape}")
    if rgb.shape[:2] != depth_mm.shape:
        raise ValueError(f"RGB/depth shapes differ: {rgb.shape[:2]} vs {depth_mm.shape}")
    return rgb, depth_mm


def detection_mask(result, detection_index: int, shape: tuple[int, int], xyxy: np.ndarray) -> np.ndarray:
    """Return a model segmentation mask, falling back to its bounding box."""
    mask = np.zeros(shape, dtype=np.uint8)
    if result.masks is not None and detection_index < len(result.masks.xy):
        polygon = np.round(result.masks.xy[detection_index]).astype(np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 1)
            return mask.astype(bool)

    x1, y1, x2, y2 = np.round(xyxy).astype(int)
    x1, x2 = np.clip([x1, x2], 0, shape[1])
    y1, y2 = np.clip([y1, y2], 0, shape[0])
    mask[y1:y2, x1:x2] = 1
    return mask.astype(bool)


def depth_colormap(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm > 0
    colored = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return colored
    low, high = np.percentile(depth_mm[valid], [2, 98])
    normalized = np.clip((depth_mm.astype(np.float32) - low) / max(high - low, 1), 0, 1)
    # Near pixels are warm, far pixels are cool.
    mapped = cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[valid] = mapped[valid]
    return colored


def add_panel_titles(image: np.ndarray, half_width: int) -> np.ndarray:
    bar_height = 42
    canvas = cv2.copyMakeBorder(image, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(canvas, "RGB + YOLOE", (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "Depth", (half_width + 10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args.dataset)
    if not 0 <= args.index < len(pairs):
        raise IndexError(f"--index must be in [0, {len(pairs) - 1}], got {args.index}")

    rgb_path, depth_path, dataset_mask_path = pairs[args.index]
    rgb, depth_mm = load_rgbd(rgb_path, depth_path)
    device: int | str = 0 if torch.cuda.is_available() else "cpu"

    model = YOLOE(str(args.model))
    model.set_classes([args.prompt])
    result = model.predict(rgb, device=device, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

    # Render the segmentation mask first. Boxes and labels are drawn after
    # upscaling because the source crops are only about 80 px wide.
    annotated = result.plot(conf=False, labels=False, boxes=False)
    detections = []
    for i, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        region = detection_mask(result, i, depth_mm.shape, xyxy)
        valid_depth = depth_mm[region & (depth_mm > 0)]
        median_mm = float(np.median(valid_depth)) if valid_depth.size else None
        confidence = float(box.conf.item())
        class_name = result.names[int(box.cls.item())]
        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "bbox_xyxy": [float(value) for value in xyxy],
                "median_depth_mm": median_mm,
                "valid_depth_pixels": int(valid_depth.size),
            }
        )
    depth_color = depth_colormap(depth_mm)
    render_scale = 720 / annotated.shape[0]
    render_size = (round(annotated.shape[1] * render_scale), 720)
    annotated_display = cv2.resize(annotated, render_size, interpolation=cv2.INTER_CUBIC)
    depth_display = cv2.resize(depth_color, render_size, interpolation=cv2.INTER_NEAREST)
    for detection in detections:
        x1, y1, x2, y2 = [round(value * render_scale) for value in detection["bbox_xyxy"]]
        cv2.rectangle(annotated_display, (x1, y1), (x2, y2), (50, 220, 50), 3)
        depth_text = (
            f"{detection['median_depth_mm'] / 1000:.2f} m"
            if detection["median_depth_mm"] is not None
            else "depth N/A"
        )
        label = f"{detection['class']} {detection['confidence']:.2f} | {depth_text}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(text_height + 8, y1)
        cv2.rectangle(
            annotated_display,
            (max(0, x1), label_y - text_height - 8),
            (min(annotated_display.shape[1] - 1, x1 + text_width + 8), label_y + 4),
            (50, 220, 50),
            -1,
        )
        cv2.putText(
            annotated_display,
            label,
            (max(0, x1) + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    display = add_panel_titles(cv2.hconcat([annotated_display, depth_display]), annotated_display.shape[1])

    args.output.mkdir(parents=True, exist_ok=True)
    output_image = args.output / "rgbd_detection.jpg"
    output_json = args.output / "rgbd_detection.json"
    cv2.imwrite(str(output_image), display)
    report = {
        "dataset": "Washington RGB-D Object Dataset / water_bottle_1",
        "dataset_pairs": len(pairs),
        "frame_index": args.index,
        "rgb": str(rgb_path.resolve()),
        "depth": str(depth_path.resolve()),
        "dataset_mask": str(dataset_mask_path.resolve()) if dataset_mask_path else None,
        "depth_format": "uint16 millimeters; 0 means missing",
        "prompt": args.prompt,
        "confidence_threshold": args.conf,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "detections": detections,
        "speed_ms": result.speed,
        "output_image": str(output_image.resolve()),
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if args.show:
        window = "YOLOE RGB-D result"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, display)
        print("Press any key in the image window or close it to exit.", flush=True)
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
