"""Evaluate RGB-D localization against masks or independently measured XYZ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from .common import MODEL_PATH, OUTPUT_DIR, RGBD_DATASET
from .geometry import CameraIntrinsics, WASHINGTON_INTRINSICS, load_crop_origin, localize_mask
from .rgbd import detection_mask, discover_pairs, load_rgbd


def mask_iou(predicted: np.ndarray, reference: np.ndarray) -> float:
    union = int(np.count_nonzero(predicted | reference))
    return float(np.count_nonzero(predicted & reference) / union) if union else 0.0


def read_ground_truth(path: Path) -> dict[str, np.ndarray]:
    """CSV: frame,x_m,y_m,z_m; frame is the RGB crop filename."""
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            xyz = np.array([float(row[key]) for key in ("x_m", "y_m", "z_m")])
            if not np.isfinite(xyz).all():
                raise ValueError(f"Non-finite ground truth for {row['frame']}")
            rows[row["frame"]] = xyz
    if not rows:
        raise ValueError("Ground-truth CSV has no rows")
    return rows


def summarize_errors(rows: list[dict], field: str) -> dict | None:
    errors = [np.asarray(row[field], dtype=np.float64) for row in rows if row.get(field) is not None]
    if not errors:
        return None
    values = np.stack(errors)
    distances = np.linalg.norm(values, axis=1)
    return {
        "count": len(values),
        "mae_xyz_m": np.mean(np.abs(values), axis=0).tolist(),
        "rmse_xyz_m": np.sqrt(np.mean(values ** 2, axis=0)).tolist(),
        "mean_euclidean_error_m": float(np.mean(distances)),
        "median_euclidean_error_m": float(np.median(distances)),
        "p95_euclidean_error_m": float(np.percentile(distances, 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=RGBD_DATASET)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--ground-truth-csv", type=Path,
                        help="Optional independent camera-frame XYZ: frame,x_m,y_m,z_m.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "rgbd_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    pairs = discover_pairs(args.dataset)
    if any(pair[2] is None for pair in pairs):
        raise FileNotFoundError("Evaluation requires annotated *_maskcrop.png files")
    intrinsics = CameraIntrinsics.from_json(args.intrinsics) if args.intrinsics else WASHINGTON_INTRINSICS
    ground_truth = read_ground_truth(args.ground_truth_csv) if args.ground_truth_csv else None
    sample_indices = np.unique(np.linspace(0, len(pairs) - 1,
                                           min(args.samples, len(pairs)), dtype=int))
    model = YOLOE(str(args.model))
    model.set_classes([args.prompt])
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    rows = []
    for index in sample_indices:
        rgb_path, depth_path, mask_path, loc_path = pairs[int(index)]
        rgb, depth = load_rgbd(rgb_path, depth_path)
        reference_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if reference_mask is None or reference_mask.shape != depth.shape:
            raise ValueError(f"Invalid reference mask: {mask_path}")
        reference_mask = reference_mask > 0
        origin = load_crop_origin(loc_path)
        reference = localize_mask(depth, reference_mask, intrinsics, origin)
        prediction = model.predict(rgb, device=device, imgsz=640, conf=args.conf,
                                   retina_masks=True, verbose=False)[0]
        row = {
            "frame_index": int(index), "frame": rgb_path.name,
            "reference_xyz_m": reference["xyz_m"],
            "reference_depth_coverage": reference["depth_coverage"],
            "detected": bool(len(prediction.boxes)),
            "mask_iou": None,
            "predicted_xyz_m": None,
            "predicted_depth_coverage": None,
            "error_vs_dataset_mask_xyz_m": None,
            "error_vs_measured_xyz_m": None,
        }
        if len(prediction.boxes):
            # The turntable dataset has one object. Use the highest-confidence
            # prediction, without selecting by reference mask IoU.
            box_index = int(np.argmax(prediction.boxes.conf.cpu().numpy()))
            box = prediction.boxes[box_index]
            mask = detection_mask(prediction, box_index, depth.shape,
                                  box.xyxy[0].cpu().numpy())
            localized = localize_mask(depth, mask, intrinsics, origin)
            row["mask_iou"] = mask_iou(mask, reference_mask)
            row["confidence"] = float(box.conf.item())
            row["predicted_depth_coverage"] = localized["depth_coverage"]
            row["predicted_xyz_m"] = localized["xyz_m"]
            if localized["xyz_m"] is not None and reference["xyz_m"] is not None:
                error = np.asarray(localized["xyz_m"]) - np.asarray(reference["xyz_m"])
                row["error_vs_dataset_mask_xyz_m"] = error.tolist()
            if ground_truth is not None and rgb_path.name in ground_truth and localized["xyz_m"] is not None:
                error = np.asarray(localized["xyz_m"]) - ground_truth[rgb_path.name]
                row["error_vs_measured_xyz_m"] = error.tolist()
        rows.append(row)
    report = {
        "dataset": str(args.dataset.resolve()),
        "prompt": args.prompt,
        "confidence_threshold": args.conf,
        "intrinsics": intrinsics.to_dict(),
        "sample_count": len(rows),
        "detection_rate": sum(row["detected"] for row in rows) / len(rows),
        "localization_rate": sum(row["predicted_xyz_m"] is not None for row in rows) / len(rows),
        "mean_mask_iou": float(np.mean([row["mask_iou"] for row in rows if row["mask_iou"] is not None])) if any(row["mask_iou"] is not None for row in rows) else None,
        "mean_reference_depth_coverage": float(np.mean([row["reference_depth_coverage"] for row in rows])),
        "mask_reference_agreement": summarize_errors(rows, "error_vs_dataset_mask_xyz_m"),
        "absolute_accuracy_if_measured_reference_provided": summarize_errors(rows, "error_vs_measured_xyz_m"),
        "measured_reference_frames": sum(row["frame"] in ground_truth for row in rows) if ground_truth is not None else 0,
        "interpretation": "Dataset-mask agreement measures the YOLOE mask/localization pipeline against an annotated-mask reference on the same depth map. It is not absolute camera accuracy. For absolute accuracy, independently measure XYZ for a clearly defined corresponding target point or use a calibrated target with known geometry.",
        "frames": rows,
    }
    if ground_truth is not None and report["measured_reference_frames"] == 0:
        raise ValueError("No sampled frame names match --ground-truth-csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2), flush=True)
    print(f"Per-frame results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
