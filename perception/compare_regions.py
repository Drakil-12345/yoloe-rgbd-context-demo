"""Compare YOLOE bounding boxes and segmentation on identical RGB-D frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from .common import MODEL_PATH, OUTPUT_DIR, RGBD_DATASET
from .evaluate_rgbd import mask_iou, summarize_errors
from .geometry import CameraIntrinsics, WASHINGTON_INTRINSICS, load_crop_origin, localize_mask
from .rgbd import box_mask, discover_pairs, load_rgbd, segmentation_mask


def region_quality(region: np.ndarray, reference: np.ndarray, depth: np.ndarray) -> dict:
    """Measure background included by a region using the dataset object mask."""
    region_pixels = int(np.count_nonzero(region))
    reference_pixels = int(np.count_nonzero(reference))
    overlap_pixels = int(np.count_nonzero(region & reference))
    valid = region & (depth > 0)
    valid_pixels = int(np.count_nonzero(valid))
    outside = region & ~reference
    return {
        "region_pixels": region_pixels,
        "mask_iou": mask_iou(region, reference),
        "reference_coverage_fraction": float(overlap_pixels / reference_pixels)
        if reference_pixels else None,
        "outside_reference_fraction": float(np.count_nonzero(outside) / region_pixels)
        if region_pixels else None,
        "valid_depth_outside_reference_fraction":
            float(np.count_nonzero(valid & ~reference) / valid_pixels)
            if valid_pixels else None,
    }


def evaluate_region(
    depth: np.ndarray,
    region: np.ndarray,
    reference_mask: np.ndarray,
    reference_xyz: list[float] | None,
    intrinsics: CameraIntrinsics,
    origin: tuple[float, float],
) -> dict:
    position = localize_mask(depth, region, intrinsics, origin)
    error = None
    if position["xyz_m"] is not None and reference_xyz is not None:
        error = (np.asarray(position["xyz_m"]) - np.asarray(reference_xyz)).tolist()
    return {
        **region_quality(region, reference_mask, depth),
        "position": position,
        "error_vs_dataset_mask_xyz_m": error,
    }


def mean_field(rows: list[dict], method: str, field: str) -> float | None:
    values = [row[method][field] for row in rows
              if row[method] is not None and row[method][field] is not None]
    return float(np.mean(values)) if values else None


def summarize_method(rows: list[dict], method: str) -> dict:
    outcomes = [row[method] for row in rows if row[method] is not None]
    return {
        "region_available_count": len(outcomes),
        "localization_count": sum(outcome["position"]["xyz_m"] is not None
                                  for outcome in outcomes),
        "localization_rate_all_frames":
            sum(outcome["position"]["xyz_m"] is not None for outcome in outcomes) / len(rows),
        "mean_depth_coverage": float(np.mean([outcome["position"]["depth_coverage"]
                                               for outcome in outcomes])) if outcomes else None,
        "mean_mask_iou": mean_field(rows, method, "mask_iou"),
        "mean_reference_coverage_fraction": mean_field(
            rows, method, "reference_coverage_fraction"),
        "mean_outside_reference_fraction": mean_field(rows, method, "outside_reference_fraction"),
        "mean_valid_depth_outside_reference_fraction":
            mean_field(rows, method, "valid_depth_outside_reference_fraction"),
        "agreement_with_dataset_mask_xyz": summarize_errors(
            outcomes, "error_vs_dataset_mask_xyz_m"),
    }


def summarize_paired(rows: list[dict]) -> dict:
    paired = [row for row in rows if row["box"] is not None
              and row["segmentation"] is not None
              and row["box"]["error_vs_dataset_mask_xyz_m"] is not None
              and row["segmentation"]["error_vs_dataset_mask_xyz_m"] is not None]
    if not paired:
        return {"count": 0}
    box_error = np.asarray([row["box"]["error_vs_dataset_mask_xyz_m"] for row in paired])
    seg_error = np.asarray([row["segmentation"]["error_vs_dataset_mask_xyz_m"] for row in paired])
    box_norm = np.linalg.norm(box_error, axis=1)
    seg_norm = np.linalg.norm(seg_error, axis=1)
    tied = np.isclose(box_norm, seg_norm, atol=1e-9, rtol=0)
    return {
        "count": len(paired),
        "box_mean_xyz_disagreement_m": float(np.mean(box_norm)),
        "segmentation_mean_xyz_disagreement_m": float(np.mean(seg_norm)),
        # Per-frame annotation removes much of the real turntable motion.
        # This is a consistency proxy, not temporal sensor noise.
        "box_reference_relative_residual_std_xyz_m": np.std(box_error, axis=0).tolist(),
        "segmentation_reference_relative_residual_std_xyz_m": np.std(seg_error, axis=0).tolist(),
        "box_lower_disagreement_count": int(np.count_nonzero(~tied & (box_norm < seg_norm))),
        "segmentation_lower_disagreement_count": int(np.count_nonzero(~tied & (seg_norm < box_norm))),
        "equal_disagreement_count": int(np.count_nonzero(tied)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=RGBD_DATASET)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "region_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    pairs = discover_pairs(args.dataset)
    if any(pair[2] is None for pair in pairs):
        raise FileNotFoundError("Comparison requires annotated *_maskcrop.png files")
    intrinsics = CameraIntrinsics.from_json(args.intrinsics) if args.intrinsics else WASHINGTON_INTRINSICS
    sample_indices = np.unique(np.linspace(0, len(pairs) - 1,
                                           min(args.samples, len(pairs)), dtype=int))
    model = YOLOE(str(args.model))
    model.set_classes([args.prompt])
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    rows = []
    for number, index in enumerate(sample_indices, start=1):
        rgb_path, depth_path, mask_path, loc_path = pairs[int(index)]
        rgb, depth = load_rgbd(rgb_path, depth_path)
        reference_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if reference_mask is None or reference_mask.shape != depth.shape:
            raise ValueError(f"Invalid reference mask: {mask_path}")
        reference_mask = reference_mask > 0
        origin = load_crop_origin(loc_path)
        reference_xyz = localize_mask(depth, reference_mask, intrinsics, origin)["xyz_m"]
        prediction = model.predict(rgb, device=device, imgsz=args.imgsz,
                                   conf=args.conf, retina_masks=True, verbose=False)[0]
        row = {"frame_index": int(index), "frame": rgb_path.name,
               "detected": bool(len(prediction.boxes)), "confidence": None,
               "reference_xyz_m": reference_xyz, "box": None, "segmentation": None}
        if len(prediction.boxes):
            # Same highest-confidence detection for both methods; annotation
            # never influences which detection is selected.
            box_index = int(np.argmax(prediction.boxes.conf.cpu().numpy()))
            box = prediction.boxes[box_index]
            row["confidence"] = float(box.conf.item())
            bbox = box_mask(depth.shape, box.xyxy[0].cpu().numpy())
            row["box"] = evaluate_region(depth, bbox, reference_mask,
                                         reference_xyz, intrinsics, origin)
            segmented = segmentation_mask(prediction, box_index, depth.shape)
            if segmented is not None:
                row["segmentation"] = evaluate_region(
                    depth, segmented, reference_mask, reference_xyz, intrinsics, origin)
        rows.append(row)
        if number % 25 == 0 or number == len(sample_indices):
            print(f"Processed {number}/{len(sample_indices)} RGB-D frames", flush=True)

    report = {
        "dataset": str(args.dataset.resolve()),
        "prompt": args.prompt,
        "confidence_threshold": args.conf,
        "intrinsics": intrinsics.to_dict(),
        "sample_count": len(rows),
        "detection_count": sum(row["detected"] for row in rows),
        "detection_rate": sum(row["detected"] for row in rows) / len(rows),
        "box": summarize_method(rows, "box"),
        "segmentation": summarize_method(rows, "segmentation"),
        "paired_localized_frames": summarize_paired(rows),
        "interpretation": (
            "Both regions come from the same YOLOE prediction. Outside-reference fractions "
            "measure overlap with the dataset's annotated object mask, including valid depth "
            "outside it; they are proxies for background contamination. XYZ disagreement and "
            "residual spread compare with the annotated mask on the same depth map. The turntable "
            "object moves, so residual spread is not temporal sensor noise. None of these metrics "
            "measure absolute camera accuracy or the physical object center."
        ),
        "frames": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"},
                     indent=2), flush=True)
    print(f"Per-frame results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
