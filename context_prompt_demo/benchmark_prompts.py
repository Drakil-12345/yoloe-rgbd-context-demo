"""Benchmark prompt candidates on labeled Washington RGB-D object crops."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def box_iou(box, target) -> float:
    x1, y1 = max(box[0], target[0]), max(box[1], target[1])
    x2, y2 = min(box[2], target[2]), min(box[3], target[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    return intersection / max(box_area + target_area - intersection, 1e-9)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=here / "prompt_library.json")
    parser.add_argument("--concept", default="water_bottle")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=here.parent / "rgbd_demo" / "dataset" / "rgbd-dataset" / "water_bottle" / "water_bottle_1",
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=here / "results" / "prompt_benchmark.json")
    args = parser.parse_args()

    library = json.loads(args.library.read_text(encoding="utf-8"))
    prompts = [row["text"] for row in library["concepts"][args.concept]["prompts"]]
    rgb_paths = sorted(args.dataset.glob("*_crop.png"), key=natural_key)
    sample_indices = np.linspace(0, len(rgb_paths) - 1, min(args.samples, len(rgb_paths)), dtype=int)
    sample_paths = [rgb_paths[index] for index in sample_indices]
    ground_truth = []
    for rgb_path in sample_paths:
        mask_path = rgb_path.with_name(rgb_path.name.replace("_crop.png", "_maskcrop.png"))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(mask > 0)
        ground_truth.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLOE(str(here.parent / "yoloe-v8s-seg.pt"))
    rows = []
    for prompt in prompts:
        model.set_classes([prompt])
        results = model.predict(
            source=[str(path) for path in sample_paths],
            device=device,
            imgsz=640,
            conf=0.001,
            max_det=5,
            verbose=False,
        )
        matched_confidences = []
        matched_ious = []
        for result, target in zip(results, ground_truth):
            matches = []
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                iou = box_iou(xyxy, target)
                if iou >= 0.3:
                    matches.append((float(box.conf.item()), iou))
            confidence, iou = max(matches, default=(0.0, 0.0))
            matched_confidences.append(confidence)
            matched_ious.append(iou)
        detection_rate = float(np.mean(np.asarray(matched_confidences) >= args.threshold))
        mean_confidence = float(np.mean(matched_confidences))
        rows.append(
            {
                "prompt": prompt,
                "detection_rate": detection_rate,
                "mean_matched_confidence": mean_confidence,
                "mean_iou": float(np.mean(matched_ious)),
                "score": detection_rate + mean_confidence,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    report = {
        "concept": args.concept,
        "samples": len(sample_paths),
        "threshold": args.threshold,
        "winner": rows[0]["prompt"],
        "ranking": rows,
        "sample_files": [path.name for path in sample_paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
