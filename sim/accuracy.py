"""Per-frame YOLOE agreement with Gazebo's hidden semantic ground truth.

This compares target regions using the same simulated depth image. It measures
detection / region quality, not the physical accuracy of an RGB-D camera.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from perception.geometry import CameraIntrinsics, localize_mask


TARGET_LABELS = {"traffic cone": 1, "cone": 1, "box": 2, "red box": 2}


def mask_iou(predicted: np.ndarray, reference: np.ndarray) -> float:
    if predicted.shape != reference.shape:
        raise ValueError("Masks must have the same shape")
    intersection = np.count_nonzero(predicted & reference)
    union = np.count_nonzero(predicted | reference)
    return float(intersection / union) if union else 0.0


def evaluate_frame(labels: np.ndarray, target_label: int, depth: np.ndarray,
                   intrinsics: CameraIntrinsics, detections: list[dict],
                   regions: list[np.ndarray]) -> dict:
    if labels.shape != depth.shape or len(detections) != len(regions):
        raise ValueError("Ground truth, depth and YOLOE regions must align")
    reference = labels == target_label
    gt_pixels = int(np.count_nonzero(reference))
    if gt_pixels == 0:
        return {"status": "false_positive" if detections else "not_visible",
                "gt_pixels": 0, "iou": None, "xyz_error_m": None,
                "reference_xyz_m": None, "matched_confidence": None,
                "best_detection_index": None, "detection_count": len(detections),
                "false_positive_count": len(detections)}
    reference_position = localize_mask(depth, reference, intrinsics, min_valid_pixels=5)
    reference_xyz = reference_position["xyz_m"]
    if not detections:
        return {"status": "missed", "gt_pixels": gt_pixels, "iou": 0.0,
                "xyz_error_m": None, "reference_xyz_m": reference_xyz,
                "matched_confidence": None, "best_detection_index": None,
                "detection_count": 0, "false_positive_count": 0}
    # The demo tracks one target: evaluate YOLOE's primary (highest-confidence)
    # prediction. Do not use ground truth to cherry-pick a lower-confidence box.
    best = int(np.argmax([item["confidence"] for item in detections]))
    iou = mask_iou(regions[best], reference)
    estimated_xyz = detections[best]["position"]["xyz_m"]
    xyz_error = (float(np.linalg.norm(np.asarray(estimated_xyz) - np.asarray(reference_xyz)))
                 if estimated_xyz is not None and reference_xyz is not None else None)
    return {"status": "matched" if iou >= 0.5 else "low_overlap",
            "gt_pixels": gt_pixels, "iou": iou,
            "xyz_error_m": xyz_error, "reference_xyz_m": reference_xyz,
            "matched_confidence": detections[best]["confidence"],
            "best_detection_index": best, "detection_count": len(detections),
            "false_positive_count": len(detections) - int(iou >= 0.5)}


class RollingAccuracy:
    def __init__(self, window: int = 50):
        if window < 1:
            raise ValueError("Window must be positive")
        self.scores: deque[dict] = deque(maxlen=window)

    def add(self, score: dict) -> dict:
        self.scores.append(score)
        visible = [item for item in self.scores if item["gt_pixels"] > 0]
        hits = sum(item["status"] == "matched" for item in visible)
        total_detections = sum(item["detection_count"] for item in self.scores)
        errors = [item["xyz_error_m"] for item in self.scores
                  if item["xyz_error_m"] is not None]
        return {"frames": len(self.scores), "visible_frames": len(visible),
                "detection_rate": (sum(item["best_detection_index"] is not None
                                       for item in visible) / len(visible)
                                   if visible else None),
                "hit_rate_iou_50": hits / len(visible) if visible else None,
                "precision_iou_50": hits / total_detections if total_detections else None,
                "false_positive_count": sum(item["false_positive_count"]
                                            for item in self.scores),
                "mean_iou": (float(np.mean([item["iou"] for item in visible]))
                             if visible else None),
                "mean_xyz_error_m": float(np.mean(errors)) if errors else None}
