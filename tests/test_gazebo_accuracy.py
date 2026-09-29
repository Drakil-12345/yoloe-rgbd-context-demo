"""Ground-truth semantics for the live Gazebo demo."""

import unittest

import numpy as np

from perception.geometry import CameraIntrinsics, localize_mask
from sim.accuracy import RollingAccuracy, evaluate_frame


class GazeboAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((12, 12), 2000, dtype=np.uint16)
        self.labels = np.zeros((12, 12), dtype=np.uint8)
        self.labels[3:9, 3:9] = 1
        self.intrinsics = CameraIntrinsics(100, 100, 5.5, 5.5)

    def detection(self, region):
        return {"confidence": 0.42,
                "position": localize_mask(self.depth, region, self.intrinsics,
                                          min_valid_pixels=5)}

    def test_exact_ground_truth_region_has_zero_xyz_error(self):
        region = self.labels == 1
        score = evaluate_frame(self.labels, 1, self.depth, self.intrinsics,
                               [self.detection(region)], [region])
        self.assertEqual(score["status"], "matched")
        self.assertEqual(score["iou"], 1.0)
        self.assertAlmostEqual(score["xyz_error_m"], 0.0)
        self.assertEqual(score["matched_confidence"], 0.42)
        self.assertEqual(score["gt_bbox_xyxy"], [3, 3, 9, 9])
        self.assertEqual(score["gt_frame_edges"], [])
        self.assertEqual(score["gt_depth_coverage"], 1.0)

    def test_visible_target_reports_clipping_and_depth_coverage(self):
        labels = np.zeros_like(self.labels)
        labels[0:3, 9:12] = 1
        depth = self.depth.copy()
        depth[0, 9] = 0
        score = evaluate_frame(labels, 1, depth, self.intrinsics, [], [])
        self.assertEqual(score["status"], "missed")
        self.assertEqual(score["gt_bbox_xyxy"], [9, 0, 12, 3])
        self.assertEqual(score["gt_frame_edges"], ["top", "right"])
        self.assertAlmostEqual(score["gt_depth_coverage"], 8 / 9)

    def test_miss_counts_against_rolling_detection_and_hit_rates(self):
        missed = evaluate_frame(self.labels, 1, self.depth, self.intrinsics, [], [])
        self.assertEqual(missed["status"], "missed")
        self.assertEqual(missed["iou"], 0.0)
        self.assertIsNone(missed["xyz_error_m"])
        region = self.labels == 1
        matched = evaluate_frame(self.labels, 1, self.depth, self.intrinsics,
                                 [self.detection(region)], [region])
        rolling = RollingAccuracy()
        rolling.add(missed)
        stats = rolling.add(matched)
        self.assertEqual(stats["visible_frames"], 2)
        self.assertEqual(stats["detection_rate"], 0.5)
        self.assertEqual(stats["hit_rate_iou_50"], 0.5)

    def test_invisible_target_is_not_an_accuracy_sample(self):
        empty = np.zeros_like(self.labels)
        region = self.labels == 1
        score = evaluate_frame(empty, 1, self.depth, self.intrinsics,
                               [self.detection(region)], [region])
        self.assertEqual(score["status"], "false_positive")
        self.assertIsNone(score["iou"])
        self.assertEqual(RollingAccuracy().add(score)["visible_frames"], 0)

    def test_duplicate_detection_reduces_precision(self):
        region = self.labels == 1
        detection = self.detection(region)
        score = evaluate_frame(self.labels, 1, self.depth, self.intrinsics,
                               [detection, detection], [region, region])
        self.assertEqual(score["false_positive_count"], 1)
        stats = RollingAccuracy().add(score)
        self.assertEqual(stats["hit_rate_iou_50"], 1.0)
        self.assertEqual(stats["precision_iou_50"], 0.5)

    def test_ground_truth_does_not_select_a_lower_confidence_box(self):
        correct = self.labels == 1
        wrong = np.zeros_like(correct)
        wrong[0:3, 0:3] = True
        high = self.detection(wrong)
        high["confidence"] = 0.9
        low = self.detection(correct)
        low["confidence"] = 0.4
        score = evaluate_frame(self.labels, 1, self.depth, self.intrinsics,
                               [high, low], [wrong, correct])
        self.assertEqual(score["best_detection_index"], 0)
        self.assertEqual(score["iou"], 0.0)
        self.assertEqual(score["status"], "low_overlap")


if __name__ == "__main__":
    unittest.main()
