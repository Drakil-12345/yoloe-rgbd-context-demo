"""Check camera coordinates, crop offsets, and depth failure modes."""

import unittest
import tempfile
from pathlib import Path

import numpy as np

from perception.geometry import CameraIntrinsics, WASHINGTON_INTRINSICS, localize_mask
from perception.evaluate_rgbd import mask_iou, read_ground_truth, summarize_errors


class GeometryTests(unittest.TestCase):
    def test_washington_crop_projection_matches_official_formula(self) -> None:
        depth = np.full((2, 2), 1000, dtype=np.uint16)
        mask = np.ones_like(depth)
        result = localize_mask(depth, mask, WASHINGTON_INTRINSICS,
                               crop_origin=(274, 113), min_valid_pixels=1)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["xyz_m"][0], (274.5 - 319) / 570.3)
        self.assertAlmostEqual(result["xyz_m"][1], (113.5 - 239) / 570.3)
        self.assertAlmostEqual(result["xyz_m"][2], 1.0)

    def test_missing_depth_returns_explicit_failure(self) -> None:
        result = localize_mask(np.zeros((5, 5), dtype=np.uint16),
                               np.ones((5, 5), dtype=np.uint8), WASHINGTON_INTRINSICS)
        self.assertEqual(result["status"], "insufficient_depth")
        self.assertIsNone(result["xyz_m"])
        self.assertEqual(result["depth_coverage"], 0.0)

    def test_depth_outlier_does_not_move_surface_point(self) -> None:
        depth = np.full((5, 5), 1000, dtype=np.uint16)
        depth[0, 0] = 3000
        intrinsics = CameraIntrinsics(100, 100, 2, 2)
        result = localize_mask(depth, np.ones((5, 5), dtype=np.uint8), intrinsics,
                               depth_outlier_m=0.12, min_valid_pixels=1)
        self.assertEqual(result["inlier_pixels"], 24)
        self.assertEqual(result["xyz_m"][2], 1.0)

    def test_evaluation_metrics_are_named_reference_errors(self) -> None:
        self.assertAlmostEqual(mask_iou(np.array([[1, 0], [0, 0]], bool),
                                        np.array([[1, 1], [0, 0]], bool)), 0.5)
        summary = summarize_errors([{"error": [0.01, 0.0, -0.02]}], "error")
        self.assertAlmostEqual(summary["mean_euclidean_error_m"], np.sqrt(0.0005))

    def test_independent_ground_truth_csv_is_parsed_by_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measured.csv"
            path.write_text("frame,x_m,y_m,z_m\nframe_1_crop.png,0.1,-0.2,0.8\n", encoding="utf-8")
            rows = read_ground_truth(path)
        np.testing.assert_allclose(rows["frame_1_crop.png"], [0.1, -0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
