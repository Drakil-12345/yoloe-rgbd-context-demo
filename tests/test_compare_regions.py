"""Check that paired RGB-D evaluation measures the selected pixels fairly."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from perception.compare_regions import region_quality, summarize_paired
from perception.rgbd import box_mask, detection_mask, segmentation_mask


class CompareRegionsTests(unittest.TestCase):
    def test_background_fraction_counts_valid_depth_outside_object(self) -> None:
        reference = np.zeros((4, 4), dtype=bool)
        reference[1:3, 1:3] = True
        box = box_mask(reference.shape, np.array([0, 0, 4, 4]))
        depth = np.full(reference.shape, 1000, dtype=np.uint16)
        depth[0, 0] = 0
        quality = region_quality(box, reference, depth)
        self.assertAlmostEqual(quality["outside_reference_fraction"], 12 / 16)
        self.assertAlmostEqual(quality["valid_depth_outside_reference_fraction"], 11 / 15)
        self.assertAlmostEqual(quality["mask_iou"], 4 / 16)
        self.assertAlmostEqual(quality["reference_coverage_fraction"], 1.0)

    def test_missing_segmentation_is_not_silently_counted_as_box(self) -> None:
        prediction = SimpleNamespace(masks=None)
        xyxy = np.array([1, 1, 3, 3])
        self.assertIsNone(segmentation_mask(prediction, 0, (4, 4)))
        np.testing.assert_array_equal(detection_mask(prediction, 0, (4, 4), xyxy),
                                      box_mask((4, 4), xyxy))

    def test_pixel_mask_keeps_disconnected_regions_without_polygon_fill(self) -> None:
        pixels = torch.zeros((1, 4, 4))
        pixels[0, 0, 0] = 1
        pixels[0, 3, 3] = 1
        prediction = SimpleNamespace(masks=SimpleNamespace(data=pixels))
        actual = segmentation_mask(prediction, 0, (4, 4))
        self.assertEqual(int(np.count_nonzero(actual)), 2)
        self.assertFalse(actual[1, 1])

    def test_paired_winners_use_the_same_frames(self) -> None:
        rows = [
            {"box": {"error_vs_dataset_mask_xyz_m": [0.01, 0, 0]},
             "segmentation": {"error_vs_dataset_mask_xyz_m": [0.02, 0, 0]}},
            {"box": {"error_vs_dataset_mask_xyz_m": [0.03, 0, 0]},
             "segmentation": {"error_vs_dataset_mask_xyz_m": [0.01, 0, 0]}},
            {"box": {"error_vs_dataset_mask_xyz_m": [0.01, 0, 0]},
             "segmentation": None},
        ]
        result = summarize_paired(rows)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["box_lower_disagreement_count"], 1)
        self.assertEqual(result["segmentation_lower_disagreement_count"], 1)
        self.assertEqual(result["equal_disagreement_count"], 0)
        self.assertAlmostEqual(result["box_mean_xyz_disagreement_m"], 0.02)


if __name__ == "__main__":
    unittest.main()
