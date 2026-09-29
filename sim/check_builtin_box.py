"""Check the known red box front plane in the Gazebo perception demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from perception.geometry import CameraIntrinsics, localize_mask
from perception.rgbd import load_rgbd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory created by sim.capture_rgbd_ros.")
    parser.add_argument("--expected-z-m", type=float, default=4.45,
                        help="Front-plane optical depth for sim/perception_demo.sdf.")
    args = parser.parse_args()
    if not np.isfinite(args.expected_z_m) or args.expected_z_m <= 0:
        raise ValueError("--expected-z-m must be positive and finite")
    rgb, depth = load_rgbd(args.input / "rgb.png", args.input / "depth.png")
    intrinsics = CameraIntrinsics.from_json(args.input / "intrinsics.json")
    channels = rgb.astype(np.int16)
    # In the demo world the box is red and the other models are not.
    region = (channels[:, :, 2] > 100) & (channels[:, :, 2] > 2 * channels[:, :, 1]) & (
        channels[:, :, 2] > 2 * channels[:, :, 0])
    position = localize_mask(depth, region, intrinsics)
    if position["xyz_m"] is None:
        raise ValueError("No red box surface with enough valid depth was found")
    z = position["xyz_m"][2]
    report = {
        "reference_definition": "front plane of the red box in perception_demo.sdf",
        "expected_z_m": args.expected_z_m,
        "estimated_z_m": z,
        "signed_difference_m": z - args.expected_z_m,
        "absolute_difference_m": abs(z - args.expected_z_m),
        "position": position,
        "note": "Checks one idealized simulator frame, not physical camera accuracy.",
    }
    cv2.imwrite(str(args.input / "red_box_mask.png"), region.astype(np.uint8) * 255)
    (args.input / "box_plane_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
