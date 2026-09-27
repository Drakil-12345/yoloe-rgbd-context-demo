"""Save one synchronized Gazebo RGB-D frame for perception.rgbd.

Run with ROS 2 Jazzy sourced after bridging Gazebo's image, depth_image and
camera_info topics to ROS. The depth topic is expected to be 32FC1 metres.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def stamp_key(message: Image) -> tuple[int, int]:
    return message.header.stamp.sec, message.header.stamp.nanosec


class Capture(Node):
    def __init__(self, output: Path, topic_prefix: str):
        super().__init__("gazebo_rgbd_capture")
        self.output = output
        self.bridge = CvBridge()
        self.images: dict[tuple[int, int], Image] = {}
        self.depths: dict[tuple[int, int], Image] = {}
        self.camera_info: CameraInfo | None = None
        self.done = False
        self.create_subscription(Image, f"{topic_prefix}/image", self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, f"{topic_prefix}/depth_image", self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, f"{topic_prefix}/camera_info", self.on_info,
                                 qos_profile_sensor_data)

    def on_image(self, message: Image) -> None:
        self.images[stamp_key(message)] = message
        self.try_save()

    def on_depth(self, message: Image) -> None:
        self.depths[stamp_key(message)] = message
        self.try_save()

    def on_info(self, message: CameraInfo) -> None:
        self.camera_info = message
        self.try_save()

    def try_save(self) -> None:
        if self.done or self.camera_info is None:
            return
        common = self.images.keys() & self.depths.keys()
        if not common:
            # Keep memory bounded while waiting for matching timestamps.
            self.images = dict(list(self.images.items())[-20:])
            self.depths = dict(list(self.depths.items())[-20:])
            return
        stamp = max(common)
        rgb_msg, depth_msg = self.images[stamp], self.depths[stamp]
        if depth_msg.encoding.upper() != "32FC1":
            raise ValueError(f"Expected 32FC1 Gazebo depth in metres, got {depth_msg.encoding}")
        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if bgr.shape[:2] != depth_m.shape:
            raise ValueError("RGB and depth are not aligned to the same image size")
        valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < 65.535)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(depth_m[valid] * 1000).astype(np.uint16)
        k = self.camera_info.k
        intrinsics = {"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
                      "depth_scale_m": 0.001}
        if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
            raise ValueError("CameraInfo has invalid focal lengths")
        self.output.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.output / "rgb.png"), bgr):
            raise OSError("Could not save RGB image")
        if not cv2.imwrite(str(self.output / "depth.png"), depth_mm):
            raise OSError("Could not save depth image")
        (self.output / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
        details = {"timestamp": {"sec": stamp[0], "nanosec": stamp[1]},
                   "rgb_encoding": rgb_msg.encoding, "depth_encoding": depth_msg.encoding,
                   "width": depth_mm.shape[1], "height": depth_mm.shape[0],
                   "valid_depth_fraction": float(np.mean(valid)),
                   "depth_min_m": float(np.min(depth_m[valid])) if np.any(valid) else None,
                   "depth_max_m": float(np.max(depth_m[valid])) if np.any(valid) else None}
        (self.output / "capture.json").write_text(json.dumps(details, indent=2))
        print(json.dumps(details, indent=2), flush=True)
        print(f"Saved RGB-D pair in {self.output.resolve()}", flush=True)
        self.done = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic-prefix", default="/rgbd_camera")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    rclpy.init()
    node = Capture(args.output, args.topic_prefix.rstrip("/"))
    try:
        deadline = time.monotonic() + args.timeout
        while not node.done and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.done:
            raise TimeoutError("No matched RGB/depth timestamps plus CameraInfo received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
