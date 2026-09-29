"""Write the newest synchronized Gazebo RGB-D pair for live YOLOE inference.

Run in WSL with ROS 2 sourced and ros_gz_bridge forwarding RGB, depth,
camera_info, and the simulator-only segmentation labels. Each .npz is replaced
atomically, so the Windows consumer
never reads half of a frame. Intermediate frames are intentionally dropped.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def stamp_key(message: Image) -> tuple[int, int]:
    return message.header.stamp.sec, message.header.stamp.nanosec


class Stream(Node):
    def __init__(self, output: Path, topic_prefix: str, max_rate: float):
        super().__init__("gazebo_rgbd_stream")
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = output.with_name(output.stem + ".tmp.npz")
        self.bridge = CvBridge()
        self.images: dict[tuple[int, int], Image] = {}
        self.depths: dict[tuple[int, int], Image] = {}
        self.labels: dict[tuple[int, int], Image] = {}
        self.camera_info: CameraInfo | None = None
        self.min_interval = 1.0 / max_rate
        self.last_write = 0.0
        self.frames_written = 0
        self.create_subscription(Image, f"{topic_prefix}/image", self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, f"{topic_prefix}/depth_image", self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, f"{topic_prefix}/camera_info", self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, f"{topic_prefix}/segmentation/labels_map",
                                 self.on_labels, qos_profile_sensor_data)

    def on_image(self, message: Image) -> None:
        self.images[stamp_key(message)] = message
        self.try_write()

    def on_depth(self, message: Image) -> None:
        self.depths[stamp_key(message)] = message
        self.try_write()

    def on_info(self, message: CameraInfo) -> None:
        self.camera_info = message
        self.try_write()

    def on_labels(self, message: Image) -> None:
        self.labels[stamp_key(message)] = message
        self.try_write()

    def try_write(self) -> None:
        if self.camera_info is None:
            return
        common = self.images.keys() & self.depths.keys() & self.labels.keys()
        if not common:
            self.images = dict(list(self.images.items())[-8:])
            self.depths = dict(list(self.depths.items())[-8:])
            self.labels = dict(list(self.labels.items())[-8:])
            return
        stamp = max(common)
        rgb_msg, depth_msg = self.images[stamp], self.depths[stamp]
        label_msg = self.labels[stamp]
        self.images.clear()
        self.depths.clear()
        self.labels.clear()
        now = time.monotonic()
        if now - self.last_write < self.min_interval:
            return
        if depth_msg.encoding.upper() != "32FC1":
            raise ValueError(f"Expected 32FC1 depth, got {depth_msg.encoding}")
        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        labels_rgb = self.bridge.imgmsg_to_cv2(label_msg, desired_encoding="passthrough")
        if (bgr.shape[:2] != depth_m.shape or labels_rgb.shape != bgr.shape or
                label_msg.encoding.lower() != "rgb8"):
            raise ValueError("RGB, depth and semantic labels must be aligned")
        if not np.array_equal(labels_rgb[:, :, 0], labels_rgb[:, :, 1]) or not np.array_equal(
                labels_rgb[:, :, 0], labels_rgb[:, :, 2]):
            raise ValueError("Expected equal RGB channels in semantic label map")
        labels = labels_rgb[:, :, 0].copy()
        k = self.camera_info.k
        if k[0] <= 0 or k[4] <= 0:
            raise ValueError("CameraInfo has invalid intrinsics")
        valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < 65.535)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(depth_m[valid] * 1000).astype(np.uint16)
        np.savez_compressed(self.temporary, bgr=bgr, depth_mm=depth_mm, labels=labels,
                            intrinsics=np.array([k[0], k[4], k[2], k[5]], dtype=np.float64),
                            stamp=np.array(stamp, dtype=np.int64))
        os.replace(self.temporary, self.output)
        self.last_write = now
        self.frames_written += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic-prefix", default="/rgbd_camera")
    parser.add_argument("--max-rate", type=float, default=5.0,
                        help="Maximum frame writes per second; inference reads only the latest.")
    args = parser.parse_args()
    if args.max_rate <= 0:
        parser.error("--max-rate must be positive")
    rclpy.init()
    node = Stream(args.output, args.topic_prefix.rstrip("/"), args.max_rate)
    print(f"Streaming RGB-D to {args.output.resolve()} (Ctrl+C to stop)", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Wrote {node.frames_written} synchronized frames", flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
