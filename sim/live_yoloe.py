"""Detect and localize objects from the live Gazebo RGB-D stream on Windows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from perception.common import MODEL_PATH, OUTPUT_DIR
from perception.geometry import CameraIntrinsics, localize_mask
from perception.rgbd import box_mask, detection_mask, render_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/rgbd/gazebo_live/frame.npz"))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "gazebo_live")
    parser.add_argument("--prompt", default="traffic cone")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--region", choices=("segmentation", "box"), default="segmentation")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames.")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def read_frame(path: Path) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as frame:
        rgb = frame["bgr"]
        depth = frame["depth_mm"]
        fx, fy, cx, cy = frame["intrinsics"].tolist()
        stamp = tuple(int(value) for value in frame["stamp"])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2] or depth.dtype != np.uint16:
        raise ValueError("Expected aligned BGR and uint16 millimetre depth arrays")
    return rgb, depth, CameraIntrinsics(fx, fy, cx, cy), stamp


def localize_detections(result, depth: np.ndarray, intrinsics: CameraIntrinsics,
                        region_mode: str) -> list[dict]:
    detections = []
    for index, box in enumerate(result.boxes):
        bbox = box.xyxy[0].cpu().numpy()
        region = (box_mask(depth.shape, bbox) if region_mode == "box"
                  else detection_mask(result, index, depth.shape, bbox))
        position = localize_mask(depth, region, intrinsics, min_valid_pixels=5)
        detections.append({"class": result.names[int(box.cls.item())],
                           "confidence": float(box.conf.item()),
                           "bbox_xyxy": [float(value) for value in bbox],
                           "region": region_mode, "position": position})
    return detections


def main() -> None:
    args = parse_args()
    if args.conf <= 0 or args.conf >= 1 or args.imgsz <= 0:
        raise SystemExit("--conf must be in (0,1) and --imgsz must be positive")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    model = YOLOE(str(args.model))
    model.set_classes([args.prompt])
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    window = "Gazebo RGB-D + YOLOE"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    started = time.monotonic()
    last_mtime = 0
    last_frame_wall = started
    previous_frame_wall: float | None = None
    display: np.ndarray | None = None
    stale_reported = False
    count = 0
    print(f"YOLOE prompt={args.prompt!r}, device={device}; waiting for {args.input}", flush=True)
    print("Coordinates use camera optical frame: +X right, +Y down, +Z forward.", flush=True)
    try:
        while args.duration is None or time.monotonic() - started < args.duration:
            if args.input.exists():
                mtime = args.input.stat().st_mtime_ns
                if mtime != last_mtime:
                    try:
                        rgb, depth, intrinsics, stamp = read_frame(args.input)
                    except (OSError, ValueError, KeyError):
                        time.sleep(0.05)
                        continue
                    last_mtime = mtime
                    frame_start = time.monotonic()
                    result = model.predict(rgb, device=device, imgsz=args.imgsz,
                                           conf=args.conf, retina_masks=True, verbose=False)[0]
                    detections = localize_detections(result, depth, intrinsics, args.region)
                    elapsed = time.monotonic() - frame_start
                    pipeline_fps = (1 / (frame_start - previous_frame_wall)
                                    if previous_frame_wall is not None and
                                    frame_start > previous_frame_wall else None)
                    previous_frame_wall = frame_start
                    last_frame_wall = time.monotonic()
                    stale_reported = False
                    report = {"gazebo_stamp": {"sec": stamp[0], "nanosec": stamp[1]},
                              "prompt": args.prompt, "region": args.region,
                              "coordinate_frame": "camera_optical: +X right, +Y down, +Z forward",
                              "inference_and_localization_ms": round(elapsed * 1000, 1),
                              "live_update_fps": round(pipeline_fps, 2) if pipeline_fps else None,
                              "detections": detections}
                    display = render_result(rgb, depth, detections, (0.0, 0.0), height=480)
                    rate_label = f"{pipeline_fps:.1f} FPS" if pipeline_fps else "warming up"
                    cv2.putText(display, f"live {rate_label} | infer {elapsed * 1000:.0f} ms",
                                (display.shape[1] - 340, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(str(args.output / "latest.jpg"), display)
                    (args.output / "latest.json").write_text(
                        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    count += 1
                    if detections:
                        for item in detections:
                            pos = item["position"]
                            print(f"frame={count} {item['class']} conf={item['confidence']:.2f} "
                                  f"XYZ={pos['xyz_m']} distance={pos['radial_distance_m']} m",
                                  flush=True)
                    else:
                        print(f"frame={count} no {args.prompt!r} detection", flush=True)
                    if not args.no_display:
                        cv2.imshow(window, display)
                    if args.max_frames is not None and count >= args.max_frames:
                        break
            if not stale_reported and time.monotonic() - last_frame_wall > 2.0:
                print("No new RGB-D frame for 2 s; check Gazebo Play and ROS bridge.", flush=True)
                stale_reported = True
                if display is not None and not args.no_display:
                    stale = display.copy()
                    cv2.rectangle(stale, (0, stale.shape[0] - 38),
                                  (stale.shape[1], stale.shape[0]), (0, 0, 160), -1)
                    cv2.putText(stale, "NO NEW FRAME - check Gazebo Play / ROS bridge",
                                (12, stale.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window, stale)
            if not args.no_display:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
            else:
                time.sleep(0.05)
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {count} RGB-D frames", flush=True)


if __name__ == "__main__":
    main()
