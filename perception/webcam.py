"""Run Tree + Prompt Library + YOLOE on a live laptop camera with FPS metrics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from .common import LIBRARY_PATH, MODEL_PATH, OUTPUT_DIR
from .image import filter_by_context
from .prompts import PromptOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="chai nước", help="Keyword/utterance received from Speech-to-Text.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--capture-fps", type=float, default=30.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--mirror", action="store_true", help="Mirror before inference so left/right matches the preview.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 runs until Q/Esc.")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV preview window.")
    parser.add_argument("--report-interval", type=float, default=2.0)
    parser.add_argument("--record", type=Path, help="Optional annotated MP4 output path.")
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "webcam_fps.json")
    return parser.parse_args()


def backend_candidates(name: str) -> list[tuple[str, int]]:
    mapping = {
        "dshow": [("DSHOW", cv2.CAP_DSHOW)],
        "msmf": [("MSMF", cv2.CAP_MSMF)],
        "any": [("ANY", cv2.CAP_ANY)],
    }
    if name != "auto":
        return mapping[name]
    if os.name == "nt":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def open_camera(args: argparse.Namespace):
    failures = []
    for backend_name, backend_id in backend_candidates(args.backend):
        capture = cv2.VideoCapture(args.camera, backend_id)
        if not capture.isOpened():
            failures.append(f"{backend_name}: open failed")
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.capture_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture, frame, backend_name
        failures.append(f"{backend_name}: read failed")
        capture.release()
    raise RuntimeError(
        f"Could not read camera index {args.camera}. Tried " + "; ".join(failures)
    )


def draw_hud(
    image: np.ndarray,
    prompt: str,
    fps: float,
    inference_ms: float,
    detection_count: int,
    device_name: str,
) -> None:
    lines = [
        f"Prompt: {prompt}",
        f"FPS end-to-end: {fps:5.1f}",
        f"YOLOE inference: {inference_ms:5.1f} ms ({1000 / max(inference_ms, 1e-6):.1f} FPS)",
        f"Detections: {detection_count} | Device: {device_name}",
        "Q / Esc: quit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.63
    thickness = 2
    line_height = 26
    panel_width = min(image.shape[1], 650)
    panel_height = 15 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    for index, line in enumerate(lines):
        color = (80, 255, 80) if index == 1 else (255, 255, 255)
        cv2.putText(
            image,
            line,
            (12, 25 + index * line_height),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def build_report(
    args: argparse.Namespace,
    resolved,
    prompt: str,
    backend_name: str,
    capture,
    device_name: str,
    frame_durations: list[float],
    inference_times: list[float],
    started_at: float,
) -> dict:
    fps_values = [1.0 / duration for duration in frame_durations if duration > 0]
    elapsed = time.perf_counter() - started_at
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "stt_text": args.text,
        "resolver": asdict(resolved),
        "yoloe_prompt": prompt,
        "model": str(args.model.resolve()),
        "device": device_name,
        "precision": "FP32",
        "mirror": args.mirror,
        "warmup_frames": args.warmup_frames,
        "camera": {
            "index": args.camera,
            "backend": backend_name,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_capture_fps": float(capture.get(cv2.CAP_PROP_FPS)),
        },
        "frames": len(frame_durations),
        "elapsed_seconds": elapsed,
        "fps": {
            "throughput_average": len(frame_durations) / max(sum(frame_durations), 1e-9),
            "median": float(statistics.median(fps_values)) if fps_values else None,
            "p05_low": percentile(fps_values, 5),
            "p95": percentile(fps_values, 95),
        },
        "inference_ms": {
            "mean": float(statistics.fmean(inference_times)) if inference_times else None,
            "median": float(statistics.median(inference_times)) if inference_times else None,
            "p95": percentile(inference_times, 95),
        },
        "measurement_note": "Model warm-up is excluded. End-to-end FPS includes capture, preprocess, inference, postprocess, drawing, and display wait.",
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_timed_frame(
    writer: cv2.VideoWriter,
    frame: np.ndarray,
    previous_frame: np.ndarray | None,
    elapsed_seconds: float,
    output_fps: float,
    frames_written: int,
) -> tuple[np.ndarray, int]:
    """Repeat the previous frame when processing is slower than video playback."""
    target_frames = max(1, int(elapsed_seconds * output_fps) + 1)
    if frames_written >= target_frames:
        return previous_frame if previous_frame is not None else frame, frames_written
    fill_frame = previous_frame if previous_frame is not None else frame
    for _ in range(target_frames - frames_written - 1):
        writer.write(fill_frame)
    writer.write(frame)
    return frame, target_frames


def main() -> None:
    args = parse_args()
    optimizer = PromptOptimizer(args.library)
    resolved = optimizer.resolve(args.text)
    if resolved.concept is None:
        raise SystemExit(f"Unknown object in STT input: {args.text!r}")
    if resolved.ambiguous_with:
        raise SystemExit(f"Ambiguous target: {resolved.concept}, {', '.join(resolved.ambiguous_with)}")

    prompt = resolved.optimized_prompt
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    model = YOLOE(str(args.model))
    model.set_classes([prompt])
    capture, first_frame, backend_name = open_camera(args)

    writer = None
    record_started_at = None
    previous_recorded_frame = None
    recorded_frames = 0
    recording_fps = None
    window_name = "YOLOE webcam | Tree + Prompt Library"
    frame_durations: list[float] = []
    inference_times: list[float] = []
    ema_fps = 0.0
    last_console_report = time.perf_counter()
    started_at = time.perf_counter()

    try:
        # Camera/model warm-up is intentionally excluded from FPS.
        warmup_frame = first_frame
        for warmup_index in range(max(1, args.warmup_frames)):
            if warmup_index:
                ok, next_frame = capture.read()
                if ok and next_frame is not None:
                    warmup_frame = next_frame
            predict_frame = cv2.flip(warmup_frame, 1) if args.mirror else warmup_frame
            model.predict(
                source=predict_frame,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )
        started_at = time.perf_counter()
        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera frame read failed; stopping.", flush=True)
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = model.predict(
                source=frame,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )[0]
            inference_ms = float(result.speed.get("inference", 0.0))
            kept = filter_by_context(result, frame.shape, resolved.spatial, resolved.selection)
            annotated = result[kept].plot(img=frame.copy()) if kept else frame.copy()
            draw_hud(annotated, prompt, ema_fps, inference_ms, len(kept), device_name)

            if args.record:
                if writer is None:
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    recording_fps = capture.get(cv2.CAP_PROP_FPS) or args.capture_fps
                    writer = cv2.VideoWriter(
                        str(args.record),
                        fourcc,
                        recording_fps,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {args.record}")
                    record_started_at = time.perf_counter()
                previous_recorded_frame, recorded_frames = write_timed_frame(
                    writer,
                    annotated,
                    previous_recorded_frame,
                    time.perf_counter() - record_started_at,
                    recording_fps,
                    recorded_frames,
                )

            key = -1
            if not args.no_display:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF

            duration = time.perf_counter() - frame_started
            frame_durations.append(duration)
            inference_times.append(inference_ms)
            instant_fps = 1.0 / max(duration, 1e-9)
            ema_fps = instant_fps if not ema_fps else 0.9 * ema_fps + 0.1 * instant_fps

            now = time.perf_counter()
            if now - last_console_report >= args.report_interval:
                report = build_report(
                    args,
                    resolved,
                    prompt,
                    backend_name,
                    capture,
                    device_name,
                    frame_durations,
                    inference_times,
                    started_at,
                )
                save_report(args.output, report)
                print(
                    f"frames={report['frames']} | FPS={report['fps']['throughput_average']:.1f} "
                    f"| inference={report['inference_ms']['mean']:.1f} ms | detections={len(kept)}",
                    flush=True,
                )
                last_console_report = now

            if key in (ord("q"), ord("Q"), 27):
                break
            if not args.no_display:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
            if args.duration > 0 and now - started_at >= args.duration:
                break
    finally:
        final_report = build_report(
            args,
            resolved,
            prompt,
            backend_name,
            capture,
            device_name,
            frame_durations,
            inference_times,
            started_at,
        )
        if args.record and writer is not None:
            final_report["recording"] = {
                "path": str(args.record.resolve()),
                "encoded_fps": recording_fps,
                "encoded_frames": recorded_frames,
                "duration_seconds": recorded_frames / recording_fps,
            }
        save_report(args.output, final_report)
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
