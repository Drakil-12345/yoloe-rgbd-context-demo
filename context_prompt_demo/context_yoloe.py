"""Resolve an STT utterance with Tree + Library rules, then run YOLOE."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLOE

from prompt_optimizer import PromptOptimizer


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Raw keyword/utterance from Speech-to-Text.")
    parser.add_argument("--source", type=Path, default=here.parent / "water_bottle_demo" / "complex_input.jpg")
    parser.add_argument("--library", type=Path, default=here / "prompt_library.json")
    parser.add_argument("--model", type=Path, default=here.parent / "yoloe-v8s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=here / "results")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def filter_by_context(result, image_shape, spatial: list[str], selection: str | None) -> list[int]:
    height, width = image_shape[:2]
    kept = []
    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        conditions = {
            "left": center_x < 0.5,
            "right": center_x >= 0.5,
            "center": 1 / 3 <= center_x <= 2 / 3,
            "top": center_y < 0.5,
            "bottom": center_y >= 0.5,
        }
        if all(conditions.get(relation, True) for relation in spatial):
            kept.append(index)

    if selection in {"largest", "smallest"} and kept:
        areas = {
            index: float(
                (result.boxes[index].xyxy[0, 2] - result.boxes[index].xyxy[0, 0])
                * (result.boxes[index].xyxy[0, 3] - result.boxes[index].xyxy[0, 1])
            )
            for index in kept
        }
        chooser = max if selection == "largest" else min
        kept = [chooser(areas, key=areas.get)]
    return kept


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    optimizer = PromptOptimizer(args.library)
    resolved = optimizer.resolve(args.text)
    if resolved.concept is None:
        raise SystemExit(f"Unknown object in STT input: {args.text!r}")
    if resolved.ambiguous_with:
        choices = ", ".join([resolved.concept, *resolved.ambiguous_with])
        raise SystemExit(f"Ambiguous STT input; multiple targets found: {choices}")

    source = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(args.source)
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLOE(str(args.model))

    prompts = [resolved.optimized_prompt, *resolved.fallback_prompts]
    prompt_trials = []
    chosen_prompt = prompts[0]
    result = None
    for prompt in prompts:
        model.set_classes([prompt])
        candidate = model.predict(
            source=source,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )[0]
        peak_confidence = max((float(box.conf.item()) for box in candidate.boxes), default=0.0)
        prompt_trials.append(
            {"prompt": prompt, "detections": len(candidate.boxes), "peak_confidence": peak_confidence}
        )
        result = candidate
        chosen_prompt = prompt
        if len(candidate.boxes):
            break

    assert result is not None
    kept_indices = filter_by_context(result, source.shape, resolved.spatial, resolved.selection)
    detections = []
    for index in kept_indices:
        box = result.boxes[index]
        detections.append(
            {
                "concept": resolved.concept,
                "prompt": chosen_prompt,
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()],
            }
        )

    if kept_indices:
        annotated = result[kept_indices].plot()
    else:
        annotated = source.copy()
        cv2.putText(
            annotated,
            "No detection matched the context constraints",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    summary = f"{resolved.concept} -> '{chosen_prompt}'"
    if resolved.spatial:
        summary += " | " + ",".join(resolved.spatial)
    annotated = cv2.copyMakeBorder(annotated, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(annotated, summary, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    image_path = args.output / "context_detection.jpg"
    report_path = args.output / "context_detection.json"
    cv2.imwrite(str(image_path), annotated)
    report = {
        "resolver": asdict(resolved),
        "chosen_prompt": chosen_prompt,
        "prompt_trials": prompt_trials,
        "source": str(args.source.resolve()),
        "confidence_threshold": args.conf,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "raw_detections": len(result.boxes),
        "context_filtered_detections": len(detections),
        "detections": detections,
        "output_image": str(image_path.resolve()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if args.show:
        window = "Tree + Library -> YOLOE"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, annotated)
        print("Press any key in the image window or close it to exit.", flush=True)
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
