import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLOE

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="water bottle")
parser.add_argument("--conf", type=float, default=0.15)
parser.add_argument("--show", action="store_true", help="Open the detected image in an OpenCV window")
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "input.jpg")
parser.add_argument("--source-url", default="https://unsplash.com/photos/clear-plastic-bottle-on-white-table-edBR3b2JAuA")
parser.add_argument("--output-prefix", default="")
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
model = YOLOE(str(ROOT.parent / "yoloe-v8s-seg.pt"))
model.set_classes([args.prompt])
result = model.predict(
    source=str(args.source),
    device=0 if torch.cuda.is_available() else "cpu",
    imgsz=640,
    conf=args.conf,
    verbose=False,
)[0]
result.save(filename=str(ROOT / (args.output_prefix + "detected.jpg")))
report = {
    "source_url": args.source_url,
    "source_file": str(args.source.resolve()),
    "prompt": args.prompt,
    "model": "yoloe-v8s-seg.pt",
    "confidence_threshold": args.conf,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "detections": json.loads(result.to_json()),
    "speed_ms": result.speed,
}
(ROOT / (args.output_prefix + "results.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({**report, "detections": [
    {"label": result.names[int(box.cls.item())], "confidence": float(box.conf.item()),
     "bbox_xyxy": box.xyxy[0].tolist()} for box in result.boxes
]}, indent=2))

if args.show:
    title = "YOLOE - " + args.prompt
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 640, 800)
    cv2.imshow(title, result.plot())
    print("Press any key in the image window or close it to exit.", flush=True)
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break
    cv2.destroyAllWindows()
