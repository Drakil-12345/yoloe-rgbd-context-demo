from ultralytics import YOLOE
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load YOLOE model
model = YOLOE("yoloe-v8s-seg.pt")

# Text prompts
classes = [
    "water bottle",
    "cup",
    "chair",
    "laptop"
]

model.set_classes(classes)

# Run inference
results = model.predict(
    source="test.jpg",
    device=0 if torch.cuda.is_available() else "cpu",
    imgsz=640,
    conf=0.20,
    save=True
)

result = results[0]

print("\n===== DETECTIONS =====")

for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    xyxy = box.xyxy[0].tolist()

    print(
        f"Class: {result.names[class_id]}, "
        f"Confidence: {confidence:.3f}, "
        f"BBox: {xyxy}"
    )

if result.masks is not None:
    print("\nNumber of segmentation masks:", len(result.masks.data))
else:
    print("\nNo segmentation mask detected.")

print("\nSpeed:")
print(result.speed)