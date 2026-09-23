from ultralytics import YOLOE
import torch
import cv2
import os

# ============================
# 1. Load model
# ============================

device = 0 if torch.cuda.is_available() else "cpu"

print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model = YOLOE("yoloe-v8s-seg.pt")

image_path = "test.jpg"

# ============================
# 2. Nhập text
# ============================

prompt = input("\nNhap vat the muon tim (English): ").strip()

print("\nText prompt:", prompt)

# Đây chính là bước đưa TEXT vào YOLOE
model.set_classes([prompt])

# ============================
# 3. Image + Text -> YOLOE
# ============================

results = model.predict(
    source=image_path,
    device=device,
    imgsz=640,
    conf=0.15,
    verbose=False
)

result = results[0]

# ============================
# 4. In kết quả
# ============================

print("\n===== RESULT =====")

if len(result.boxes) == 0:
    print("Khong tim thay:", prompt)

else:
    print("So vat the:", len(result.boxes))

    for i, box in enumerate(result.boxes):

        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        bbox = box.xyxy[0].tolist()

        print(
            f"{i + 1}. "
            f"Object = {result.names[cls_id]}, "
            f"Confidence = {confidence:.3f}, "
            f"BBox = {[round(x, 1) for x in bbox]}"
        )

# ============================
# 5. Mask
# ============================

if result.masks is not None:
    print("Segmentation masks:", len(result.masks.data))
else:
    print("Khong co segmentation mask")

# ============================
# 6. Save ảnh kết quả
# ============================

os.makedirs("text_results", exist_ok=True)

safe_name = prompt.replace(" ", "_").replace("/", "_")

output_path = f"text_results/{safe_name}.jpg"

annotated = result.plot()

cv2.imwrite(output_path, annotated)

print("\nSaved:", output_path)

# ============================
# 7. Speed
# ============================

print("\nInference speed:")
print(result.speed)