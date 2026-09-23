import cv2
import os

input_path = "test_far.jpg"
img = cv2.imread(input_path)

os.makedirs("degraded", exist_ok=True)

h, w = img.shape[:2]

# 1. Giảm độ phân giải rất mạnh rồi phóng lại
for scale in [0.5, 0.25, 0.125, 0.0625]:
    small = cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        f"degraded/downscale_{int(scale*100)}.jpg",
        restored
    )

# 2. JPEG compression mạnh
for quality in [50, 20, 10, 5]:
    cv2.imwrite(
        f"degraded/jpeg_q{quality}.jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

# 3. Blur mạnh
for k in [9, 21, 41]:
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    cv2.imwrite(
        f"degraded/blur_{k}.jpg",
        blurred
    )

print("Done")