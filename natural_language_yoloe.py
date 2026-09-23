import re
import os
import cv2
import torch
from ultralytics import YOLOE


# ============================================================
# 1. DICTIONARY: cách người dùng có thể gọi từng vật
# ============================================================

OBJECT_ALIASES = {
    # Bottle
    "water bottle": "water bottle",
    "bottle": "water bottle",
    "chai nước": "water bottle",
    "chai nuoc": "water bottle",
    "chai": "water bottle",

    # Cup
    "cup": "cup",
    "mug": "cup",
    "cốc": "cup",
    "coc": "cup",
    "ly": "cup",

    # Chair
    "chair": "chair",
    "ghế": "chair",
    "ghe": "chair",

    # Laptop
    "laptop": "laptop",
    "máy tính xách tay": "laptop",
    "may tinh xach tay": "laptop",

    # Phone
    "phone": "cell phone",
    "smartphone": "cell phone",
    "cell phone": "cell phone",
    "điện thoại": "cell phone",
    "dien thoai": "cell phone",

    # Remote
    "remote": "remote control",
    "remote control": "remote control",
    "điều khiển": "remote control",
    "dieu khien": "remote control",
}


# ============================================================
# 2. MÀU SẮC
# ============================================================

COLOR_ALIASES = {
    "blue": "blue",
    "xanh dương": "blue",
    "xanh duong": "blue",

    "red": "red",
    "đỏ": "red",
    "do": "red",

    "green": "green",
    "xanh lá": "green",
    "xanh la": "green",

    "yellow": "yellow",
    "vàng": "yellow",
    "vang": "yellow",

    "black": "black",
    "đen": "black",
    "den": "black",

    "white": "white",
    "trắng": "white",
    "trang": "white",
}


# ============================================================
# 3. HÀM XỬ LÝ TEXT
# ============================================================

def normalize_text(text):
    """
    Chuẩn hóa câu người dùng.
    """
    text = text.lower().strip()

    # bỏ một số punctuation
    text = re.sub(r"[,.!?;:]", " ", text)

    # bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)

    return text


def find_object(text):
    """
    Tìm object trong câu.

    Ưu tiên cụm dài hơn:
    'water bottle' trước 'bottle'
    """
    aliases = sorted(
        OBJECT_ALIASES.keys(),
        key=len,
        reverse=True
    )

    for alias in aliases:
        if alias in text:
            return OBJECT_ALIASES[alias]

    return None


def find_color(text):
    """
    Tìm màu sắc nếu có.
    """
    aliases = sorted(
        COLOR_ALIASES.keys(),
        key=len,
        reverse=True
    )

    for alias in aliases:
        if alias in text:
            return COLOR_ALIASES[alias]

    return None


def extract_target(user_command):
    """
    Natural language -> YOLOE prompt
    """

    text = normalize_text(user_command)

    obj = find_object(text)
    color = find_color(text)

    if obj is None:
        return None

    if color is not None:
        target = f"{color} {obj}"
    else:
        target = obj

    return target


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NATURAL LANGUAGE -> YOLOE")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("GPU not available -> using CPU")

    # --------------------------------------------------------
    # Load YOLOE
    # --------------------------------------------------------

    print("\nLoading YOLOE...")

    model = YOLOE("yoloe-v8s-seg.pt")

    print("YOLOE loaded successfully.")

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_path = "test.jpg"

    if not os.path.exists(image_path):
        print("\nERROR: Không tìm thấy", image_path)
        return

    os.makedirs("natural_language_results", exist_ok=True)

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    while True:

        print("\n" + "=" * 60)

        command = input(
            "\nNhap yeu cau cua ban\n"
            "(vi du: Find the blue water bottle)\n"
            "Nhap 'quit' de thoat:\n> "
        )

        if command.lower().strip() in ["quit", "exit", "q"]:
            print("Bye!")
            break

        # ----------------------------------------------------
        # Natural Language -> target
        # ----------------------------------------------------

        target = extract_target(command)

        print("\nUser command:")
        print("   ", command)

        if target is None:
            print("\n[Parser]")
            print("Khong xac dinh duoc vat the.")
            continue

        print("\n[Parser]")
        print("Target object:")
        print("   ", target)

        # ----------------------------------------------------
        # Target -> YOLOE text prompt
        # ----------------------------------------------------

        print("\n[YOLOE]")
        print("Text prompt:")
        print("   ", target)

        model.set_classes([target])

        # ----------------------------------------------------
        # YOLOE inference
        # ----------------------------------------------------

        results = model.predict(
            source=image_path,
            device=device,
            imgsz=640,
            conf=0.15,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print("\n[Detection Result]")

        if len(result.boxes) == 0:

            print("Khong tim thay:", target)

        else:

            print("So vat the tim thay:", len(result.boxes))

            for i, box in enumerate(result.boxes):

                confidence = float(box.conf.item())

                bbox = box.xyxy[0].tolist()

                print(
                    f"{i + 1}. "
                    f"{target} | "
                    f"confidence = {confidence:.3f} | "
                    f"bbox = {[round(v, 1) for v in bbox]}"
                )

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        if result.masks is not None:
            print(
                "Segmentation masks:",
                len(result.masks.data)
            )
        else:
            print("No segmentation mask")

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        annotated = result.plot()

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            target
        )

        output_path = os.path.join(
            "natural_language_results",
            safe_name + ".jpg"
        )

        cv2.imwrite(
            output_path,
            annotated
        )

        print("\nSaved:")
        print(output_path)

        # ----------------------------------------------------
        # Show image
        # ----------------------------------------------------

        cv2.imshow(
            f"Target: {target}",
            annotated
        )

        print("\nNhan phim bat ky tren cua so anh de tiep tuc.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()