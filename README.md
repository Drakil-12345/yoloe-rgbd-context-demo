# YOLOE Perception

Pipeline perception nhận câu/keyword từ Speech-to-Text, dùng cây khái niệm và
prompt library để chọn prompt YOLOE. Repo cũng có detection từ ảnh hoặc webcam,
ước lượng khoảng cách từ RGB-D, tạo point cloud và benchmark prompt.

## Cấu trúc

```text
perception/       Mã nguồn: prompts, image, webcam, rgbd, point_cloud, benchmark
tests/            Kiểm thử cho bộ giải ngữ cảnh
data/             Ảnh và RGB-D tải về (không commit)
outputs/          Ảnh, JSON, PLY và video sinh ra (không commit)
```

`yoloe-v8s-seg.pt` và `mobileclip_blt.ts` là model/cache tải về, được giữ ở
thư mục gốc để YOLOE tìm được theo mặc định; cả hai không được commit.

## Cài đặt

Chạy PowerShell tại thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Đặt checkpoint `yoloe-v8s-seg.pt` tại thư mục gốc. Nếu chưa có dataset RGB-D,
tải instance `water_bottle_1` từ
[Washington RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)
và giải nén các file `*_crop.png`, `*_depthcrop.png`, `*_maskcrop.png`, `*_loc.txt`
vào `data/rgbd/water_bottle_1/`.

## Chạy

Mọi lệnh chạy từ thư mục gốc bằng `python -m perception.<module>`. Trên Windows,
thay `python` bằng `.\.venv\Scripts\python.exe` nếu chưa kích hoạt môi trường.

```powershell
# Xem Tree + Library giải câu STT thành prompt nào
python -m perception.prompts tìm chai nước màu xanh bên phải

# Detect trên ảnh RGB; thay bằng đường dẫn ảnh của bạn
python -m perception.image --text "tìm chai nước bên phải" --source "C:\path\to\image.jpg" --show

# Camera laptop, hiển thị FPS; Q hoặc Esc để dừng
python -m perception.webcam --text "tìm chai nước" --camera 0 --mirror

# RGB-D: detect trên RGB và đo khoảng cách từ depth
python -m perception.rgbd --show

# Tạo point cloud PLY có màu và hiện ảnh preview ba góc nhìn
python -m perception.point_cloud --index 23 --show

# So sánh các prompt bằng 12 frame RGB-D có mask
python -m perception.benchmark --samples 12

# Kiểm thử bộ giải ngữ cảnh
python -m unittest discover -s tests -v
```

Các lệnh nhận `--help` để xem tùy chọn như `--model`, `--dataset`, `--index`,
`--output` và `--duration`. Mặc định ảnh/JSON/PLY được ghi vào `outputs/`.
Trên máy thử nghiệm, camera 640×480 đạt khoảng 30 FPS, còn 1280×720 đạt khoảng
10 FPS dù inference YOLOE chỉ mất khoảng 18–21 ms/frame. Khi ưu tiên tốc độ,
dùng `--width 640 --height 480`; thêm `--record outputs/camera.mp4` để lưu video
theo thời gian thực.

## Tree + Prompt Library

Ví dụ `tìm chai nước to nhất ở bên phải` được chuyển thành:

```text
concept: water_bottle
YOLOE prompt: bottle
spatial: right
selection: largest
fallback: water bottle, plastic bottle
```

`perception/prompt_library.json` lưu ontology, từ đồng nghĩa Việt/Anh, các biến
thể STT và thứ tự prompt. Bộ giải chuẩn hóa dấu và dấu câu, ưu tiên alias khớp
nguyên từ, dùng fuzzy match khi STT sai nhẹ, rồi tách màu/vị trí/kích thước khỏi
prompt YOLOE. Nếu prompt đầu không có detection, pipeline thử prompt kế tiếp.
Vị trí và kích thước được lọc sau khi detect. Có thể thêm object bằng cách sửa
library mà không phải sửa mã inference.

Màu hiện được parse nhưng chưa dùng để lọc box. Nhiều target trong một câu được
báo là ambiguous; tham chiếu hội thoại như `nó` chưa có session state. Benchmark
trước đây trên 12 frame `water_bottle_1` cho thấy prompt `bottle` đạt detection
rate 100% ở ngưỡng 0.15 với checkpoint đang dùng.

## RGB-D và point cloud

YOLOE xử lý ảnh RGB ba kênh. Sau detection/segmentation, chương trình lấy median
depth trong vùng vật thể để báo khoảng cách mét. Point cloud dùng intrinsics
Kinect của dataset (`fx = fy = 570.3`, `cx = 320`, `cy = 240`) và crop offset trong
`loc.txt`; tọa độ PLY tính bằng mét (`+X` phải, `+Y` xuống, `+Z` về trước).
Mặc định point cloud bỏ nền và depth lệch quá 120 mm so với median vật thể;
`--all-pixels` hoặc `--depth-outlier-mm 0` thay đổi bộ lọc này.

Nguồn dữ liệu: Kevin Lai, Liefeng Bo, Xiaofeng Ren, Dieter Fox,
"A Large-Scale Hierarchical Multi-View RGB-D Object Dataset," ICRA 2011.
