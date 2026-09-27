# RGB-D Object Localization

Mục tiêu chính của dự án là **xác định tọa độ 3D của một vật thể trong hệ tọa độ
camera RGB-D** và đánh giá các yếu tố ảnh hưởng đến kết quả. YOLOE tìm vùng vật
thể trên RGB; depth đã căn chỉnh và thông số hiệu chuẩn camera biến vùng đó
thành tọa độ mét `(X, Y, Z)`. Point cloud chỉ là công cụ xem dữ liệu phụ trợ.

## Tọa độ được báo là điểm nào?

`perception.rgbd` trả về một **tọa độ đại diện cho vùng vật thể nhìn thấy**: tia đi qua trung
bình tọa độ pixel có depth hợp lệ trong mask, lấy `Z` là median depth của các
pixel đó. Cách này ổn định hơn lấy depth của đúng một pixel; tọa độ nhận được
**không nhất thiết là một điểm thật trên bề mặt**, không phải tâm thể tích của
quả táo/hộp hay tọa độ trong phòng. Hệ camera optical: `+X` sang
phải, `+Y` xuống dưới, `+Z` hướng ra trước. Khoảng cách thẳng từ camera tới điểm
là `radial_distance_m`; `Z` chỉ là khoảng cách theo trục quang học.

Với ảnh RGB/depth đã căn chỉnh và intrinsics tương ứng:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_raw * depth_scale_m
```

Ảnh crop Washington dùng `loc.txt` để đưa `(u,v)` về hệ pixel ảnh gốc trước
khi chiếu 3D. Mã dùng intrinsics zero-indexed `fx=fy=570.3, cx=319, cy=239`,
tương đương công thức one-indexed `(320,240)` của
[depthToCloud.m](https://rgbd-dataset.cs.washington.edu/software/depthToCloud.m).

## Cài đặt và dữ liệu

Chạy PowerShell trong thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Đặt checkpoint `yoloe-v8s-seg.pt` ở thư mục gốc. Dữ liệu, trọng số model và
output được `.gitignore` loại khỏi Git. Bộ dữ liệu thử là
[Washington RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/),
chỉ dùng cho nghiên cứu/giáo dục phi thương mại. Tải bản crop
[`apple_1.tar`](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/apple_1.tar)
và giải nén các file `*_crop.png`, `*_depthcrop.png`, `*_maskcrop.png`, `*_loc.txt`
vào `data/rgbd/apple_1/`. Trên máy phát triển đã có sẵn 607 frame táo.

## Xác định tọa độ trên dataset

```powershell
# Frame 200, dùng YOLOE tìm quả táo; thêm --show để hiện RGB, depth và dấu cộng tọa độ
.\.venv\Scripts\python.exe -m perception.rgbd --dataset data\rgbd\apple_1 --index 200 --prompt apple --show

# Cùng frame, dùng mask chú giải để tính tọa độ tham chiếu trên chính depth map đó
.\.venv\Scripts\python.exe -m perception.rgbd --dataset data\rgbd\apple_1 --index 200 --prompt apple --mask-source dataset --output outputs\apple\mask_reference
```

Lệnh đầu ghi `outputs/rgbd/localization.jpg` và `localization.json`.
Trong JSON, `detections[].position.xyz_m` là `[X,Y,Z]`; `depth_coverage`
là tỷ lệ pixel mask có depth hợp lệ; `depth_mad_m` mô tả độ phân tán depth
trong vùng sau lọc. Nếu depth không đủ, `status` báo lỗi và `xyz_m` là `null`.

## Đánh giá độ chính xác

```powershell
.\.venv\Scripts\python.exe -m perception.evaluate_rgbd --dataset data\rgbd\apple_1 --prompt apple --samples 20 --output outputs\apple\evaluation_20.json
```

Báo cáo có detection/localization rate, mask IoU, độ phủ depth và sai khác XYZ
giữa vùng YOLOE với mask chú giải. Hai kết quả dùng **cùng depth map**, vì vậy
`mask_reference_agreement` đo ảnh hưởng của phân vùng vật thể; nó **không đo sai
số tuyệt đối của camera**. Những frame YOLOE bỏ sót được tính vào localization
rate, không được lặng lẽ bỏ khỏi mẫu.

Muốn đo sai số tuyệt đối, cần một điểm mục tiêu được định nghĩa rõ và có thể đo
độc lập trong hệ camera (ví dụ tâm marker trên vật chuẩn có kích thước/hình học
đã biết). Thuật toán định vị phải xuất tọa độ của **cùng điểm đó**. CSV tham chiếu
có các cột `frame,x_m,y_m,z_m`, ví dụ:

```csv
frame,x_m,y_m,z_m
apple_1_1_201_crop.png,-0.007,0.012,0.704
```

Sau đó thêm `--ground-truth-csv đường_dẫn.csv`. Các số ví dụ trên chỉ mô tả
định dạng, **không phải ground truth**. Với vật không có điểm đánh dấu/định nghĩa
rõ ràng, không thể gọi tâm vật thể hoặc median bề mặt là ground truth. Khi đo
camera thật, cần ghi nhiều khoảng cách/vị trí, kiểm tra hiệu chuẩn và RGB-depth
alignment, báo MAE/RMSE XYZ, Euclidean error, depth fill rate và tỷ lệ định vị.
[Tài liệu Intel về depth quality](https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_DepthQualityTesting.pdf)
cũng phân biệt accuracy so với khoảng cách chuẩn, fill rate và nhiễu.

## Dùng cặp ảnh từ camera RGB-D khác

Lưu một ảnh RGB và một ảnh depth **đã căn chỉnh**, cùng kích thước. Depth phải
là PNG 16-bit một kênh. Tạo JSON intrinsics của **hệ ảnh đã căn chỉnh**; nếu
depth lưu bằng mm thì `depth_scale_m=0.001`:

```json
{"fx": 600.0, "fy": 600.0, "cx": 319.5, "cy": 239.5, "depth_scale_m": 0.001}
```

Các số trên chỉ là ví dụ; phải thay bằng calibration của camera đang dùng.

```powershell
.\.venv\Scripts\python.exe -m perception.rgbd --rgb data\my_camera\rgb.png --depth data\my_camera\depth.png --intrinsics data\my_camera\intrinsics.json --prompt apple --show
```

Nếu dùng ảnh crop, truyền thêm `--crop-origin X Y` (tọa độ góc trên trái của
crop trong ảnh gốc, zero-indexed). Với ảnh full frame không cần tùy chọn này.
Webcam RGB của laptop không cung cấp depth nên không thể dùng để đo XYZ.

## Công cụ phụ trợ

```powershell
# Xem point cloud từ mask chú giải của frame táo
.\.venv\Scripts\python.exe -m perception.point_cloud --dataset data\rgbd\apple_1 --index 200 --show

# Thử Tree + Library cho câu Speech-to-Text
.\.venv\Scripts\python.exe -m perception.prompts tìm chai nước bên phải

# Kiểm thử
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`perception/` chứa code định vị, hình học, đánh giá và các demo còn lại;
`tests/` chứa kiểm thử. `data/` giữ input tải về, `outputs/` giữ ảnh và JSON.

Nguồn dataset: Kevin Lai, Liefeng Bo, Xiaofeng Ren, Dieter Fox,
"A Large-Scale Hierarchical Multi-View RGB-D Object Dataset," ICRA 2011.
