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
Mặc định lấy một vùng có confidence cao nhất cho mục tiêu cần định vị; thêm
`--all-detections` khi cần xem tất cả vùng YOLOE phát hiện.

### Thử thêm một vật thể khác

Tải [`cereal_box_1.tar`](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/cereal_box_1.tar)
và giải nén RGB/depth/mask/loc vào `data/rgbd/cereal_box_1/` (570 frame trên máy
phát triển). Chạy:

```powershell
.\.venv\Scripts\python.exe -m perception.rgbd --dataset data\rgbd\cereal_box_1 --index 200 --prompt box --show
.\.venv\Scripts\python.exe -m perception.evaluate_rgbd --dataset data\rgbd\cereal_box_1 --prompt box --samples 100 --output outputs\cereal_box\evaluation_100_box_prompt.json
```

Kết quả trên 100 frame trải đều mỗi dataset (`--conf 0.15`):

| Dataset / prompt | Định vị thành công | Mask IoU trung bình | Sai khác Euclidean trung bình so với mask chú giải |
| --- | ---: | ---: | ---: |
| `apple_1` / `apple` | 79/100 | 0,887 | 2,56 mm (79 frame) |
| `cereal_box_1` / `cereal box` | 70/100 | 0,738 | 22,05 mm (70 frame) |
| `cereal_box_1` / `box` | 90/100 | 0,734 | 22,64 mm (90 frame) |

Prompt `box` tăng tỷ lệ định vị cho bộ hộp ngũ cốc đơn vật thể, nhưng đây không
phải bằng chứng nó tốt hơn trong cảnh có nhiều hộp. Sai khác tọa độ trong bảng
chỉ so với mask chú giải trên cùng depth map, **không phải độ chính xác tuyệt đối
của camera**.

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

Một ảnh tĩnh chỉ cho biết **camera báo bao nhiêu**, không tự chứng minh kết quả
đúng. Để kiểm tra `Z` trên camera thật, đặt một mặt phẳng có texture ở các mốc
đo độc lập (ví dụ 0,5 / 1 / 1,5 / 2 m tính từ mặt phẳng cảm biến theo trục
quang học), giữ RGB-depth đã căn chỉnh, lấy nhiều frame tại mỗi mốc rồi so sánh
`Z` với khoảng cách chuẩn. Đo tới **cùng mặt phẳng bề mặt** mà mask lấy depth;
đừng so với tâm của vật dày. Báo bias, MAE/RMSE và độ phủ depth theo từng mốc;
`depth_mad_m` chỉ là độ phân tán của pixel depth, không phải sai số so với thực tế.
Nếu đo bằng thước từ camera tới một điểm lệch trục, so với `radial_distance_m`
thay vì `Z`.

Khi có số đo độc lập `D` (m) tới đúng mặt phẳng cần kiểm tra, thêm
`--measured-z-m D` vào lệnh `perception.rgbd`. File `localization.json` sẽ có
`measured_z_check.absolute_error_m` và sai số phần trăm cho **frame đó**. Không
điền chính giá trị mà depth map đã báo làm số đo chuẩn.

### Ảnh RGB-D ở xa hơn

[Washington RGB-D Scenes](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes/)
có cảnh phòng với vật ở nhiều vị trí. Tải `desk_1.tar`, giải nén các file
`desk_1_<số>.png` và `desk_1_<số>_depth.png` vào `data/rgbd/desk_1/`. Mẫu
`desk_1_1` cho laptop `Z=1,173 m`, khoảng cách thẳng `1,251 m`, độ phủ depth
33,5%; xa hơn các mẫu turntable ở khoảng 0,6–0,7 m. Chạy:

```powershell
.\.venv\Scripts\python.exe -m perception.rgbd --rgb data\rgbd\desk_1\desk_1_1.png --depth data\rgbd\desk_1\desk_1_1_depth.png --intrinsics configs\washington_intrinsics.json --prompt laptop --show
```

Đây là số đo do ảnh depth ghi lại, **không phải sai số được kiểm chứng**; dataset
scene này không có phép đo XYZ độc lập của laptop cho phép kết luận độ chính xác.

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
