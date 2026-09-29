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
| `apple_1` / `apple` | 79/100 | 0,889 | 2,31 mm (79 frame) |
| `cereal_box_1` / `cereal box` | 70/100 | 0,708 | 23,40 mm (70 frame) |
| `cereal_box_1` / `box` | 90/100 | 0,706 | 26,27 mm (90 frame) |

Prompt `box` tăng tỷ lệ định vị cho bộ hộp ngũ cốc đơn vật thể, nhưng đây không
phải bằng chứng nó tốt hơn trong cảnh có nhiều hộp. Sai khác tọa độ trong bảng
chỉ so với mask chú giải trên cùng depth map, **không phải độ chính xác tuyệt đối
của camera**.

## Có cần segmentation cho tọa độ Perception?

`perception.compare_regions` chạy **một lần YOLOE trên mỗi frame** rồi so sánh
hai vùng của cùng detection: bounding box và mask pixel segmentation. Không dùng
mask chú giải để chọn detection. Chạy trên 100 frame trải đều mỗi bộ:

```powershell
.\.venv\Scripts\python.exe -m perception.compare_regions --dataset data\rgbd\apple_1 --prompt apple --samples 100 --output outputs\apple\region_comparison_100.json
.\.venv\Scripts\python.exe -m perception.compare_regions --dataset data\rgbd\cereal_box_1 --prompt box --samples 100 --output outputs\cereal_box\region_comparison_100.json
```

| Dataset / vùng | Định vị | Pixel depth hợp lệ ngoài vật theo mask chú giải | Phần vật được vùng chọn bao phủ | Sai khác XYZ trung bình so với mask chú giải |
| --- | ---: | ---: | ---: | ---: |
| Táo / box | 79/100 | 17,2% | 94,9% | 5,81 mm |
| Táo / segmentation | 79/100 | 1,8% | 90,5% | 2,31 mm |
| Hộp / box | 90/100 | 15,5% | 93,2% | 16,20 mm |
| Hộp / segmentation | 90/100 | 8,1% | 74,9% | 26,27 mm |

Độ lệch chuẩn của sai khác `(X,Y,Z)` so với mask chú giải trên cùng các frame
(mm): táo box `(1,17; 4,41; 1,32)`, táo segmentation
`(0,92; 2,29; 1,06)`; hộp box `(3,57; 17,42; 10,32)`, hộp segmentation
`(11,40; 29,19; 40,13)`. Đây là chỉ số nhất quán *so với tham chiếu theo frame*,
không phải mức rung tọa độ của một vật đứng yên.

Segmentation loại bớt nền ở cả hai bộ, nhưng **không luôn cải thiện tọa độ**.
Với táo nó gần tọa độ từ mask chú giải hơn trên 74/79 frame. Với hộp, hai
cách gần như ngang nhau theo số frame (46/90 nghiêng về segmentation), nhưng
vài mask lỗi làm sai khác trung bình của segmentation lớn hơn. Ví dụ frame
`cereal_box_1_4_160_crop.png`: mask chú giải cho `Z=0,592 m`, box cho
`0,601 m`, còn segmentation chỉ phủ khoảng 33% mask chú giải và cho
`Z=0,790 m`. Vì vậy **bounding box đủ làm baseline**, còn segmentation là
tùy chọn cần kiểm chứng theo loại vật/cảnh, không phải yêu cầu bắt buộc.

JSON chứa kết quả từng frame, IoU, độ phủ depth, vùng ngoài mask chú giải,
độ phân tán của sai khác tọa độ theo các frame và thống kê trên **các frame mà
cả hai cách đều định vị được**. Vật trên bàn xoay thay đổi góc nhìn, nên độ
phân tán này chỉ là thước đo tính nhất quán *so với mask chú giải từng frame*,
không phải nhiễu thời gian của camera. Hai cách dùng cùng model segmentation,
vì vậy thử nghiệm này **không so tốc độ** với một model chỉ xuất bounding box.
Các sai khác XYZ đều dùng **cùng ảnh depth** và cùng định nghĩa tọa độ đại diện;
chúng không chứng minh sai số tuyệt đối hay tọa độ tâm vật thật.

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

## Thử RGB-D không cần camera thật bằng Gazebo

Máy phát triển có Gazebo Sim 8 (Harmonic) và ROS 2 Jazzy trong WSL Ubuntu 24.04.
World gọn `sim/perception_demo.sdf` giữ lại một cảm biến RGB-D, khối đỏ từ ví dụ
`sensors_demo.sdf` của Gazebo và một nón dựng bằng hình khối ngay trong world.
Giao diện chỉ hiện ảnh RGB-D
màu và depth; không hiện các bảng thermal, lidar hay camera phụ. World còn có
một segmentation camera **ẩn**, cùng pose/FOV/độ phân giải với RGB-D, gán nhãn
`1` cho nón và `2` cho hộp để đánh giá YOLOE; YOLOE không nhận mask này làm input.
Cảm biến phát RGB,
depth float32 tính bằng mét và `camera_info`. Script capture chờ **RGB và depth
cùng timestamp**, đổi depth sang PNG 16-bit mm, lưu intrinsics từ `camera_info`
để `perception.rgbd` dùng được. Đây là bài kiểm tra **một frame**, chưa phải
pipeline real-time.

Mở ba terminal WSL. Terminal 1 chạy Gazebo với giao diện:

```bash
gz sim -r /mnt/d/yoloe_test/sim/perception_demo.sdf
```

Thêm `-s` nếu chỉ muốn chạy server không mở cửa sổ Gazebo.

Terminal 2 nối ba topic sang ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  '/rgbd_camera/image@sensor_msgs/msg/Image[gz.msgs.Image' \
  '/rgbd_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image' \
  '/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' \
  '/rgbd_camera/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image'
```

Terminal 3 lưu một cặp ảnh đã đồng bộ (đường dẫn ví dụ của máy phát triển):

```bash
source /opt/ros/jazzy/setup.bash
python3 /mnt/d/yoloe_test/sim/capture_rgbd_ros.py \
  --output /mnt/d/yoloe_test/data/rgbd/gazebo_minimal
```

Trong PowerShell ở repo, chạy Perception trên ảnh vừa lấy:

```powershell
.\.venv\Scripts\python.exe -m perception.rgbd --rgb data\rgbd\gazebo_minimal\rgb.png --depth data\rgbd\gazebo_minimal\depth.png --intrinsics data\rgbd\gazebo_minimal\intrinsics.json --prompt "traffic cone" --output outputs\gazebo_minimal
.\.venv\Scripts\python.exe -m sim.check_builtin_box --input data\rgbd\gazebo_minimal
```

Kết quả lần thử với world gọn: ảnh `320×240`, 55,2% pixel có depth hợp lệ; YOLOE nhận
`traffic cone` với confidence `0,656`, tọa độ đại diện trong hệ camera
`(1,016; -0,192; 4,870) m`. YOLOE **không nhận** khối đỏ đơn giản với prompt
`box`, nên `sim.check_builtin_box` tách riêng phép thử hình học bằng màu đỏ.
Trong world mẫu, mặt trước khối đỏ cách camera xấp xỉ `4,450 m` theo trục quang
học; depth thu được cho `4,448 m`, chênh `2 mm` ở frame đó. Đây là kiểm tra
world mô phỏng lý tưởng và đường chuyển đổi dữ liệu, **không phải độ chính xác
của camera vật lý**. Dữ liệu một frame và output được lưu ở `data/rgbd/gazebo_minimal`
và `outputs/gazebo_minimal` (không đẩy lên Git).

### Chạy YOLOE trực tiếp trên Gazebo

Giữ Gazebo và ROS bridge ở hai terminal đầu như trên, rồi trong WSL chạy bộ
chuyển RGB, depth và mask chuẩn **cùng timestamp**. Nó chỉ ghi frame mới nhất (tối đa 5 Hz),
không tích lũy một dataset lớn:

```bash
source /opt/ros/jazzy/setup.bash
python3 /mnt/d/yoloe_test/sim/stream_rgbd_ros.py \
  --output /mnt/d/yoloe_test/data/rgbd/gazebo_live/frame.npz
```

Trong PowerShell ở repo, mở cửa sổ kết quả YOLOE:

```powershell
.\.venv\Scripts\python.exe -m sim.live_yoloe --prompt "traffic cone"
```

Kéo `rgbd_camera` hoặc `traffic_cone` trong Gazebo khi mô phỏng đang **Play**.
Cửa sổ kết quả cập nhật RGB, depth, vùng YOLOE, `(X,Y,Z)` theo hệ camera optical
và `radial_distance_m` (khoảng cách thẳng tới tọa độ đại diện vùng nhìn thấy).
Nó còn hiển thị confidence cao nhất của YOLOE trong frame, IoU với mask chuẩn
Gazebo, sai khác XYZ tính bằng cm và `hit@0.5` / precision trên tối đa 50 frame
gần nhất. Với một mục tiêu, phép đo dùng **khung YOLOE confidence cao nhất**;
không dùng mask chuẩn để chọn khung đẹp hơn. `hit@0.5` là tỷ lệ frame có nón
nhìn thấy mà khung này đạt IoU ≥ 0,5; precision tính mỗi khung YOLOE thừa là false positive. Khi
không thấy vật, frame vẫn được thống kê đúng là miss hoặc false positive.
Nếu Gazebo bị Pause quá 2 giây, cửa sổ hiện cảnh báo thiếu frame. Nhấn `Q`
hoặc `Esc` để đóng YOLOE; `Ctrl+C` dừng bộ chuyển frame. Có thể thay prompt,
chọn `--region box` để so sánh với segmentation, hoặc chạy tự động bằng
`--duration 10 --no-display`. Ảnh/JSON mới nhất ở `outputs/gazebo_live/`.

`live_update_fps` là tốc độ cập nhật toàn luồng, chịu giới hạn bởi `--max-rate`
và thời gian YOLOE; `inference_and_localization_ms` là thời gian xử lý một frame.
Các tọa độ là **so với camera đang di chuyển**, không phải tọa độ world của Gazebo.
Chúng mô tả vùng bề mặt nhìn thấy, không mặc nhiên là tâm thật của vật. Sai khác
XYZ dùng mask chuẩn và YOLOE trên **cùng depth**, nên đo ảnh hưởng của việc chọn
vùng vật, **không phải sai số tuyệt đối của camera**. Những frame lặp lại của cảnh
đứng yên không phải nhiều mẫu độc lập để kết luận độ chính xác tổng quát.
Mask chuẩn chỉ được định nghĩa cho prompt `traffic cone` / `cone` và `box` /
`red box`; prompt khác vẫn chạy YOLOE nhưng báo `Gazebo GT: unavailable`.

## Thử trước bằng webcam RGB 2D

Trên máy phát triển, webcam thật là camera `0` qua backend `msmf` (camera `1`
là OBS Virtual Camera). Đặt chai nước trong khung hình, đủ sáng, rồi chạy từ
PowerShell để mở cửa sổ nhận diện và hiện FPS:

```powershell
.\.venv\Scripts\python.exe -m perception.webcam --text "chai nước" --camera 0 --backend msmf --width 640 --height 480 --mirror --snapshot outputs\webcam_2d\bottle_snapshot.jpg --output outputs\webcam_2d\fps.json
```

Nhấn `Q` hoặc `Esc` để dừng. Tree + Library đổi `chai nước` thành prompt YOLOE
`bottle`; có thể thay bằng `--prompt bottle` để thử YOLOE trực tiếp. JSON ghi
FPS toàn luồng, thời gian suy luận, số frame có detection và confidence cao nhất.
`--snapshot` lưu frame có detection mạnh nhất. Thêm `--duration 10 --no-display`
để chạy benchmark 10 giây không mở cửa sổ. Webcam 2D chỉ kiểm tra detection,
luồng xử lý và FPS; nó không cung cấp phép đo `Z` hay độ chính xác XYZ theo mét.

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
