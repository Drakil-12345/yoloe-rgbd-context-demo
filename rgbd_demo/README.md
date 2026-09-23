# YOLOE on an RGB-D dataset

This demo uses the University of Washington RGB-D Object Dataset, instance
`water_bottle_1`. The cropped archive contains 561 aligned RGB/depth pairs plus
object masks. Depth PNG values are unsigned 16-bit millimeters; zero means that
depth is unavailable.

YOLOE remains a three-channel RGB model. The script detects/segments the object
from RGB, then samples the aligned depth image inside the predicted segmentation
mask and reports the median object distance. This avoids pretending that the
pretrained checkpoint accepts a fourth input channel.

Run the tested frame and show the result window:

```powershell
.\.venv\Scripts\python.exe rgbd_demo\rgbd_yoloe.py --show
```

Select another pair with `--index 0` through `--index 560`. The result image and
JSON report are written to `rgbd_demo/results/`.

Create an object-masked colored point cloud (`.ply`) and a three-view preview:

```powershell
.\.venv\Scripts\python.exe rgbd_demo\create_point_cloud.py
```

Add `--all-pixels` to include the crop background. The projection uses the
dataset authors' one-indexed Kinect constants (`fx = fy = 570.3`, `cx = 320`, `cy = 240`)
and the per-frame `loc.txt` crop offset. PLY coordinates are in meters with
`+X` right, `+Y` down, and `+Z` forward.
The default object cloud also removes masked depth values more than 120 mm from
the median, which suppresses Kinect background leakage through transparent
parts of the bottle. Pass `--depth-outlier-mm 0` to keep every valid value.

Dataset: https://rgbd-dataset.cs.washington.edu/dataset/

Citation: Kevin Lai, Liefeng Bo, Xiaofeng Ren, and Dieter Fox, "A Large-Scale
Hierarchical Multi-View RGB-D Object Dataset," ICRA 2011.
