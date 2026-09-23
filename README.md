# YOLOE Context, RGB-D and Webcam Demo

Prototype perception pipeline for YOLOE with deterministic context understanding,
RGB-D distance estimation, point-cloud generation, and live webcam FPS testing.
The project maps Vietnamese/English Speech-to-Text output to optimized YOLOE
prompts using an ontology tree and a versioned prompt library, without an LLM.

## Features

- Tree + Prompt Library resolver for Vietnamese, unaccented Vietnamese, and English.
- Whole-token aliases, fuzzy ASR recovery, clause negation, colors, spatial rules,
  and largest/smallest selection.
- Empirical prompt benchmarking on labeled RGB-D frames.
- YOLOE image detection with ordered prompt fallbacks.
- Live laptop-camera detection with end-to-end FPS and inference latency.
- RGB-D detection with metric object depth.
- Colored PLY point-cloud generation from Washington RGB-D crops.

## Project layout

```text
context_prompt_demo/   Tree, prompt library, tests, benchmark, image and webcam runners
rgbd_demo/             RGB-D detection and colored point-cloud scripts
```

## Setup

Create a Python environment and install the runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the YOLOE checkpoint at the repository root:

```text
yoloe-v8s-seg.pt
```

The checkpoint, MobileCLIP cache, virtual environment, and downloaded RGB-D
dataset are intentionally excluded from Git because they are large generated or
third-party artifacts.

## Test the prompt resolver

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s context_prompt_demo -p "test_*.py" -v

.\.venv\Scripts\python.exe context_prompt_demo\prompt_optimizer.py `
  tìm chai nước màu xanh bên phải
```

## Run YOLOE on an image

```powershell
.\.venv\Scripts\python.exe context_prompt_demo\context_yoloe.py `
  --text "tìm chai nước to nhất ở bên phải" `
  --source "C:\path\to\your\image.jpg" `
  --show
```

Replace `--source` with the path to an RGB image on your machine.

## Run webcam detection and FPS measurement

```powershell
.\.venv\Scripts\python.exe context_prompt_demo\webcam_yoloe.py `
  --text "tìm chai nước" `
  --camera 0 `
  --mirror
```

Press `Q` or `Esc` to stop. Use `--duration 30` for a fixed benchmark.
On the development laptop (GTX 1650, 1280x720 camera), the measured result was
about 30 FPS end-to-end with roughly 16 ms YOLOE inference latency.

## Prompt benchmark

```powershell
.\.venv\Scripts\python.exe context_prompt_demo\benchmark_prompts.py --samples 12
```

For the included water-bottle experiment, `bottle` reached a 100% detection rate
on 12 sampled frames at confidence 0.15, while the more specific prompts did not.

## RGB-D and point cloud

Download the `water_bottle_1` cropped archive from the
[Washington RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)
and extract it under:

```text
rgbd_demo/dataset/rgbd-dataset/water_bottle/water_bottle_1/
```

Then run:

```powershell
.\.venv\Scripts\python.exe rgbd_demo\rgbd_yoloe.py --show
.\.venv\Scripts\python.exe rgbd_demo\create_point_cloud.py
```

More details are available in [context_prompt_demo/DESIGN.md](context_prompt_demo/DESIGN.md)
and [rgbd_demo/README.md](rgbd_demo/README.md).
