# Tree + Prompt Library for YOLOE

This prototype converts raw Speech-to-Text output into a stable YOLOE prompt
without an LLM.

Pipeline:

1. Normalize Vietnamese accents, punctuation, and common STT formatting.
2. Resolve the longest object alias against an ontology tree.
3. Use fuzzy matching only as a fallback for likely ASR errors.
4. Select the empirically ranked YOLOE base prompt from `prompt_library.json`.
5. Keep color, spatial, and size words as structured constraints instead of
   polluting the detection prompt.
6. Run YOLOE and apply spatial/size constraints to the resulting boxes.
7. Try lower-ranked prompts only if the preferred prompt detects nothing.

Inspect only the resolver:

```powershell
.\.venv\Scripts\python.exe context_prompt_demo\prompt_optimizer.py tìm chai nước màu xanh bên phải
```

Run the complete pipeline:

```powershell
.\.venv\Scripts\python.exe context_prompt_demo\context_yoloe.py `
  --text "tìm chai nước bên phải" `
  --source water_bottle_demo\complex_input.jpg `
  --show
```

Benchmark every stored prompt against labeled RGB-D frames:

```powershell
.\.venv\Scripts\python.exe context_prompt_demo\benchmark_prompts.py --samples 12
```

Run live detection on the laptop camera with end-to-end FPS and model inference
FPS displayed in the window:

```powershell
.\.venv\Scripts\python.exe context_prompt_demo\webcam_yoloe.py `
  --text "tìm chai nước" `
  --camera 0
```

Press `Q` or `Esc` to stop. The final measurement is saved to
`context_prompt_demo/results/webcam_fps.json`. Use `--duration 30` for a fixed
benchmark, `--mirror` for a selfie-style view with correct left/right semantics,
or `--record output.mp4` to save the annotated camera stream.

Add vocabulary by editing `prompt_library.json`; application code does not need
to change. A production Speech-to-Text service can call
`PromptOptimizer.resolve(text)` directly.
