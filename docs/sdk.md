**English** · [繁體中文](zh-TW/sdk.md)

# SDK and ONNX Runtime

Since v0.14 the detection logic can be embedded without the web demo. `presence_demo.sdk.PresenceEngine` takes camera
frames (and optionally audio) and returns presence state and events, and every neural model runs on
[ONNX Runtime](https://onnxruntime.ai) through one small layer, `presence_demo.inference`.

## Install

```bash
pip install presence-demo                 # SDK: numpy, OpenCV (image ops), onnxruntime, psutil
pip install "presence-demo[web]"          # + the FastAPI web demo (presence-demo command)
pip install "presence-demo[litert]"       # + LiteRT: MoveNet TFLite fallback, SFace TFLite / NPU delegate backend
uv sync                                   # development: SDK + web + litert + test tools
```

For a GPU or NPU build, install the matching ONNX Runtime package instead of `onnxruntime` (for example
`onnxruntime-gpu`, `onnxruntime-directml`, `onnxruntime-qnn`, `onnxruntime-openvino`). They provide the same
`onnxruntime` module, so nothing else changes.

## Use

```python
from presence_demo.sdk import PresenceEngine

engine = PresenceEngine()  # YuNet + SFace on ONNX Runtime, downloaded on first use
while running:
    frame = camera.read()  # BGR uint8, unmirrored
    step = engine.process_frame(frame, speaking=my_vad())
    for ev in step.events:  # PresenceEvent(kind, reason, extra)
        if ev.kind == "transition":
            print(ev.extra["frm"], "->", ev.extra["to"], ev.reason)
    state = step.snapshot["state"]  # ACTIVE | IDLE | MAYBE_AWAY | AWAY | ENDED
```

| Method | Purpose |
|---|---|
| `process_frame(img, speaking=None)` / `process_jpeg(bytes, ...)` | One camera frame; returns a `Step` (events + snapshot) |
| `set_speaking(bool)` | Your VAD changed between frames |
| `tick()` | Advance timers when no input arrives (about every 0.5 s) |
| `process_audio(pcm16)` | PureSound isolation + clean VAD + vision cross-check (needs the `puresound` extra and `audio_pipeline="puresound"`); returns cleaned PCM16 and a `Step` when speech changed |
| `configure({...})` | Change settings (same flat keys and validation as the web settings); returns warnings |
| `calibrate_distance(m)`, `reset()`, `info()`, `close()` | Distance calibration, new session, diagnostics, release audio state |

Notes:

- The engine has no threads of its own; calls are serialized by an internal lock, so capture and audio callbacks
  may call from different threads. One engine is one camera/microphone stream. Engines in the same process share
  the model sessions (ONNX Runtime `run` is thread-safe) but no state.
- Time comes from `clock` (default `time.monotonic`). For replays or tests, pass a virtual clock:
  `PresenceEngine(clock=lambda: t)`.
- `PresenceEngine(pipeline=VisionPipeline(detector=..., embedder=...))` plugs in your own detector or embedder
  (see the protocols in `presence_demo.detectors`); `pose=False` disables the pose model, `pose=my_detector`
  replaces it.
- Importing `presence_demo.sdk` loads no web framework, starts no thread and reads no model (checked by
  `tests/test_sdk.py` and `tests/test_layering.py`).

## ONNX Runtime options

```python
from presence_demo.inference import OrtOptions

engine = PresenceEngine(ort=OrtOptions(providers=("coreml", "cpu"), intra_op_threads=4))
```

| Field | Default | Meaning |
|---|---|---|
| `providers` | `("cpu",)` | Execution providers by alias (`cpu`, `cuda`, `tensorrt`, `rocm`, `directml`, `coreml`, `nnapi`, `qnn`, `openvino`, `xnnpack`, `webgpu`) or ORT name. `auto` = first available GPU provider (CUDA, ROCm, DirectML), else CPU. Unavailable providers are skipped; the CPU is always the last fallback. |
| `provider_options` | `{}` | Per-provider options, e.g. `{"cuda": {"device_id": 1}}` |
| `intra_op_threads` | `0` | Threads per model (0 = ORT decides) |
| `graph_optimization` | `"all"` | `disable`, `basic`, `extended` or `all` |
| `sanitize` | `True` | Strip graph inputs that duplicate weights (see below) |

`OrtOptions.from_env()` (the default) reads `PRESENCE_ORT_PROVIDERS` (e.g. `coreml,cpu`), `PRESENCE_ORT_THREADS`
and `PRESENCE_ORT_GRAPH_OPT`; the web demo uses the same variables. NPU-style providers (CoreML, NNAPI, QNN,
OpenVINO) are used only when named, because operator coverage differs per model: on an Apple M-series Mac, SFace
ran about 3× slower on CoreML than on the CPU. Measure before switching. The PureSound audio model always uses the
CPU provider (10 ms frames gain nothing from an accelerator).

**Graph sanitising.** SFace (fp32 and int8) lists every weight as a graph input as well as an initializer. ONNX
Runtime treats such weights as overridable, so it cannot fold constants or fuse Conv + BatchNorm. `load_model`
removes those duplicate inputs in memory before creating the session (no file is written; the result is
byte-identical to ONNX Runtime's own `remove_initializer_from_input.py`). SFace runs about 1.7× faster this way.

**Sessions.** `load_model(path, options)` caches one session per (file, options) for the whole process, so every
engine and every web connection shares it. `loaded_models()` lists them; `info()` shows the providers actually in
use, load time and any fallback note (for example a GPU provider that failed and fell back to the CPU).

## Models

All model files are described in one catalogue, `presence_demo.inference.CATALOG`, and stored by `ModelStore`:
`ModelStore(root)`, else `$PRESENCE_MODEL_DIR`, else `models/` of a source checkout, else
`~/.cache/presence-demo/models` (or `$XDG_CACHE_HOME`). Downloads are verified with sha256 and written atomically.

| Key | File | Runtime | How to get it |
|---|---|---|---|
| `yunet` | `yunet_s_dynamic.onnx` | ONNX Runtime (default face detector) | downloaded on first use |
| `sface` | `face_recognition_sface_2021dec.onnx` | ONNX Runtime (default embedder) | downloaded on first use |
| `sface-int8` | `face_recognition_sface_2021dec_int8.onnx` | ONNX Runtime (`embed_backend = int8`) | downloaded when selected |
| `movenet` | `movenet_multipose_lightning.onnx` | ONNX Runtime (pose) | `scripts/export_movenet_onnx.py` (one-off, separate env with TensorFlow) |
| `puresound` | `dpcrn_curriculum_v2.onnx` + `.json` | ONNX Runtime (PureSound SDK) | `scripts/fetch_puresound_model.py --clone` |
| `yunet-opencv` | `face_detection_yunet_2023mar.onnx` | OpenCV DNN fallback | used only if already present |
| `movenet-tflite` | `movenet_multipose_lightning_fp16.tflite` | LiteRT fallback (`litert` extra) | downloaded on first use of pose |
| `sface-tflite-*` | `face_recognition_sface_2021dec_{int8,fp16}.tflite` | LiteRT (`embed_backend = npu`) | `scripts/export_sface_tflite.py` |

`yunet_s_dynamic.onnx` is the same YuNet network and weights as opencv_zoo's `face_detection_yunet_2023mar.onnx`
(that file is byte-identical to `yunet_s_640_640.onnx` in the upstream training repository), exported with a dynamic
input size, which ONNX Runtime needs because frames and ROI crops come in any size. The ONNX implementation
reproduces OpenCV `FaceDetectorYN`'s padding, decoding and NMS: rows match within 1e-3 px
(`tests/test_inference.py`).

MoveNet has no official ONNX release. The export script converts the TFLite file with tf2onnx (same weights,
dynamic input) and refuses the result unless it matches LiteRT within 1e-4 on several input sizes (measured:
3e-6). When the ONNX file is absent, pose falls back to LiteRT. For a shipped product, export once and bundle the
file with your application.

## Porting checklist

1. Install the ONNX Runtime package for the target (CPU, GPU or NPU build) and set `providers`.
2. Bundle the model files (`ModelStore.status()` lists what is present) and point `PRESENCE_MODEL_DIR` or
   `ModelStore(root)` at them; use `allow_download=False` on devices without network access.
3. Feed `process_frame` with unmirrored BGR frames at 0.5–2 fps (`presence_demo.optimizer.recommend_fps` gives
   the adaptive rate the web demo uses) and your VAD, or 16 kHz PCM16 to `process_audio`.
4. Read `step.events` / `step.snapshot`; `engine.info()` shows which provider each model actually runs on.

## Measured cost (Apple M-series, CPU, onnxruntime 1.31, OpenCV 4.14)

| Stage | OpenCV DNN / LiteRT (v0.13) | ONNX Runtime (v0.14) |
|---|---|---|
| YuNet, 640×480 | 4.6 ms | 2.7 ms |
| SFace fp32, one face | 5.4 ms | 3.0 ms |
| SFace int8, one face | 5.6 ms | 3.8 ms |
| MoveNet MultiPose | 3.2 ms (LiteRT XNNPACK) | 5.2 ms |

MoveNet is slower on ONNX Runtime's CPU provider than on LiteRT's XNNPACK kernels on this machine; at the
0.5–2 fps the demo uses, the difference is about 2 ms per pose call. Install the `litert` extra and remove the
ONNX file if that matters on your target.
