**English** · [繁體中文](zh-TW/performance.md)

# Performance

The target is a small ARM device, so the pipeline is built to do as little as possible. All numbers below were
measured on an 8-core x86 test server (OpenCV DNN on the CPU). Re-measure on your own hardware.

## Four power savers

| Saver | What it does | Parameters |
|---|---|---|
| **Motion gate** | Shrinks the frame to 64×48 grey and compares it with the previous one. Below the threshold and in a stable state, detection is skipped and the last result reused. A full detection still runs every `force_detect_seconds`, and always in MAYBE_AWAY / AWAY. | `motion_gate`, `motion_threshold`, `force_detect_seconds` |
| **Embedding throttle** | Reuses a face's embedding unless the face is new, moved / resized by more than `embed_move_thresh`, or `embed_interval` has passed. | `embed_throttle`, `embed_interval`, `embed_move_thresh` |
| **Adaptive fps** | The server suggests a frame rate: `fps_low` when stable, `fps_high` when faces appear, participants vanish, tracks just changed, or in MAYBE_AWAY / AWAY. | `adaptive_fps`, `fps_low`, `fps_high` |
| **ROI crop** | Crops the ROI plus `crop_margin`, scales it to at most `max_width`, detects, maps boxes back. | `roi_crop`, `crop_margin`, `max_width` |

### Benchmark

`uv run python scripts/benchmark.py` simulates 60 s at 1 fps: a person standing still (0–15 s), moving (15–30 s),
an empty ROI (30–45 s), the person back (45–60 s). Frames are synthesized and JPEG-coded in memory; no image file
is written. Best of 3 runs.

| Setting | ms / frame avg | p95 | Total CPU | vs. all off |
|---|---|---|---|---|
| All savers off | 21.2 | 27.2 | 6.27 s | — |
| ROI crop only | 16.4 | 23.0 | 4.43 s | −29% |
| Motion gate only | 10.1 | 25.9 | 2.94 s | −53% |
| Embedding throttle only | 14.4 | 25.5 | 4.50 s | −28% |
| All on, fixed fps | 7.9 | 22.1 | 2.06 s | −67% |
| All on + adaptive fps | 6.9 | 18.2 | 1.81 s | **−71%** |
| Pose, savers off | 31.1 | 37.9 | 7.93 s | +27% |
| All on + pose | 13.7 | 31.9 | 2.97 s | −53% |
| All on + pose + distance | 14.0 | 32.4 | 3.09 s | −51% |

Every setting still detected AWAY correctly. The p95 barely drops: frames that need a full detection cost the same,
the savings come from frames that need nothing.

## Where the time goes

Per-stage latency, 2 fps, a moving face at 640×480, all savers on (p50 ms):

| Stage | legacy + rules | kalman + hmm + score |
|---|---|---|
| JPEG decode | 0.7 | 0.6 |
| Motion gate | 0.2 | 0.2 |
| Face detection (YuNet, cropped) | 2.4 | 2.4 |
| **Embedding (SFace)** | **9.9** | **10.0** |
| Pose (MoveNet, when on) | 6.3 | 6.1 |
| Clothing histogram | — | 0.1 |
| Tracker + presence + intent | < 0.1 | 0.3 |
| **Total** (no pose / pose) | 13.7 / 19.9 | 14.0 / 20.2 |

- The face embedding is the biggest cost; then pose; then detection.
- The smarter layers (Kalman, HMM, intent) together cost less than 0.5 ms.
- 1280×720 input adds about 1 ms (only decoding gets slower, thanks to the ROI crop).
- Empty or static scenes finish most frames within 1 ms.
- WebSocket end-to-end p50: about 17–18 ms without pose, about 25 ms with pose.

`max_width` matters for small faces: at 320 px, faces under about 80 px tall were missed, so the default is 640
(detection ≈ 2.4 → 5 ms). Faces should be at least 40–50 px wide after downscaling.

## Embedding backends and NPU

`embed_backend` selects how SFace is computed. The model is the same; only the runtime changes.

| Backend | Model | Runtime | Use |
|---|---|---|---|
| `fp32` (default) | opencv_zoo SFace ONNX | ONNX Runtime (v0.13: OpenCV DNN) | Reference, most accurate |
| `int8` | opencv_zoo official INT8 ONNX | ONNX Runtime (v0.13: OpenCV DNN) | 4× smaller; often faster on ARM CPUs |
| `npu` | TFLite exported by this project (`int8` or `fp16`) | LiteRT, optional external delegate | Android NPU / GPU; XNNPACK on a server |

Since v0.14 `fp32` and `int8` run on ONNX Runtime with the execution providers from `PRESENCE_ORT_PROVIDERS`;
per-stage numbers for both runtimes are in [SDK and ONNX Runtime](sdk.md#measured-cost-apple-m-series-cpu-onnxruntime-131-opencv-414).
The table below is the v0.13 measurement (OpenCV DNN, x86 server).

Whole pipeline with an embedding on every frame (p50 / p95 ms, 640×480):

| Backend | Embed | Total |
|---|---|---|
| fp32, OpenCV DNN | 10.5 / 12.7 | 14.7 / 17.2 |
| int8, OpenCV DNN | 12.2 / 14.6 | 16.6 / 19.4 |
| npu fp16, LiteRT XNNPACK | 3.6 / 4.1 | 7.7 / 8.4 |

Accuracy against fp32 (`scripts/eval_embed_backends.py`: 31 CC0 portraits, 6 augmentations each, 651 pairs,
threshold 0.40):

| Backend | Mean cosine vs fp32 | Same decision as fp32 |
|---|---|---|
| int8 ONNX | 0.972 | 99.85% |
| npu int8 TFLite | 0.991 | 100% |
| npu fp16 TFLite | 1.000 | 100% |

So `cos_match_int8` defaults to the same 0.40. This compares quantized models with fp32; it is not a recognition
accuracy benchmark.

### Getting the TFLite models

They are not in git (10 and 19 MB). Export them in a separate environment:

```bash
uv venv -p 3.12 /tmp/.venv-sface-export
VIRTUAL_ENV=/tmp/.venv-sface-export uv pip install onnx2tf tensorflow tf-keras onnx onnxruntime \
    onnx-graphsurgeon onnxsim sng4onnx psutil ai-edge-litert opencv-python-headless
/tmp/.venv-sface-export/bin/python scripts/export_sface_tflite.py
```

The script works on a copy of the ONNX file, rewrites PReLU so it becomes a single `PRELU` op (NPU-friendly),
writes an fp16 and a full-INT8 model (calibrated on CC0 portraits in memory) and checks both against fp32.

### Using a delegate

- On the server, set `PRESENCE_NPU_DELEGATE=/path/to/delegate.so`. For security it can only be set through the
  environment, never over the WebSocket: loading a shared library runs native code.
- If the delegate fails to load, LiteRT falls back to the CPU. The resource panel shows the backend actually in
  use and why.
- Without a delegate, `npu_precision = int8` automatically uses the fp16 model, which is about ten times faster on
  an x86 CPU.
- On Android, use the same `.tflite` in the app with the NNAPI or a vendor delegate. Input: int8
  `[1, 112, 112, 3]`, RGB − 128, aligned with the same 5-point similarity transform. Measure with TensorFlow's
  `benchmark_model` (`--use_nnapi`, `--external_delegate_path`, `--enable_op_profiling`) to see which ops really
  run on the NPU. None of this has been validated on a device yet.

## Audio cost

The PureSound pipeline costs about 5 ms per 20 ms chunk on one core (real-time factor ≈ 0.25), far more than vision
at 1–2 fps. Measure on the target device first. See [Audio](audio.md#performance).

## Live monitoring and round-one measurements

Adaptive FPS recommendations now travel in every `state` message and in
`stats.conn.optim.recommended_fps`. The browser accepts both and only replaces its
capture timer when the effective rate changes. Previously it read only `state`,
while the recommendation was sent only in `stats`, so it stayed at the fixed FPS.

The resource panel separates vision queue wait (receipt to worker start), executor
wait (submission to thread start), inference lock wait, and model stages. These
are separate stages: queue wait precedes the existing `total` measurement, which
starts in the worker. Audio retains cumulative `rtf` and adds the last five
wall-clock seconds' duration-weighted `rtf_recent`, complete-block processing p95,
and receipt-to-processing queue wait avg/p95. RTF measures elapsed processing time
divided by audio duration; it is not a measurement of CPU utilization.

Client capture, JPEG encode, and state/audio/stats render timings retain at most
200 samples each. DOM/canvas work counters and WebSocket queued bytes are also
shown. Render timings cover synchronous callbacks, not subsequent browser layout
or compositing. Audio metadata renders at most 5 Hz while recording and timeline
data still consume every message. Canvas backing stores resize only when needed;
offscreen plots skip drawing, and spectrogram scrolling avoids pixel readback.

[Raw measurements](benchmarks/20261022-round1.json) from 2026-10-22:

| Same input workload | Before | After |
|---|---:|---:|
| Browser audio-panel renders, 100 metadata messages over 10 s | 100 | 51 |
| DOM `innerHTML` writes in that replay | 190 | 66 |
| Canvas width/height assignments | 658 | 4 |
| Spectrogram pixel readbacks | 206 | 0 |
| Measured synchronous rendering work, including deferred audio renders | 222.3 ms | 92.1 ms |
| Frames sent in 16 s of stable blank video + streaming PCM | 16 | 9 |

The browser replay uses stable state with changing audio levels, synthetic data,
and the actual pre/post-fix message schemas; it does not open camera/microphone or
call cloud services. Counts include a final 250 ms drain. The server replay uses
the same SDK/ORT 1.31.0 on both sides, with pose enabled and cloud services off.
All three server runs had zero audio drops. Audio RTF was 0.213 before and
0.218/0.353 in two after runs; process CPU also varied. These short shared-host
measurements do **not** establish an RTF, CPU, or per-frame latency improvement.

Reproduce with the project environment (include `--extra puresound` when syncing):

```bash
python scripts/benchmark_frontend.py --before-ref b093837aa5b53fcfc35706e1cf088556b9c69ebb --output /tmp/frontend.json
# Open the printed URL, click Run before / after, and keep it foreground until complete.
python scripts/benchmark_realtime.py --output /tmp/realtime.json
```

## Round two: independent pose cadence and binary JPEG

`pose_interval` defaults to 2 s (range 0–2; 0 disables throttling). Stable faces
can reuse their matched bodies until this deadline, independently of face
detection. Significant image motion, changed/missing faces, body-only detections,
unstable state or configuration changes force fresh inference. Cached bodies are
matched one-to-one to face anchors and never create unmatched body-only people.
The 2 s default aligns with `force_detect_seconds`: already gated static scenes
keep their existing pose cadence. Their workload is not reduced again. Distance
and posture can retain the cached values for up to the interval on stable frames.

The browser encodes with asynchronous `toBlob()` and sends a 10-byte video header
plus JPEG bytes (binary tag `0x04`). Legacy base64 JSON clients remain supported.
One encoding is allowed in flight; stopped/reconnected/obsolete callbacks are
discarded, and video is skipped above 256 KiB of socket backlog before encoding
and again before sending. Audio framing is unchanged. The monitor counts encoded,
sent, skipped and backpressure frames. JPEG encode timing now reports asynchronous
wall time, not synchronous CPU or main-thread blocking time.

`frame_bytes_avg` and bandwidth now count complete accepted application messages:
UTF-8 JSON bytes or binary header + JPEG, excluding WebSocket/TCP/TLS overhead.
This is a more complete definition than the previous data-URI-only video count.

[Measurements from 2026-10-22](benchmarks/20261022-round2.json), compared against a
saved **round-one working tree**, with the same SDK/ORT 1.31.0:

| Controlled workload | Round one | Round two |
|---|---:|---:|
| Browser bytes/frame, identical 48,309-byte JPEG | 64,495 | 48,319 |
| Browser capture callback average / p95 | 7.20 / 9.30 ms | 2.37 / 3.30 ms |
| Real models, stable face, 40 frames @ 5 fps, pose calls | 40 | 4 |
| Same model case, median total CPU time (3 runs) | 1,019.1 ms | 628.7 ms |
| Same model case, median frame p50 | 7.86 ms | 4.26 ms |
| Default static gate, 40 frames @ 2 fps, pose calls | 10 | 10 |
| Unstable or no-face scenario, 40 frames, pose calls | 40 | 40 |
| Live replay, 80 frames + PCM over 16 s, pose calls | 80 | 8 |
| Live replay, mean vision latency | 15.03 ms | 9.86 ms |
| Live replay, audio drops | 0 | 0 |

The 5 fps model and live scenarios deliberately disable the face motion gate to
isolate pose scheduling; these savings do not predict every default-camera scene.
Model runs use two OpenCV threads on both sides; this does not change the server's
thread configuration. The browser uses a fake in-memory socket and synthetic
pixels: packet-observer overhead is excluded from callback time, and pending
encodes drain before counting. All 30 JPEG hashes match across versions; neither
side drops a frame. This isolates blocking and framing, not network latency or
camera cost. Total JPEG completion time was 7.05 vs 6.44 ms on this run; asynchronous
encoding primarily frees the main thread rather than proving a faster encoder.

The single shared-host live comparison reported process CPU 46.1% → 40.0%, but
vision p95 was 17.37 → 17.62 ms and audio RTF 0.195 → 0.203. These short samples do
not establish an RTF or tail-latency improvement. Real multi-person camera clips
remain to be evaluated; deterministic tests cover new body-only entry on motion,
face disappearance, identity/order changes, cache expiry and stop/reconnect races.

Reproduce after saving an old project snapshot (including access to its models):

```bash
python scripts/benchmark_round2.py --source /path/to/round-one --output /tmp/pose-before.json
python scripts/benchmark_round2.py --output /tmp/pose-after.json
python scripts/benchmark_frontend.py --before-dir /path/to/round-one/presence_demo/static --mode capture --seconds 6 --output /tmp/capture.json
python scripts/benchmark_realtime.py --scene face --fps 5 --full-vision --transport binary --output /tmp/live.json
```

## Ideas for an embedded port

- Run detection and embedding on the NPU; keep the gate, tracker and state machine on the CPU.
- The tracker and state machine are small and dependency-free; they can be ported (for example to Rust or Kotlin)
  with this Python version and its scenario tests as the reference.
- Use the camera's Y plane directly for the motion gate and skip JPEG encode / decode.
- With no face and no sound, drop to 0.2 fps and wake on voice activity.
