**English** · [繁體中文](zh-TW/licenses.md)

# Licenses of third-party components

> **This is not legal advice.** It is an engineering summary of the license texts read from the official
> repositories, model cards and package metadata (October 2026). Have a lawyer review it before commercial use,
> especially the question of *training data* behind model weights, which license texts usually do not address.

Legend: ✅ license allows commercial use · ⚠️ allowed, but with open questions or risks · ❌ not recommended

This project itself is licensed under the [Apache License 2.0](../LICENSE) (see also [NOTICE](../NOTICE)). Model
weights and third-party packages are **not** covered by that license; their terms are below.

## Models

| Component | License | Source | Commercial? | Notes / risks |
|---|---|---|---|---|
| **YuNet** face detector (`yunet_s_dynamic.onnx` for ONNX Runtime; `face_detection_yunet_2023mar.onnx` for the OpenCV DNN fallback; same weights) | BSD-3-Clause (libfacedetection.train, `yunet_s_dynamic.onnx`); MIT (opencv_zoo) | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train) | ⚠️ mostly | Trained on WIDER FACE, whose terms are not stated in opencv_zoo. Whether weights trained on a research dataset may be used commercially is legally unsettled. Lower risk than SFace. |
| **SFace** face embedding (`face_recognition_sface_2021dec.onnx`) | Apache-2.0 (opencv_zoo directory license) | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) · [SFace](https://github.com/zhongyy/SFace) | ⚠️ **high risk** | The training set of these weights is not stated. The paper uses CASIA-WebFace, VGGFace2 and MS-Celeb-1M, which are web-scraped celebrity photos restricted to non-commercial research (MS-Celeb-1M was withdrawn). Face embeddings are also biometric data (GDPR, BIPA…). **Do not let a product depend on this model** without review; see alternatives below. |
| **SFace int8** (`…_int8.onnx`, `embed_backend=int8`) | Apache-2.0 | same | ⚠️ high risk (as SFace) | Official quantized version; sha256 verified on download. |
| **SFace TFLite** (`…_int8.tflite`, `…_fp16.tflite`, `embed_backend=npu`) | Derived from the Apache-2.0 SFace weights | generated locally by `scripts/export_sface_tflite.py`; **not in git, not distributed** | ⚠️ high risk (as SFace) | Format conversion does not change the weights' license or data questions. INT8 calibration uses CC0 portraits, in memory only. |
| **MoveNet MultiPose Lightning** (`movenet_multipose_lightning_fp16.tflite`; `movenet_multipose_lightning.onnx` exported locally by `scripts/export_movenet_onnx.py`, not distributed) | Apache-2.0 | [Kaggle model page](https://www.kaggle.com/models/google/movenet) | ✅ | Released by Google. The model card lists surveillance and identification as out of scope; this project uses it only for presence, orientation and distance. Validate accuracy for conversation distances yourself. |
| **PureSound `dpcrn_curriculum_v2`** (streaming ONNX + manifest, `audio_pipeline=puresound`) | Apache-2.0 (Copyright 2026 Milo Wu and PureSound contributors; code, SDK, configs, docs and weights) | [PureSound](https://github.com/mcw519/puresound), pinned commit; sha256 checked by `scripts/fetch_puresound_model.py` | ⚠️ weights yes, **datasets to be checked** | PureSound's NOTICE states that datasets and recordings are *not* covered by Apache-2.0. Typical speech-enhancement corpora have their own terms (some research-only); confirm the actual training data before commercial use. Not in git. |
| Haar cascade (fallback, shipped with OpenCV) | Intel License Agreement (BSD-like, in the XML header) | `cv2/data/` | ✅ | Used only when YuNet is unavailable. |
| HSV histogram "embedding" (fallback) | This project's code, no weights | `presence_demo/vision.py` | ✅ | No training data at all, but weak (mostly clothing colour). |
| ~~YOLOv8n-pose~~ (v0.3.0) | AGPL-3.0 | Ultralytics | ❌ removed | AGPL also covers network services. Removed in v0.4.0. |

## Cloud services (optional, off by default)

| Service | Terms | Notes |
|---|---|---|
| Typesafe "Jev" images API (`reid_enabled`) | Provider's terms of service (paid, not open source; commercial terms and the meaning of the `TypeSafe-Danger-Zone-Images` header not yet confirmed) | Needs `reid_consent`. Face crops are sent to the provider. See [Re-ID](reid.md). |
| AssemblyAI Universal-Streaming v3 (`asr_provider=assemblyai`) | Provider's terms of service (paid, not open source) | Audio is sent to the provider (US / EU regions selectable). See [ASR](asr.md). |

## Python packages (runtime)

| Package | License | Commercial? | Notes |
|---|---|---|---|
| fastapi | MIT | ✅ | |
| starlette | BSD-3-Clause | ✅ | |
| uvicorn[standard] (uvloop, httptools, watchfiles, websockets, python-dotenv, PyYAML) | BSD-3-Clause (uvloop MIT/Apache-2.0; httptools, watchfiles, PyYAML MIT; websockets, python-dotenv BSD-3) | ✅ | |
| pydantic / pydantic-core | MIT | ✅ | |
| numpy | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0 parts) | ✅ | |
| opencv-python-headless | Apache-2.0 | ⚠️ yes, with notices | The wheel bundles FFmpeg (LGPL-2.1), OpenBLAS, libvpx, libaom, OpenSSL and others (`LICENSE-3RD-PARTY.txt`). Dynamically linked LGPL is usually fine, but ship the licenses and allow replacement. |
| psutil | BSD-3-Clause | ✅ | |
| ai-edge-litert (LiteRT, `litert` extra) | Apache-2.0 | ✅ | Pulls in flatbuffers (Apache-2.0), protobuf (BSD-3), ml-dtypes (Apache-2.0), tqdm (MPL-2.0 AND MIT). |
| puresound-streaming (optional extra) | Apache-2.0, with NOTICE | ✅ ship the NOTICE | Installed as a uv git dependency from the public repo; not copied into this project. |
| onnxruntime (runs every model) | MIT | ✅ | GPU / NPU builds (onnxruntime-gpu, -directml, -qnn, -openvino) are MIT too, but bundle vendor libraries (CUDA, QNN SDK…) with their own terms. |
| anyio, h11, idna, click, annotated-types, typing-inspection | MIT / BSD-3 | ✅ | |
| typing-extensions | PSF-2.0 | ✅ | |
| certifi, tqdm | MPL-2.0 (tqdm MPL-2.0 AND MIT) | ✅ | File-level copyleft: only changes to those files must be shared. |

Development only (not shipped): pytest (MIT), httpx (BSD-3), ruff (MIT), websockets (BSD-3), onnx (Apache-2.0). The front end uses no
third-party JavaScript libraries and no CDN.

## Export tools (separate environment, not runtime)

onnx2tf (MIT), TensorFlow / tf-keras (Apache-2.0), onnx (Apache-2.0), onnxruntime (MIT), onnx-graphsurgeon
(Apache-2.0), onnxsim (MIT AND (Apache-2.0 OR BSD-2-Clause)), sng4onnx (MIT), tf2onnx (Apache-2.0): all ✅.

NPU runtimes: NNAPI is part of Android (AOSP, Apache-2.0). Vendor NPU delegates are usually proprietary SDKs; check
their terms with the chip vendor.

## Front-end assets (shipped)

| Component | License | Source | Notes |
|---|---|---|---|
| PureSound web design system (`tokens.css`, `base.css`, `components.css`, `shell.css`, `screens-shared.css`) | Apache-2.0 (Copyright 2026 Milo Wu and PureSound contributors) | [PureSound](https://github.com/mcw519/puresound) `puresound/web/static/css`, pinned revision | In `presence_demo/static/vendor/puresound/` with LICENSE and NOTICE; only font URLs made relative. |
| IBM Plex Sans / Mono (woff2, latin subset) | SIL Open Font License 1.1 | same, `fonts/` | `fonts/LICENSE.txt` included; OFL allows bundling, not selling the fonts alone. |

## Test data

| File | License | Notes |
|---|---|---|
| `tests/data/face.jpg` | CC0 1.0 | Julia Margaret Cameron, *Julia Jackson Duckworth*, 1867, [Cleveland Museum of Art 1996.343](https://clevelandart.org/art/1996.343). 500 px thumbnail, used only in tests and benchmarks. |
| Cleveland Museum of Art Open Access portraits (evaluation / calibration only) | CC0 1.0 (queried with `cc0=1`) | `scripts/faces_cc0.py` loads them into memory only; nothing is stored or distributed. |

## Problem components and alternatives

1. **SFace (face embedding): the main risk.** Options:
   - `PRESENCE_FACE_EMBEDDER=hist` disables SFace and uses the HSV histogram. Re-identification gets weaker, but
     the product only needs "is the same group still here?", not "who is it?", so this may be enough.
   - Use body appearance and position tracking instead (MoveNet + Kalman + clothing colour already cover much of
     it). Many person re-ID weights are trained on Market-1501 / DukeMTMC (the latter withdrawn).
   - Train or fine-tune a small model (e.g. MobileFaceNet) on data you are allowed to use commercially.
   - License a commercial face SDK.
   - Do not assume that "Apache / MIT code" means the weights are free for commercial use: most public face
     recognition weights are trained on MS1M, VGGFace2, Glint360K or WebFace.
2. **YuNet (face detection): low to medium risk.** Alternatives: MediaPipe BlazeFace (check its model card), or
   head position from MoveNet's nose / eye / ear keypoints without any face detector.
3. **Other pose models considered**: RTMPose / RTMO (code Apache-2.0, weights trained on COCO and other datasets
   with their own terms), MediaPipe Pose Landmarker (single person). MoveNet MultiPose was the only one officially
   released under Apache-2.0, multi-person, and runnable from pip on Linux without a GPU.
