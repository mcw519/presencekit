[English](../sdk.md) · **繁體中文**

# SDK 與 ONNX Runtime

從 v0.14 開始，偵測邏輯可以不透過網頁 demo 直接嵌入其他程式。`presence_demo.sdk.PresenceEngine` 接收鏡頭畫面（也可以加上
音訊），回傳在場狀態和事件；所有神經網路模型都透過一個小小的推論層 `presence_demo.inference`，在
[ONNX Runtime](https://onnxruntime.ai) 上執行。

## 安裝

```bash
pip install presence-demo                 # SDK：numpy、OpenCV（影像處理）、onnxruntime、psutil
pip install "presence-demo[web]"          # 再加上 FastAPI 網頁 demo（presence-demo 指令）
pip install "presence-demo[litert]"       # 再加上 LiteRT：MoveNet TFLite 備援、SFace TFLite／NPU delegate 後端
uv sync                                   # 開發：SDK + web + litert + 測試工具
```

要用 GPU 或 NPU，改裝對應的 ONNX Runtime 套件取代 `onnxruntime`（例如 `onnxruntime-gpu`、`onnxruntime-directml`、
`onnxruntime-qnn`、`onnxruntime-openvino`）。它們提供同一個 `onnxruntime` 模組，其他程式都不用改。

## 使用方式

```python
from presence_demo.sdk import PresenceEngine

engine = PresenceEngine()  # YuNet + SFace 跑在 ONNX Runtime，第一次使用時下載
while running:
    frame = camera.read()  # BGR uint8、不鏡像
    step = engine.process_frame(frame, speaking=my_vad())
    for ev in step.events:  # PresenceEvent(kind, reason, extra)
        if ev.kind == "transition":
            print(ev.extra["frm"], "->", ev.extra["to"], ev.reason)
    state = step.snapshot["state"]  # ACTIVE | IDLE | MAYBE_AWAY | AWAY | ENDED
```

| 方法 | 用途 |
|---|---|
| `process_frame(img, speaking=None)` / `process_jpeg(bytes, ...)` | 一張鏡頭畫面；回傳 `Step`（事件 + snapshot） |
| `set_speaking(bool)` | 兩張畫面之間，你的 VAD 狀態改變了 |
| `tick()` | 沒有新輸入時推進計時器（大約每 0.5 秒一次） |
| `process_audio(pcm16)` | PureSound 人聲分離 + 乾淨 VAD + 影像交叉確認（需要 `puresound` extra 和 `audio_pipeline="puresound"`）；回傳乾淨的 PCM16，說話狀態改變時另外回傳 `Step` |
| `configure({...})` | 修改設定（和網頁設定相同的扁平 key 與驗證規則）；回傳警告 |
| `calibrate_distance(m)`、`reset()`、`info()`、`close()` | 距離校正、開新 session、診斷資訊、釋放音訊狀態 |

說明：

- 引擎本身不開執行緒；所有呼叫由內部鎖依序執行，所以擷取和音訊的 callback 可以從不同執行緒呼叫。一個引擎對應一組
  鏡頭／麥克風串流。同一個程序裡的多個引擎共用模型 session（ONNX Runtime 的 `run` 是 thread-safe），但不共用狀態。
- 時間來自 `clock`（預設 `time.monotonic`）。重播或測試時可以給虛擬時鐘：`PresenceEngine(clock=lambda: t)`。
- `PresenceEngine(pipeline=VisionPipeline(detector=..., embedder=...))` 可以換成自己的偵測器或 embedder（protocol 見
  `presence_demo.detectors`）；`pose=False` 關閉姿勢模型，`pose=my_detector` 換成自己的。
- 匯入 `presence_demo.sdk` 不會載入網頁框架、不會開執行緒、不會讀模型（由 `tests/test_sdk.py` 和
  `tests/test_layering.py` 檢查）。

## ONNX Runtime 選項

```python
from presence_demo.inference import OrtOptions

engine = PresenceEngine(ort=OrtOptions(providers=("coreml", "cpu"), intra_op_threads=4))
```

| 欄位 | 預設 | 意思 |
|---|---|---|
| `providers` | `("cpu",)` | 執行提供者，用別名（`cpu`、`cuda`、`tensorrt`、`rocm`、`directml`、`coreml`、`nnapi`、`qnn`、`openvino`、`xnnpack`、`webgpu`）或 ORT 全名。`auto` = 第一個可用的 GPU 提供者（CUDA、ROCm、DirectML），否則用 CPU。不可用的提供者會略過；CPU 永遠是最後的備援。 |
| `provider_options` | `{}` | 各提供者的選項，例如 `{"cuda": {"device_id": 1}}` |
| `intra_op_threads` | `0` | 每個模型的執行緒數（0 = 由 ORT 決定） |
| `graph_optimization` | `"all"` | `disable`、`basic`、`extended` 或 `all` |
| `sanitize` | `True` | 移除重複列為輸入的權重（見下方） |

預設的 `OrtOptions.from_env()` 會讀 `PRESENCE_ORT_PROVIDERS`（例如 `coreml,cpu`）、`PRESENCE_ORT_THREADS` 和
`PRESENCE_ORT_GRAPH_OPT`；網頁 demo 也用同樣的變數。NPU 類的提供者（CoreML、NNAPI、QNN、OpenVINO）只有在明確指定時才會
使用，因為每個模型支援的運算子不同：在 Apple M 系列 Mac 上，SFace 用 CoreML 反而比 CPU 慢約 3 倍。切換前請先量測。
PureSound 音訊模型一律用 CPU 提供者（10 ms 的音框用加速器沒有好處）。

**圖整理（sanitize）。** SFace（fp32 和 int8）把每個權重同時列為 initializer 和圖的輸入。ONNX Runtime 會把這種權重視為
可覆寫，因此無法做常數摺疊，也無法把 Conv 和 BatchNorm 合併。`load_model` 在建立 session 前，於記憶體中移除這些重複的
輸入（不寫任何檔案；結果和 ONNX Runtime 自己的 `remove_initializer_from_input.py` 逐位元組相同）。SFace 因此快約 1.7 倍。

**Session。** `load_model(path, options)` 在整個程序中，每個（檔案、選項）只建立一個 session，所有引擎和網頁連線共用。
`loaded_models()` 列出它們；`info()` 顯示實際使用的提供者、載入時間和備援說明（例如 GPU 提供者失敗、改用 CPU）。

## 模型

所有模型檔都登記在同一份目錄 `presence_demo.inference.CATALOG`，由 `ModelStore` 存放：依序是 `ModelStore(root)`、
`$PRESENCE_MODEL_DIR`、原始碼目錄下的 `models/`、`~/.cache/presence-demo/models`（或 `$XDG_CACHE_HOME`）。下載的檔案
會用 sha256 驗證，並以原子方式寫入。

| Key | 檔案 | 執行環境 | 取得方式 |
|---|---|---|---|
| `yunet` | `yunet_s_dynamic.onnx` | ONNX Runtime（預設臉部偵測器） | 第一次使用時下載 |
| `sface` | `face_recognition_sface_2021dec.onnx` | ONNX Runtime（預設 embedder） | 第一次使用時下載 |
| `sface-int8` | `face_recognition_sface_2021dec_int8.onnx` | ONNX Runtime（`embed_backend = int8`） | 選用時下載 |
| `movenet` | `movenet_multipose_lightning.onnx` | ONNX Runtime（姿勢） | `scripts/export_movenet_onnx.py`（只需做一次，另開含 TensorFlow 的環境） |
| `puresound` | `dpcrn_curriculum_v2.onnx` + `.json` | ONNX Runtime（PureSound SDK） | `scripts/fetch_puresound_model.py --clone` |
| `yunet-opencv` | `face_detection_yunet_2023mar.onnx` | OpenCV DNN 備援 | 只在檔案已存在時使用 |
| `movenet-tflite` | `movenet_multipose_lightning_fp16.tflite` | LiteRT 備援（`litert` extra） | 第一次使用姿勢時下載 |
| `sface-tflite-*` | `face_recognition_sface_2021dec_{int8,fp16}.tflite` | LiteRT（`embed_backend = npu`） | `scripts/export_sface_tflite.py` |

`yunet_s_dynamic.onnx` 和 opencv_zoo 的 `face_detection_yunet_2023mar.onnx` 是同一個 YuNet 網路與權重（後者和上游訓練
repository 的 `yunet_s_640_640.onnx` 逐位元組相同），只是以動態輸入大小匯出；畫面和 ROI 裁切的大小不固定，ONNX Runtime
需要這個版本。ONNX 實作重現了 OpenCV `FaceDetectorYN` 的補邊、解碼和 NMS：輸出差距在 1e-3 像素以內
（`tests/test_inference.py`）。

MoveNet 沒有官方的 ONNX 版本。匯出腳本用 tf2onnx 轉換 TFLite 檔（相同權重、動態輸入），並且在多種輸入大小上和 LiteRT
比對，差距超過 1e-4 就不採用（實測 3e-6）。沒有 ONNX 檔時，姿勢會退回 LiteRT。正式產品請匯出一次，把檔案和應用程式
一起打包。

## 移植檢查清單

1. 為目標平台安裝對應的 ONNX Runtime 套件（CPU、GPU 或 NPU 版本），並設定 `providers`。
2. 打包模型檔（`ModelStore.status()` 列出哪些已存在），用 `PRESENCE_MODEL_DIR` 或 `ModelStore(root)` 指向它們；沒有網路
   的裝置請用 `allow_download=False`。
3. 以 0.5–2 fps 把不鏡像的 BGR 畫面送進 `process_frame`（`presence_demo.optimizer.recommend_fps` 提供網頁 demo 使用的
   自適應頻率），再加上你的 VAD；或把 16 kHz PCM16 送進 `process_audio`。
4. 讀取 `step.events`／`step.snapshot`；`engine.info()` 顯示每個模型實際跑在哪個提供者上。

## 實測成本（Apple M 系列、CPU、onnxruntime 1.31、OpenCV 4.14）

| 階段 | OpenCV DNN／LiteRT（v0.13） | ONNX Runtime（v0.14） |
|---|---|---|
| YuNet，640×480 | 4.6 ms | 2.7 ms |
| SFace fp32，一張臉 | 5.4 ms | 3.0 ms |
| SFace int8，一張臉 | 5.6 ms | 3.8 ms |
| MoveNet MultiPose | 3.2 ms（LiteRT XNNPACK） | 5.2 ms |

在這台機器上，MoveNet 用 ONNX Runtime 的 CPU 提供者比 LiteRT 的 XNNPACK 慢；以 demo 使用的 0.5–2 fps 計算，每次姿勢
推論約多 2 ms。如果在你的目標平台上這很重要，可以安裝 `litert` extra 並移除 ONNX 檔。
