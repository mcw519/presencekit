[English](../licenses.md) · **繁體中文**

# 第三方元件授權

> **這不是法律意見。** 本文是工程上的整理，依據 2026 年 10 月從官方儲存庫、模型卡和套件中繼資料讀到的授權文字。
> 正式商用前請由法務確認，特別是授權文字通常沒有處理的「模型權重的*訓練資料*」問題。

圖例：✅ 授權允許商用　⚠️ 允許，但有未確認或潛在風險　❌ 不建議商用

本專案採用 [Apache License 2.0](../../LICENSE)（另見 [NOTICE](../../NOTICE)）。模型權重和第三方套件**不在**此授權範圍內，
各自的條款如下。

## 模型

| 元件 | 授權 | 來源 | 商用？ | 說明／風險 |
|---|---|---|---|---|
| **YuNet** 人臉偵測（ONNX Runtime 用 `yunet_s_dynamic.onnx`；OpenCV DNN 備援用 `face_detection_yunet_2023mar.onnx`；權重相同） | BSD-3-Clause（libfacedetection.train，`yunet_s_dynamic.onnx`）；MIT（opencv_zoo） | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train) | ⚠️ 大致可以 | 用 WIDER FACE 訓練，opencv_zoo 沒有寫明該資料集的條款。用研究資料集訓練的權重能否商用，在法律上沒有定論。風險比 SFace 低。 |
| **SFace** 人臉 embedding（`face_recognition_sface_2021dec.onnx`） | Apache-2.0（opencv_zoo 目錄授權） | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) · [SFace](https://github.com/zhongyy/SFace) | ⚠️ **高風險** | 沒有寫明這個權重的訓練資料。論文使用 CASIA-WebFace、VGGFace2、MS-Celeb-1M，這些是從網路蒐集的名人照片，只限非商業研究（MS-Celeb-1M 已撤下）。臉部 embedding 也屬於生物特徵資料（GDPR、BIPA 等）。**未經審查不要讓產品依賴這個模型**，替代方案見下方。 |
| **SFace int8**（`…_int8.onnx`，`embed_backend=int8`） | Apache-2.0 | 同上 | ⚠️ 高風險（同 SFace） | 官方量化版；下載時檢查 sha256。 |
| **SFace TFLite**（`…_int8.tflite`、`…_fp16.tflite`，`embed_backend=npu`） | 衍生自 Apache-2.0 的 SFace 權重 | 由 `scripts/export_sface_tflite.py` 在本機產生；**不放進 git、不散布** | ⚠️ 高風險（同 SFace） | 格式轉換不會改變權重的授權和資料疑慮。INT8 校正只用 CC0 人像，只在記憶體中處理。 |
| **MoveNet MultiPose Lightning**（`movenet_multipose_lightning_fp16.tflite`；`movenet_multipose_lightning.onnx` 由 `scripts/export_movenet_onnx.py` 在本機匯出，不散布） | Apache-2.0 | [Kaggle 模型頁](https://www.kaggle.com/models/google/movenet) | ✅ | Google 官方釋出。模型卡把監控和身分辨識列為非預期用途；本專案只用來判斷有沒有人、朝向和距離。對話距離下的準確度請自行驗證。 |
| **PureSound `dpcrn_curriculum_v2`**（串流 ONNX + manifest，`audio_pipeline=puresound`） | Apache-2.0（Copyright 2026 Milo Wu and PureSound contributors；程式碼、SDK、設定、文件和權重） | [PureSound](https://github.com/mcw519/puresound)，固定 commit；由 `scripts/fetch_puresound_model.py` 檢查 sha256 | ⚠️ 權重可以，**資料集待確認** | PureSound 的 NOTICE 說明資料集和錄音*不在* Apache-2.0 範圍內。常見的語音增強語料各有條款（部分只限研究），商用前請確認實際使用的訓練資料。不放進 git。 |
| Haar cascade（備援，OpenCV 內附） | Intel License Agreement（BSD 類，寫在 XML 檔頭） | `cv2/data/` | ✅ | 只在沒有 YuNet 時使用。 |
| HSV 直方圖「embedding」（備援） | 本專案程式碼，沒有權重 | `presence_demo/vision.py` | ✅ | 沒有訓練資料問題，但辨識力弱（主要靠衣服顏色）。 |
| ~~YOLOv8n-pose~~（v0.3.0） | AGPL-3.0 | Ultralytics | ❌ 已移除 | AGPL 也涵蓋網路服務。v0.4.0 已移除。 |

## 雲端服務（選用，預設關閉）

| 服務 | 條款 | 說明 |
|---|---|---|
| Typesafe「Jev」影像 API（`reid_enabled`） | 服務條款（付費、非開源；商用條款和 `TypeSafe-Danger-Zone-Images` header 的意義尚未確認） | 需要 `reid_consent`。人臉裁切會送到服務提供者。見 [Re-ID](reid.md)。 |
| AssemblyAI Universal-Streaming v3（`asr_provider=assemblyai`） | 服務條款（付費、非開源） | 聲音會送到服務提供者（可選美國／歐盟區域）。見[語音辨識](asr.md)。 |

## Python 套件（執行時）

| 套件 | 授權 | 商用？ | 說明 |
|---|---|---|---|
| fastapi | MIT | ✅ | |
| starlette | BSD-3-Clause | ✅ | |
| uvicorn[standard]（uvloop、httptools、watchfiles、websockets、python-dotenv、PyYAML） | BSD-3-Clause（uvloop MIT/Apache-2.0；httptools、watchfiles、PyYAML MIT；websockets、python-dotenv BSD-3） | ✅ | |
| pydantic / pydantic-core | MIT | ✅ | |
| numpy | BSD-3-Clause（含 0BSD、MIT、Zlib、CC0 元件） | ✅ | |
| opencv-python-headless | Apache-2.0 | ⚠️ 可以，需附授權 | wheel 內附 FFmpeg（LGPL-2.1）、OpenBLAS、libvpx、libaom、OpenSSL 等（`LICENSE-3RD-PARTY.txt`）。動態連結的 LGPL 一般可以，但散布時要附授權並允許替換。 |
| psutil | BSD-3-Clause | ✅ | |
| ai-edge-litert（LiteRT，`litert` extra） | Apache-2.0 | ✅ | 相依 flatbuffers（Apache-2.0）、protobuf（BSD-3）、ml-dtypes（Apache-2.0）、tqdm（MPL-2.0 AND MIT）。 |
| puresound-streaming（選用 extra） | Apache-2.0，附 NOTICE | ✅ 需附 NOTICE | 以 uv git 相依從公開 repo 安裝，沒有複製進本專案。 |
| onnxruntime（執行所有模型） | MIT | ✅ | GPU／NPU 版本（onnxruntime-gpu、-directml、-qnn、-openvino）也是 MIT，但會附帶廠商函式庫（CUDA、QNN SDK…），各有條款。 |
| anyio、h11、idna、click、annotated-types、typing-inspection | MIT / BSD-3 | ✅ | |
| typing-extensions | PSF-2.0 | ✅ | |
| certifi、tqdm | MPL-2.0（tqdm 為 MPL-2.0 AND MIT） | ✅ | 檔案層級的 copyleft：只有修改這些檔案本身才需要公開。 |

開發用（不隨產品散布）：pytest（MIT）、httpx（BSD-3）、ruff（MIT）、websockets（BSD-3）、onnx（Apache-2.0）。前端沒有使用第三方 JavaScript
函式庫或 CDN。

## 匯出工具（獨立環境，非執行時）

onnx2tf（MIT）、TensorFlow / tf-keras（Apache-2.0）、onnx（Apache-2.0）、onnxruntime（MIT）、onnx-graphsurgeon（Apache-2.0）、
onnxsim（MIT AND (Apache-2.0 OR BSD-2-Clause)）、sng4onnx（MIT）、tf2onnx（Apache-2.0）：都是 ✅。

NPU 執行環境：NNAPI 是 Android 的一部分（AOSP，Apache-2.0）。廠商的 NPU delegate 通常是專有 SDK，條款請向晶片廠商確認。

## 前端資源（隨套件散布）

| 元件 | 授權 | 來源 | 說明 |
|---|---|---|---|
| PureSound 網頁設計系統（`tokens.css`、`base.css`、`components.css`、`shell.css`、`screens-shared.css`） | Apache-2.0（Copyright 2026 Milo Wu and PureSound contributors） | [PureSound](https://github.com/mcw519/puresound) `puresound/web/static/css`，固定 revision | 放在 `presence_demo/static/vendor/puresound/`，附 LICENSE 和 NOTICE；只把字型網址改成相對路徑。 |
| IBM Plex Sans / Mono（woff2，latin 子集） | SIL Open Font License 1.1 | 同上 `fonts/` | 附 `fonts/LICENSE.txt`；OFL 允許隨軟體散布，不可單獨販售字型。 |

## 測試資料

| 檔案 | 授權 | 說明 |
|---|---|---|
| `tests/data/face.jpg` | CC0 1.0 | Julia Margaret Cameron，〈Julia Jackson Duckworth〉，1867，[Cleveland Museum of Art 1996.343](https://clevelandart.org/art/1996.343)。500 px 縮圖，只用在測試和基準測試。 |
| Cleveland Museum of Art Open Access 人像（只用於評估和校正） | CC0 1.0（查詢時加 `cc0=1`） | `scripts/faces_cc0.py` 只載入到記憶體，不保存、不散布。 |

## 有問題的元件與替代方案

1. **SFace（人臉 embedding）：最大的風險。** 可以考慮：
   - 用 `PRESENCE_FACE_EMBEDDER=hist` 停用 SFace，改用 HSV 直方圖。重新辨識能力會變弱，但產品只需要知道「同一群人還在不在」，
     不需要知道「是誰」，所以可能已經足夠。
   - 改用身體外觀和位置追蹤（MoveNet + Kalman + 衣服顏色已經涵蓋大部分）。很多行人 re-ID 權重用 Market-1501 / DukeMTMC 訓練
     （後者已撤下）。
   - 用可以商用的資料訓練或微調小模型（例如 MobileFaceNet）。
   - 購買有商用授權的人臉 SDK。
   - 不要假設「Apache / MIT 程式碼」就代表權重可以商用：大部分公開的人臉辨識權重都是用 MS1M、VGGFace2、Glint360K 或 WebFace 訓練的。
2. **YuNet（人臉偵測）：中低風險。** 替代方案：MediaPipe BlazeFace（請閱讀模型卡），或直接用 MoveNet 的鼻子／眼睛／耳朵關鍵點
   估計頭部位置，完全不需要人臉偵測器。
3. **評估過的其他姿勢模型**：RTMPose / RTMO（程式碼 Apache-2.0，權重用 COCO 等資料集訓練，各有條款）、MediaPipe Pose Landmarker
   （以單人為主）。MoveNet MultiPose 是唯一官方以 Apache-2.0 釋出、支援多人、而且能在沒有 GPU 的 Linux 上用 pip 執行的選項。
