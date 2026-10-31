[English](../performance.md) · **繁體中文**

# 效能

目標是小型 ARM 裝置，所以管線設計成盡量少做事。以下數字都在 8 核 x86 測試伺服器上量測（OpenCV DNN、CPU）。
請在自己的硬體上重新量測。

## 四個省電機制

| 機制 | 作法 | 參數 |
|---|---|---|
| **移動閘門** | 把畫面縮成 64×48 灰階，和上一張比較。低於門檻且狀態穩定時跳過偵測、沿用上次結果。每 `force_detect_seconds` 仍會完整偵測；MAYBE_AWAY / AWAY 時一定完整偵測。 | `motion_gate`、`motion_threshold`、`force_detect_seconds` |
| **Embedding 節流** | 除非是新的臉、移動或大小變化超過 `embed_move_thresh`，或超過 `embed_interval`，否則沿用上次的 embedding。 | `embed_throttle`、`embed_interval`、`embed_move_thresh` |
| **自適應 fps** | 伺服器建議 fps：穩定時 `fps_low`；有臉出現、參與者消失、track 剛變動，或 MAYBE_AWAY / AWAY 時 `fps_high`。 | `adaptive_fps`、`fps_low`、`fps_high` |
| **ROI 裁切** | 裁下 ROI 加 `crop_margin`，縮到最寬 `max_width`，偵測後再把框換算回去。 | `roi_crop`、`crop_margin`、`max_width` |

### 基準測試

`uv run python scripts/benchmark.py` 以 1 fps 模擬 60 秒：有人靜止（0–15 s）、移動（15–30 s）、ROI 空（30–45 s）、
人回來（45–60 s）。畫面在記憶體中合成和 JPEG 編解碼，不寫任何影像檔。每組跑 3 次取最好的一次。

| 設定 | 每張平均 ms | p95 | 總 CPU | 相對全部關閉 |
|---|---|---|---|---|
| 全部關閉 | 21.2 | 27.2 | 6.27 s | — |
| 只開 ROI 裁切 | 16.4 | 23.0 | 4.43 s | −29% |
| 只開移動閘門 | 10.1 | 25.9 | 2.94 s | −53% |
| 只開 embedding 節流 | 14.4 | 25.5 | 4.50 s | −28% |
| 全部開啟，固定 fps | 7.9 | 22.1 | 2.06 s | −67% |
| 全部開啟 + 自適應 fps | 6.9 | 18.2 | 1.81 s | **−71%** |
| 姿勢，省電機制關閉 | 31.1 | 37.9 | 7.93 s | +27% |
| 全部開啟 + 姿勢 | 13.7 | 31.9 | 2.97 s | −53% |
| 全部開啟 + 姿勢 + 距離 | 14.0 | 32.4 | 3.09 s | −51% |

每種設定都正確判定出 AWAY。p95 只小幅下降：需要完整偵測的畫面成本一樣，節省來自「不用做」的畫面。

## 時間花在哪裡

各階段延遲，2 fps，640×480 的移動人臉，省電機制全開（p50 ms）：

| 階段 | legacy + rules | kalman + hmm + score |
|---|---|---|
| JPEG 解碼 | 0.7 | 0.6 |
| 移動閘門 | 0.2 | 0.2 |
| 人臉偵測（YuNet，裁切後） | 2.4 | 2.4 |
| **Embedding（SFace）** | **9.9** | **10.0** |
| 姿勢（MoveNet，開啟時） | 6.3 | 6.1 |
| 衣服直方圖 | — | 0.1 |
| 追蹤 + 在場 + 意圖 | < 0.1 | 0.3 |
| **總計**（無姿勢 / 有姿勢） | 13.7 / 19.9 | 14.0 / 20.2 |

- 臉部 embedding 是最大的成本，其次是姿勢，再來是偵測。
- 進階的幾層（Kalman、HMM、意圖）加起來不到 0.5 ms。
- 1280×720 輸入只多約 1 ms（因為有 ROI 裁切，只有解碼變慢）。
- 空畫面或靜止畫面大多在 1 ms 內完成。
- WebSocket 端到端 p50：沒有姿勢約 17–18 ms，有姿勢約 25 ms。

`max_width` 會影響小臉：320 px 時，高度約 80 px 以下的臉會漏掉，所以預設是 640（偵測約 2.4 → 5 ms）。
縮小後臉的寬度最好至少 40–50 px。

## Embedding 後端與 NPU

`embed_backend` 選擇 SFace 的計算方式。模型相同，只有執行環境不同。

| 後端 | 模型 | 執行方式 | 用途 |
|---|---|---|---|
| `fp32`（預設） | opencv_zoo SFace ONNX | ONNX Runtime（v0.13：OpenCV DNN） | 參考基準，最準 |
| `int8` | opencv_zoo 官方 INT8 ONNX | ONNX Runtime（v0.13：OpenCV DNN） | 小 4 倍；ARM CPU 上常比較快 |
| `npu` | 本專案匯出的 TFLite（`int8` 或 `fp16`） | LiteRT，可掛外部 delegate | Android NPU / GPU；伺服器上用 XNNPACK |

從 v0.14 開始，`fp32` 和 `int8` 跑在 ONNX Runtime 上，使用 `PRESENCE_ORT_PROVIDERS` 指定的執行提供者；兩種執行環境的
各階段數字見 [SDK 與 ONNX Runtime](sdk.md#實測成本apple-m-系列cpuonnxruntime-131opencv-414)。下表是 v0.13 的量測
（OpenCV DNN，x86 伺服器）。

每張都計算 embedding 的整條管線（p50 / p95 ms，640×480）：

| 後端 | Embed | 總計 |
|---|---|---|
| fp32，OpenCV DNN | 10.5 / 12.7 | 14.7 / 17.2 |
| int8，OpenCV DNN | 12.2 / 14.6 | 16.6 / 19.4 |
| npu fp16，LiteRT XNNPACK | 3.6 / 4.1 | 7.7 / 8.4 |

和 fp32 的一致性（`scripts/eval_embed_backends.py`：31 張 CC0 人像，每張 6 種增強，651 對，門檻 0.40）：

| 後端 | 和 fp32 的平均 cosine | 判定和 fp32 一致 |
|---|---|---|
| int8 ONNX | 0.972 | 99.85% |
| npu int8 TFLite | 0.991 | 100% |
| npu fp16 TFLite | 1.000 | 100% |

所以 `cos_match_int8` 預設也是 0.40。這裡比較的是量化模型和 fp32 是否一致，不是辨識準確率。

### 取得 TFLite 模型

模型不放在 git（10 MB 和 19 MB），請在獨立環境中匯出：

```bash
uv venv -p 3.12 /tmp/.venv-sface-export
VIRTUAL_ENV=/tmp/.venv-sface-export uv pip install onnx2tf tensorflow tf-keras onnx onnxruntime \
    onnx-graphsurgeon onnxsim sng4onnx psutil ai-edge-litert opencv-python-headless
/tmp/.venv-sface-export/bin/python scripts/export_sface_tflite.py
```

腳本在 ONNX 的複本上轉換，把 PReLU 改寫成單一的 `PRELU` op（對 NPU 友善），產生 fp16 和全 INT8 模型
（只在記憶體中用 CC0 人像校正），並和 fp32 比對。

### 使用 delegate

- 在伺服器上設定 `PRESENCE_NPU_DELEGATE=/path/to/delegate.so`。基於安全，只能用環境變數設定，不能透過 WebSocket：
  載入共享函式庫等於執行原生程式碼。
- delegate 載入失敗時，LiteRT 會改用 CPU。資源面板會顯示實際使用的後端和原因。
- 沒有 delegate 時，`npu_precision = int8` 會自動改用 fp16 模型，在 x86 CPU 上快約 10 倍。
- 在 Android 上，可以在 App 裡用同一個 `.tflite` 搭配 NNAPI 或廠商的 delegate。輸入：int8 `[1, 112, 112, 3]`，
  RGB − 128，用相同的 5 點相似變換對齊。用 TensorFlow 的 `benchmark_model`（`--use_nnapi`、`--external_delegate_path`、
  `--enable_op_profiling`）確認哪些 op 真的在 NPU 上執行。這些步驟都還沒在實機上驗證。

## 音訊成本

PureSound 管線在一顆核心上每 20 ms 一塊約 5 ms（即時率約 0.25），比 1–2 fps 的影像貴得多。請先在目標裝置上量測。
見[音訊](audio.md#效能)。

## 即時監控與第一輪量測

自適應 FPS 建議現在同時放在每則 `state` 與 `stats.conn.optim.recommended_fps`。
前端接受兩者，只有實際送幀頻率改變才重建 timer。原本前端只讀 `state`，server 卻只在
`stats` 傳送建議，實際會一直使用固定 FPS。

資源面板新增影像佇列等待（收到資料至 worker 開始）、executor 等待（提交至執行緒開始）、
推論鎖等待與 appearance 階段。佇列等待發生在既有 `total` 計時開始之前，不能將兩者混為一談。
音訊保留累積 `rtf`，另加近 5 秒實際時間窗內、依音訊長度加權的 `rtf_recent`、完整 block
處理 p95，以及收到資料至真正開始處理的等待平均／p95。RTF 是經過處理的時間除以音訊長度，
不是 CPU 使用率。

前端擷取、JPEG 編碼、狀態／音訊／統計繪製各保留最多 200 筆時間樣本，另顯示 DOM／canvas
工作量與 WebSocket 待送 bytes。繪製時間只涵蓋同步 callback，不包含之後的排版或合成。
音訊面板最多每秒重繪 5 次，但錄製與時間軸仍逐筆接收資料。Canvas 只在尺寸變動時重設；
畫面外的圖表跳過繪製，頻譜捲動改用 `drawImage()`，避免像素讀回。

2026-10-22 的[原始量測](../benchmarks/20261022-round1.json)：

| 相同輸入工作量 | 修改前 | 修改後 |
|---|---:|---:|
| 10 秒、100 則 metadata 的音訊面板繪製次數 | 100 | 51 |
| 同次回播的 DOM `innerHTML` 寫入 | 190 | 66 |
| Canvas width／height 指派 | 658 | 4 |
| 頻譜像素讀回 | 206 | 0 |
| 同步繪圖工作總時間，包含延後執行的音訊繪製 | 222.3 ms | 92.1 ms |
| 16 秒穩定空場影像＋PCM 串流的送幀張數 | 16 | 9 |

瀏覽器回播使用穩定狀態、持續變化的音訊電平與各版本實際訊息格式；不開啟鏡頭／麥克風，
不呼叫雲端。次數包含最後 250 ms 等待。Server 測試兩邊都使用同一版 SDK／ORT 1.31.0，
開啟 pose、關閉雲端服務。三次 server 測試的音訊丟棄皆為 0；RTF 修改前 0.213，修改後兩次
為 0.218／0.353，程序 CPU 也有波動。這些共用主機上的短測試，**不能證明 RTF、CPU 或單張延遲改善**。

在專案環境重現（同步環境時需加 `--extra puresound`）：

```bash
python scripts/benchmark_frontend.py --before-ref b093837aa5b53fcfc35706e1cf088556b9c69ebb --output /tmp/frontend.json
# 開啟印出的網址，按 Run before / after，保持頁面在前景直到完成。
python scripts/benchmark_realtime.py --output /tmp/realtime.json
```

## 第二輪：pose 獨立節流與 binary JPEG

`pose_interval` 預設 2 秒，範圍 0–2；0 關閉節流。穩定的人臉可重用對應的身體結果，
到期時可獨立於 face detection 更新。顯著影像變動、人臉改變／消失、body-only、
不穩定狀態或設定變更會強制重算。Body cache 與人臉 anchor 一對一匹配，
不會用 unmatched 舊 body 製造新的在場證據。預設 2 秒與 `force_detect_seconds` 對齊，
已被 motion gate 跳過的靜止場景維持原本頻率，不會再有同等幅度的節省。
穩定畫面的姿態與距離資訊可能沿用至多這段間隔。

瀏覽器改用非同步 `toBlob()`，以 binary `0x04` 的 10-byte header 加 JPEG bytes 傳送，
保留舊 base64 JSON 相容。一次只允許一筆編碼；停止、重連或排程變更後的過期 callback
不送出。Socket 待送量超過 256 KiB 時會在編碼前與送出前跳過影像，音訊格式不變。
面板新增編碼／送出／跳過／背壓計數；JPEG 編碼時間現在是非同步完成的 wall time，
不是同步 CPU 時間或主執行緒阻塞時間。

`frame_bytes_avg` 與頻寬改計完整 application message：JSON UTF-8 bytes 或 binary header＋JPEG，
不含 WebSocket／TCP／TLS overhead，比原本只計 data URI 更完整。

2026-10-22 與保存的**第一輪工作目錄**對比，兩邊皆使用同一版 SDK／ORT 1.31.0：

| 控制工作量 | 第一輪 | 第二輪 |
|---|---:|---:|
| Browser 每張封包 bytes，JPEG 同為 48,309 bytes | 64,495 | 48,319 |
| Browser 擷取 callback 平均／p95 | 7.20／9.30 ms | 2.37／3.30 ms |
| 真模型、穩定臉、40 張 @ 5 fps，pose 次數 | 40 | 4 |
| 同上，總 CPU 時間的三次中位數 | 1,019.1 ms | 628.7 ms |
| 同上，單張 p50 的三次中位數 | 7.86 ms | 4.26 ms |
| 預設靜態 gate、40 張 @ 2 fps，pose 次數 | 10 | 10 |
| 不穩定／無臉，40 張，pose 次數 | 40 | 40 |
| 即時回播 16 秒、80 張＋PCM，pose 次數 | 80 | 8 |
| 即時回播影像平均延遲 | 15.03 ms | 9.86 ms |
| 即時回播音訊丟棄 | 0 | 0 |

5 fps 的模型與即時回播刻意關閉 face motion gate，以分離 pose 節流效益，不能外推至所有預設
攝影機場景。模型 benchmark 兩邊各用 2 個 OpenCV threads，未改動 server 的 thread 設定。
Browser 使用合成像素與記憶體中的假 socket，callback 計時排除觀測封包的額外成本，
並等待最後一次編碼完成再統計。兩邊 30 張 JPEG 的 SHA-256 完全相同，皆未丟幀。
這測的是阻塞與封包成本，不是網路延遲或實體相機成本。這次 JPEG 完成平均時間為
7.05 → 6.44 ms；非同步編碼主要釋放主執行緒，不能據此宣稱 encoder 本身變快。

單次共用主機的即時測量中，程序 CPU 為 46.1% → 40.0%，但影像 p95 為 17.37 → 17.62 ms，
音訊 RTF 為 0.195 → 0.203，不能證明 RTF 或尾端延遲改善。真實多人影片仍待評估；
確定性測試已涵蓋有 motion 時新的 body-only 入場、人臉消失、身份／排序變更、cache 到期，
以及停止／重連的非同步競態。[原始數據](../benchmarks/20261022-round2.json)保留所有情境與限制。

保存舊版專案目錄（含模型存取）後可重現：

```bash
python scripts/benchmark_round2.py --source /path/to/round-one --output /tmp/pose-before.json
python scripts/benchmark_round2.py --output /tmp/pose-after.json
python scripts/benchmark_frontend.py --before-dir /path/to/round-one/presence_demo/static --mode capture --seconds 6 --output /tmp/capture.json
python scripts/benchmark_realtime.py --scene face --fps 5 --full-vision --transport binary --output /tmp/live.json
```

## 移植到嵌入式裝置的想法

- 偵測和 embedding 放在 NPU，閘門、追蹤器、狀態機留在 CPU。
- 追蹤器和狀態機很小、沒有相依套件，可以移植（例如 Rust 或 Kotlin），以這個 Python 版本和情境測試作為參考。
- 直接用相機的 Y 平面做移動閘門，省掉 JPEG 編解碼。
- 沒有臉也沒有聲音時降到 0.2 fps，用語音活動喚醒。
