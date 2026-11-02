[English](../configuration.md) · **繁體中文**

# 設定參數

伺服器端參數在 `presencekit/config.py`（`Config`），允許範圍在 `RANGES`。網頁送來的值會先檢查、限制在範圍內，
**只套用到該連線**，並回傳警告。`GET /api/config` 會回傳預設值和範圍。

**設定**抽屜依管線模組分組，順序和資料流相同：

```
輸入 → 偵測 → 姿勢（選用） → 距離（選用） → 追蹤 → 意圖 → 在場 → 狀態／逾時
     → 音訊（PureSound + 交叉確認） → 語音辨識（雲端） → re-ID（雲端）      ［效能、顯示：橫跨各模組］
```

每個模組最上面是開關或模式；下面的參數只在適用時顯示，變灰的參數會說明原因。*進階*參數要開啟「**進階參數**」才顯示
（改過的進階參數一律顯示）。每個模組都有自己的「**↺ 重設**」。

設定存在瀏覽器（`localStorage["presence-cfg-v1"]`），每次連線都會重送；也可以匯出、匯入 JSON。存過的設定會蓋過新版的
預設值，升級後如果想用新的預設值，請按「**預設**」。

## 分組檢視（v0.13）

`Config` 仍保留扁平鍵：它們依然是儲存方式、傳輸格式（`/api/config`、WebSocket `config` 訊息、預設組合、匯出的
JSON）以及下方的參數表，在 v0.13 完全照舊可用。此外每個鍵都恰好屬於 `presencekit.config.GROUPS` 中的一個分組
（`core`、`tracker`、`presence`、`intent`、`vision`、`pose`、`perf`、`audio`、`asr`、`reid`）。
`cfg.group("reid").reid_threshold` 與 `cfg.reid_threshold` 是同一個值；`cfg.group(name).update({...})` 只對該分組
套用相同的驗證規則；`cfg.as_groups()` 回傳巢狀複本。伺服器程序層級的設定仍然使用環境變數。

> **預計棄用：** 在 Python 程式碼中直接以扁平鍵存取，預計於**下一個**版本標示為棄用，改用分組檢視。v0.13 不會發出
> 任何警告，扁平的傳輸格式也不受影響。

## 模組之間的相依

| 模組 | 開關／模式 | 說明 |
|---|---|---|
| 輸入 | — | `adaptive_fps` 開啟時 `fps` 無效；`vad_*` 只在 `audio_pipeline = energy_vad` 使用 |
| 偵測 | — | `npu_precision` 只在 `embed_backend = npu`；`cos_match_int8` 只在 int8 / npu |
| 姿勢 | `pose_enabled` | `roi_rule = body` 需要姿勢；`turned_away_seconds` 需要 `turned_away_early` |
| 距離 | `distance_enabled` | `shoulder_width_m` 需要姿勢；影響意圖 `w_distance` 和音訊的 `false_accept` |
| 追蹤 | `tracker_mode` | `coast_*`、`gate_mahalanobis`、`appearance_*` 只在 `kalman` |
| 意圖 | `intent_model` | 權重和門檻只在 `score`；`intent_weighted_presence` 需要 `presence_model = hmm` |
| 在場 | `presence_model` | 機率參數只在 `hmm`；`present_frames` / `absent_frames` 只在 `rules` |
| 狀態／逾時 | — | `present_max_silence` 只在 `vision` 模式；`present_timeout` 用在 `vision` 以外 |
| 音訊 | `audio_pipeline` | 其他參數只在 `puresound`；`xcheck_*` 需要 `xcheck_enabled`；`false_accept` 和 `xcheck_require_calib` 需要距離 |
| 語音辨識 | `asr_provider` | 需要 `audio_pipeline = puresound`；部分選項只適用特定模型 |
| Re-ID | `reid_enabled` | 還需要 `reid_consent` 和 API key |
| 效能 | — | 子參數需要對應的省電機制開啟 |
| 顯示 | — | 只影響這個瀏覽器 |

## 預設組合

「推薦」是實地測試過的設定；「低角度鏡頭」「單人」「多人圍觀」「吵雜環境」建立在「推薦」之上，只列出差異。
「低功耗」只改效能相關的值，套用在目前的設定上。

| 組合 | 內容 |
|---|---|
| 推薦 | tracker `kalman`、presence `hmm`、intent `score`、姿勢開、距離開、`audio_pipeline` `puresound`、交叉確認開、`near_range` 1.6 m、`min_conf` 0.4、`leave_halflife` 1.5 s、`force_detect_seconds` 2 s、`join_seconds` 1.5 s、`grace_seconds` 10 s、臉部 ROI (0.25, 0.1, 0.8, 0.89)、身體 ROI (0.18, 0.04, 0.85, 0.92) |
| 低角度鏡頭 | 推薦 + `min_conf` 0.35、`leave_halflife` 2 s、`force_detect_seconds` 1.5 s、臉部 ROI (0.2, 0.05, 0.85, 0.95)、身體 ROI (0.1, 0, 0.9, 1)、`body_min_conf` 0.35 |
| 單人 | 推薦 + 臉部 ROI (0.25, 0.05, 0.75, 0.95)、`present_timeout` 45、`empty_timeout` 5、`new_group_seconds` 3、`absent_frames` 3 |
| 多人圍觀 | 推薦 + 臉部 ROI (0.08, 0.02, 0.92, 0.98)、`join_seconds` 3、`grace_seconds` 20、`present_frames` 2、`absent_frames` 4、`present_timeout` 90、`empty_timeout` 8、`new_group_seconds` 5、`cos_match` 0.42、`fps` 2、`near_range` 2 m |
| 吵雜環境 | 推薦 + `priority_mode` `vision`、`vad_k` 3.5、`vad_min` 0.02、`vad_hang` 300 ms、`min_conf` 0.5、`near_range` 1.2 m、`base_timeout` 15、`present_timeout` 45、`empty_timeout` 5 |
| 低功耗 | 移動閘門、embedding 節流（5 s）、自適應 fps 0.5–1、ROI 裁切、`max_width` 480、姿勢和距離關閉 |
| 預設 | 回到下表的預設值 |

## 環境變數

| 變數 | 意思 |
|---|---|
| `PRESENCE_TOKEN` | 設定後，頁面、API、WebSocket 都需要 `?token=` |
| `PRESENCE_MODEL_DIR` | 模型存放位置（預設：原始碼目錄下的 `models/`，否則 `~/.cache/presencekit/models`） |
| `PRESENCE_ORT_PROVIDERS` | ONNX Runtime 執行提供者，例如 `coreml,cpu` 或 `auto`（預設 `cpu`）；見 [SDK 與 ONNX Runtime](sdk.md) |
| `PRESENCE_ORT_THREADS`、`PRESENCE_ORT_GRAPH_OPT` | 每個模型的執行緒數（0 = 由 ORT 決定）與圖最佳化等級（`disable`/`basic`/`extended`/`all`） |
| `PRESENCE_FACE_EMBEDDER=hist` | 不載入 SFace，改用 HSV 直方圖備援 |
| `PRESENCE_NPU_DELEGATE` | `embed_backend = npu` 的外部 TFLite delegate 函式庫（只能在伺服器端設定） |
| `ASSEMBLYAI_API_KEY` | 選用語音辨識的 key |
| `TYPESAFE_API_KEY` | 選用 re-ID 服務的 key；`REID_JEV_URL` 可以改端點 |
| `PRESENCE_LOG_*`、`PRESENCE_EVENT_BUFFER`、`PRESENCE_VISION_LOCK_TIMEOUT` | 日誌，見[日誌](logging.md) |

`npu_delegate_path` 沒有介面，也不能透過 WebSocket 修改，只能用 `PRESENCE_NPU_DELEGATE` 設定。

## 參數表

由 `scripts/gen_config_docs.py` 從程式碼和介面字串表產生；內容過期時測試會失敗。名稱和說明和設定頁上顯示的完全相同。

<!-- BEGIN GENERATED: scripts/gen_config_docs.py -->

### 輸入（鏡頭／麥克風）

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `fps` | 1 | — | **每秒送出張數** *(只在瀏覽器)* — 每秒送幾張畫面到伺服器。越高越即時，但越耗頻寬和 CPU。 |
| `vad_k` | 2.5 | — | **VAD 靈敏度（門檻倍數）** *(只在瀏覽器)* — 音量超過「噪音底 × 這個倍數」才算說話。越小越靈敏。 |
| `vad_min` | 0.005 | — | **最低說話音量** *(進階, 只在瀏覽器)* — 絕對音量門檻（RMS），環境吵時調高。 |
| `vad_hang` | 400 | — | **說話尾音保留** *(進階, 只在瀏覽器)* — 音量變小後還要維持「說話中」多久，避免句子中間斷掉。 |

### 視覺偵測

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `face_roi` | (0.2, 0.05, 0.8, 0.95) | box (x0, y0, x1, y1) | **臉部 ROI (x0,y0,x1,y1)** — 臉的中心點要落在這個框（青色）裡才算數。可以直接在影像上拖曳或拉角調整。 |
| `min_conf` | 0.4 | 0 – 1 | **最低偵測信心** — 低於這個值的偵測不算有人也不算沒人，會退回只看聲音。 |
| `embed_backend` | `fp32` | `fp32` / `int8` / `npu` | **嵌入後端** — 計算人臉特徵（SFace）的方式。fp32：原始模型，最準（預設）。int8：opencv_zoo 官方的 INT8 量化版，檔案小 4 倍；在這台 x86 伺服器上反而比較慢，但在 ARM 手機上通常比較快。npu：轉成 TFLite，在 Android 上可以交給 NPU（NNAPI／NeuroPilot）計算；伺服器上沒有 NPU，會用 CPU 執行，並在「資源監控」顯示實際使用的加速器。 |
| `npu_precision` | `int8` | `int8` / `fp16` | **NPU 模型精度** *(進階)* — npu 後端使用的 TFLite 模型。int8：全整數量化，最適合 NPU。fp16：半精度，適合 GPU；也是沒有 NPU 時 CPU 最快的選項（int8 在沒有 NPU 時會自動改用 fp16）。 |
| `cos_match_int8` | 0.4 | 0 – 1 | **同一人門檻（量化模型）** *(進階)* — 使用 int8 或 npu int8 後端時的「同一人」相似度門檻。實測量化模型和 fp32 的判定一致率約 99.9%，預設和 fp32 相同。 |

### 姿勢（選配）

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `pose_enabled` | off | on / off | **啟用姿勢偵測** *(模式／開關)* — 用 MoveNet MultiPose Lightning（Apache-2.0，最多 6 人）偵測身體與關鍵點（肩、髖、鼻、眼）。側身或背對時沒有臉也能判斷有人。CPU 成本較高，預設關閉。 |
| `pose_conf` | 0.4 | 0.05 – 1 | **身體偵測信心** — 低於這個值的身體偵測會被忽略。 |
| `pose_interval` | 2 | 0 – 2 | **Pose 推論間隔** *(進階)* — 穩定時最多重用最近身體結果的秒數（預設 2 秒）；顯著影像變動、位置／人數變更、無臉或狀態不穩時立即更新。0 關閉節流。 |
| `body_min_conf` | 0.4 | 0.05 – 1 | **只有身體時的信心門檻** *(進階)* — 看不到臉、只偵測到身體時，信心達到這個值才算「有人」（有臉時用「偵測信心」）。 |
| `pose_roi` | (0, 0, 1, 1) | box (x0, y0, x1, y1) | **身體 ROI (x0,y0,x1,y1)** — 身體框的中心點落在這個框（紫色）裡才算數，通常比臉部 ROI 大。 |
| `roi_rule` | `either` | `either` / `face` / `body` | **參與者判定規則** — 臉或身體：臉在臉部 ROI，或身體在身體 ROI，任一成立就算。只看臉：只用臉部 ROI。只看身體：只用身體 ROI（需要開啟姿勢偵測）。 |
| `turned_away_early` | on | on / off | **轉身離開提早預警** — 所有參與者都轉身（側面或背對），而且身形變小（往遠處走）時，提早進入 MAYBE_AWAY。 |
| `turned_away_seconds` | 2 | 0 – 30 | **轉身持續時間** *(進階)* — 轉身＋遠離的狀態要持續多久才觸發。 |
| `recede_ratio` | 0.1 | 0.01 – 0.9 | **遠離判定比例** *(進階)* — 約 2 秒內身形（或臉寬）縮小超過這個比例，就視為正在遠離。 |

### 距離（選配）

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `distance_enabled` | off | on / off | **啟用距離估計** *(模式／開關)* — 用臉寬（約 0.15 m）和肩寬（約 0.38 m，只在正面時使用）加上鏡頭視角估計距離。 |
| `camera_hfov` | 60 | 20 – 170 | **鏡頭水平視角** — 鏡頭的水平 FOV。不知道的話用預設值，再用下方的校正。 |
| `distance_calib` | 1 | 0.2 – 5 | **校正係數** — 估計距離乘上這個係數。可以用下方「我站在 X 公尺」自動計算。 |
| `zone_min` | 0.4 | 0 – 10 | **對話區最近** — 比這更近的標示為「太近」。 |
| `zone_max` | 2 | 0.1 – 20 | **對話區最遠** — 對話區的外緣。超過但還沒到離開距離時標示為「偏遠」。 |
| `leave_distance` | 2.5 | 0.2 – 30 | **離開距離** — 超過這個距離，就算還在畫面裡也視為已離開。 |
| `face_width_m` | 0.15 | 0.08 – 0.3 | **平均臉寬** *(進階)* — 用於估計距離的平均臉寬。 |
| `shoulder_width_m` | 0.38 | 0.2 – 0.6 | **平均肩寬** *(進階)* — 用於估計距離的平均肩寬。 |
| `distance_ema` | 0.3 | 0.01 – 1 | **平滑係數** *(進階)* — 每次新估計值的權重（EMA）。越小越平滑，但反應越慢。 |

### 追蹤

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `tracker_mode` | `legacy` | `legacy` / `kalman` | **追蹤模式** *(模式／開關)* — 傳統：IoU + 臉部相似度，逐一貪婪配對（目前預設）。Kalman：每個人用 Kalman 濾波預測位置和速度，結合 IoU、移動距離、臉部相似度和衣服顏色（軀幹 HSV 直方圖，不需要模型），用匈牙利演算法一次配對；短暫被擋住時沿著預測位置繼續追蹤（虛線框）。 |
| `join_seconds` | 2 | 0 – 30 | **加入參與者所需時間** — 在 ROI 裡待滿這麼久才算參與者（開口說話會立即加入）。 |
| `grace_seconds` | 15 | 0 – 120 | **消失保留時間** — 參與者看不到超過這麼久才移除，避免短暫被擋住就被踢掉。 |
| `cos_match` | 0.4 | 0 – 1 | **同一人相似度門檻** — 臉部 embedding 的 cosine 相似度達到這個值就算同一人（SFace 約 0.36）。 |
| `cos_weak` | 0.2 | 0 – 1 | **弱相似度（位置重疊時）** *(進階)* — 框的位置大致重疊時，相似度只要達到這個較低的值也算同一人。 |
| `iou_match` | 0.3 | 0 – 1 | **位置重疊門檻 IoU** *(進階)* — 前後兩張畫面的框重疊多少算「位置一樣」。 |
| `embed_ema` | 0.15 | 0 – 1 | **模板更新速度** *(進階)* — 每張畫面更新多少比例的臉部模板。越大越能適應光線變化，但也越容易被別人帶偏。 |
| `coast_seconds` | 2 | 0 – 10 | **預測延續時間** — Kalman：沒偵測到時，沿著預測位置繼續追蹤多久。之後就當作不可見（寬限期照常計算）。 |
| `coast_present_conf` | 0.4 | 0 – 1 | **預測算有人的門檻** *(進階)* — Kalman：預測框的可見信心（從 1 隨時間降到 0）高於這個值時，仍然算「有人」。調高會更快判定離開。 |
| `gate_mahalanobis` | 9.21 | 1 – 100 | **移動距離門檻** *(進階)* — Kalman：偵測位置和預測位置的統計距離（Mahalanobis²）超過這個值就不配對，除非臉部相似度很高。9.21 約等於 99%。 |
| `appearance_min_sim` | 0.6 | 0 – 1 | **衣服相似度門檻** *(進階)* — Kalman：看不到臉時，衣服顏色相似度低於這個值就視為不同的人。 |
| `appearance_ema` | 0.2 | 0 – 1 | **衣服模板更新率** *(進階)* — Kalman：每次配對後，衣服顏色模板用多少比例更新成新的樣子。 |

### 互動意圖

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `intent_model` | `rules` | `rules` / `score` | **加入規則** *(模式／開關)* — 規則：在 ROI 待滿「加入時間」就成為參與者（目前預設）。意圖分數：每個人有 0–1 的互動意圖（是否面向機器人、距離、停留、是否走近、是否只是路過、是否在畫面中央、說話），分數高於門檻並持續「加入時間」才加入；路人、旁觀者不會加入，也不會讓對話一直持續。 |
| `join_threshold` | 0.6 | 0.05 – 0.99 | **加入門檻** — 意圖分數模式：意圖高於這個值並持續「加入時間」，才成為參與者（也才會開始新的對話）。 |
| `drop_threshold` | 0.35 | 0 – 0.95 | **降為旁觀者門檻** — 意圖分數模式：參與者的意圖低於這個值並持續「降級時間」，變成旁觀者（最後一位參與者不會被降級，離開與否由在場機率決定）。 |
| `drop_seconds` | 8 | 0 – 120 | **降級時間** *(進階)* — 意圖低於門檻要持續多久才降為旁觀者。 |
| `intent_tau` | 1 | 0.1 – 20 | **平滑時間** *(進階)* — 意圖分數的平滑時間常數，越大越穩定但反應越慢。 |
| `intent_weighted_presence` | off | on / off | **在場機率依意圖加權** — 開啟後（機率模式），參與者的在場機率乘上意圖比例：意圖低的參與者比較撐不住對話。 |
| `w_facing` | 0.45 | 0 – 1 | **權重：面向** *(進階)* — 臉是否正對（YuNet 五點）＋肩膀方向（姿勢）。面向也會整體縮放分數：背對的人很難得到高分。 |
| `w_distance` | 0.1 | 0 – 1 | **權重：距離** *(進階)* — 在對話距離內＝1，到「離開距離」降為 0。距離估計關閉時自動忽略。 |
| `w_dwell` | 0.1 | 0 – 1 | **權重：停留** *(進階)* — 在 ROI 待越久越高（4 s 滿分）。 |
| `w_approach` | 0.05 | 0 – 1 | **權重：走近** *(進階)* — 變大（走近）加分、變小（走遠）扣分。 |
| `w_still` | 0.15 | 0 – 1 | **權重：沒在路過** *(進階)* — 橫向移動越快分數越低，用來排除走過去的路人。 |
| `w_center` | 0.15 | 0 – 1 | **權重：位置** *(進階)* — 越接近 ROI 中央越高。 |

### 在場判斷

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `presence_model` | `rules` | `rules` / `hmm` | **在場判斷模型** *(模式／開關)* — 規則：用連續幀計數（有人／沒人）加逾時判斷（目前預設）。機率：每個人有一個「在場機率」，看到就往上、沒看到就往下，最後用整群的機率加門檻決定狀態，比較不怕偵測閃爍和短暫遮擋。搭配 Kalman 追蹤效果最好。 |
| `priority_mode` | `vision` | `vision` / `balanced` / `audio` | **判斷優先** — 影像優先：畫面有人就不會因為靜默離開，ROI 確定沒人就照 ROI 空的逾時離開（不管聲音），聲音只在影像不確定時使用，而且單靠聲音不會從 AWAY 恢復，適合吵雜環境。平衡：聲音和影像一起決定。聲音優先：主要看靜默，影像只微調逾時。 |
| `speech_join` | `auto` | `auto` / `off` / `single_candidate` / `all` | **說話立即加入** *(進階)* — 有人說話時，是否不用等「加入時間」就讓 ROI 內的人成為參與者（聲音沒有位置資訊）。自動：影像優先時關閉，其他模式用「唯一候選人」。唯一候選人：只有一位可見的候選人、臉在臉部 ROI 內、正面朝向（開距離時還要在對話區內）才立即加入。全部：所有可見的候選人都立即加入。 |
| `present_frames` | 2 | 1 – 20 | **確認有人所需張數** — 連續幾張畫面有人才算「有人」。 |
| `absent_frames` | 3 | 1 – 20 | **確認沒人所需張數** — 連續幾張畫面沒人才算「ROI 空」，可以防止偵測閃爍。 |
| `leave_halflife` | 1.5 | 0.1 – 30 | **離開半衰期** — 機率模式：完全沒看到人時，「在場」的勝算每隔這麼多秒減半。越小越快判定離開。預設 1.5 s，可以容忍短暫的偵測空檔。 |
| `detect_gain` | 4 | 0.1 – 20 | **偵測證據強度** *(進階)* — 機率模式：每秒偵測到人時，機率往上升的速度（乘上偵測信心）。越大回到 ACTIVE 越快。 |
| `lowconf_factor` | 0.15 | 0 – 1 | **低信心衰減比例** *(進階)* — 機率模式：只有低信心偵測時，機率下降速度是「完全沒看到」的多少倍。小一點可以在光線不好時不輕易判定離開。 |
| `p_active` | 0.8 | 0.05 – 0.99 | **ACTIVE 門檻** *(進階)* — 機率模式：群組機率高於這個值，才從 MAYBE_AWAY／AWAY 回到 ACTIVE。 |
| `p_maybe` | 0.6 | 0.02 – 0.98 | **MAYBE_AWAY 門檻** — 機率模式：群組機率低於這個值，進入 MAYBE_AWAY。 |
| `p_away` | 0.3 | 0.01 – 0.95 | **AWAY 門檻** — 機率模式：群組機率低於這個值並持續「AWAY 確認時間」，判定 AWAY。 |
| `away_hold_seconds` | 1 | 0 – 30 | **AWAY 確認時間** *(進階)* — 機率模式：機率要持續低於 AWAY 門檻多久才判定 AWAY。 |

### 狀態機／逾時

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `idle_after` | 3 | 0 – 60 | **靜默轉 IDLE** — 靜默多久會從 ACTIVE 轉成 IDLE。 |
| `present_timeout` | 15 | 1 – 600 | **有人時的逾時** — 平衡模式：畫面裡有參與者但都不說話時，靜默多久判定離開。預設 15 s，比影像不確定時的 20 s 短。 |
| `empty_timeout` | 5 | 1 – 120 | **ROI 空時的逾時** — 沒人又靜默時，多快判定離開（縮短）。 |
| `base_timeout` | 20 | 1 – 300 | **影像不確定時的逾時** *(進階)* — 影像信心太低時，只看聲音的逾時。 |
| `present_max_silence` | 0 | 0 – 3600 | **有人時最長靜默（影像優先）** — 影像優先模式下，人在畫面裡但一直不說話，最多等多久就判定離開。0＝不限制。 |
| `maybe_away_frac` | 0.5 | 0.05 – 0.95 | **MAYBE_AWAY 比例** *(進階)* — 逾時進行到這個比例時進入 MAYBE_AWAY。 |
| `end_after_away` | 5 | 0 – 120 | **AWAY 轉 ENDED** — AWAY 持續多久後結束對話並清除 embedding。 |
| `new_group_seconds` | 3 | 0 – 60 | **換組確認時間** — 舊參與者都不見、新的一群人出現多久後觸發 NEW_GROUP。 |

### 音訊（PureSound＋影像交叉確認）

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `audio_pipeline` | `energy_vad` | `energy_vad` / `puresound` | **音訊管線** *(模式／開關)* — 瀏覽器 VAD：用瀏覽器端的音量 VAD 判斷說話（目前預設）。PureSound：麥克風音訊（16 kHz PCM）串流到伺服器，用 PureSound 的串流人聲分離（dpcrn_curriculum_v2）去掉背景聲音，再用「乾淨音軌」判斷說話。需要伺服器安裝 puresound 套件與模型；沒有的話會在「乾淨音軌」面板顯示原因，並繼續用瀏覽器 VAD。 |
| `onset_entry_only` | on | on / off | **只在「進場」時開始對話** — 開啟時（PureSound 模式），對話只會在「乾淨音軌偵測到說話開頭」而且「畫面中有一位靠近的人」同時發生時開始；只有人經過、或只有聲音（電視、遠處的人）都不會開始。 |
| `onset_needs_vision` | on | on / off | **說話需搭配畫面** — PureSound 模式下，說話要「開始對話」或「從 AWAY 回來」時，最近 1.5 秒內畫面中必須有一位靠近的人（影像不確定時例外，改成只看聲音）。 |
| `iso_onset_guard` | off | on / off | **起音保護（onset guard）** — PureSound 內建：每段說話的前 1 秒先保留原始聲音，確認有人持續說話後才換成分離後的聲音，減少開頭字被吃掉。 |
| `iso_reset_on_change` | on | on / off | **換人時重置** *(進階)* — NEW_GROUP、新的對話或參與者換人時，重置人聲分離的串流狀態（也重新啟動起音保護）。 |
| `clean_vad_margin_db` | 10 | 3 – 30 | **乾淨 VAD 門檻** — 分離後的聲音比噪音底高出多少 dB 才算說話。越小越靈敏。 |
| `clean_vad_min_db` | -50 | -80 – -10 | **乾淨 VAD 最低音量** *(進階)* — 分離後的聲音至少要多大聲（dBFS）才算說話。遠處或很小聲的說話會被忽略。 |
| `clean_vad_hang` | 0.3 | 0 – 2 | **乾淨 VAD 尾音保留** *(進階)* — 分離後的聲音變小後，還要維持「說話中」多久。 |
| `xcheck_enabled` | on | on / off | **影像交叉確認** — PureSound 一直在過濾（收音範圍約 1 m），每 20 ms 用畫面確認一次結果：有乾淨說話但畫面沒人＝漏音（壓低、不算說話）；有乾淨說話但所有人都在近距離範圍外＝聲學誤判-超出範圍（壓低、不算說話）；乾淨音軌沒聲音，但範圍內有人而且原始麥克風有說話＝聲學誤判-漏收（多放一些原始聲音、算說話）。畫面不確定時只看聲音。 |
| `near_range` | 1 | 0.2 – 10 | **近距離範圍** — PureSound 的收音範圍。距離估計開啟時，所有人都比這個遠就視為「超出範圍」。沒開距離估計時只看 ROI 內有沒有人。 |
| `xcheck_hold` | 0.75 | 0 – 5 | **畫面保持時間** — 畫面每秒只有 0.5–2 張，聲音每 20 ms 一塊；看到人之後保持這麼久，對齊兩者、避免句首被切掉。 |
| `xcheck_stale` | 2 | 0.2 – 30 | **畫面過期時間** *(進階)* — 超過這麼久沒有新的畫面結果，就只看聲音。 |
| `xcheck_require_calib` | off | on / off | **距離需先校正** — 開啟時，要先在「距離」做過校正才會判斷「超出範圍」。 |
| `xcheck_floor_db` | -40 | -80 – 0 | **壓低音量** — 漏音或超出範圍時輸出降低多少（≤ -60 dB 為靜音）。 |
| `xcheck_recover_mix` | 0.3 | 0 – 0.9 | **漏收時的增強比例** *(進階)* — 聲學誤判-漏收時，輸出中分離後聲音的比例；其餘是（時間對齊的）原始聲音。越小保留越多原聲。 |
| `xcheck_attack` | 0.01 | 0.001 – 0.5 | **音量上升時間** *(進階)* — 壓低解除時的淡入時間常數。 |
| `xcheck_release` | 0.05 | 0.005 – 2 | **音量下降時間** *(進階)* — 開始壓低時的淡出時間常數。 |

### 語音辨識（雲端，選用）

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `asr_provider` | `none` | `none` / `assemblyai` | **語音辨識服務** *(模式／開關)* — 無：不做語音辨識（預設）。AssemblyAI：把聲音即時送到 AssemblyAI 雲端（Universal-Streaming v3）轉成文字。⚠ 這是外部雲端服務，聲音會離開這台裝置；請確認使用者同意與當地法規。需要 PureSound 管線與 API key。 |
| `asr_source` | `clean` | `clean` / `raw` | **送出的聲音** — 乾淨：送出經過 PureSound 與影像確認後的乾淨音軌（預設）。原始：送出麥克風原始聲音。 |
| `asr_feed` | `speech` | `speech` / `session` | **送出時機** — 只在說話時：只有「校正後算說話」的片段送出聲音，其餘送靜音（保持時間對齊，內容不離開裝置）。整個對話：對話進行中的聲音全部送出。對話結束（ENDED）一律斷線。 |
| `asr_region` | `global` | `global` / `us` / `eu` | **資料區域** — AssemblyAI 端點：全球（最近的區域）、美國或歐盟（資料不離開該區域）。 |
| `asr_speech_model` | `universal-3-6-pro` | `universal-3-6-pro` / `universal-3-5-pro` / `universal-streaming-english` / `universal-streaming-multilingual` | **模型** — Universal-3.6 Pro：32 種語言（含中文 zh、粵語 yue），可混用語言（預設）。Universal-3.5 Pro：19 種語言（含 zh，不含 yue）。Universal-Streaming English：只有英文。Universal-Streaming Multilingual：英、西、德、法、葡、義，不含中文。 |
| `asr_language_codes` | `zh,en` | text ≤ 200 | **語言** — 逗號分隔的語言代碼（例如 zh,en），偏向這些語言但仍可混用；留空＝不限制。只適用 Universal-3.x Pro。 |
| `asr_keyterms` | — | text ≤ 2000 | **關鍵詞** *(進階)* — 逗號分隔，最多 100 個，提高這些詞的辨識率（例如人名、產品名）。 |
| `asr_format_turns` | on | on / off | **格式化結果** *(進階)* — 句子加標點與大小寫。只適用 Universal-Streaming 模型（3.x Pro 一律格式化）。 |
| `asr_eot_confidence` | 0.4 | 0 – 1 | **句尾信心門檻** *(進階)* — 判斷一句話結束的信心門檻。只適用 Universal-Streaming 模型。 |
| `asr_min_turn_silence` | 400 | 50 – 10000 | **最短句尾靜默** *(進階)* — 靜默多久開始檢查是否句尾。 |
| `asr_max_turn_silence` | 1280 | 100 – 10000 | **最長句尾靜默** *(進階)* — 靜默這麼久一定結束這句話。 |

### 跨 session 使用者判斷（雲端，選用）

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `reid_enabled` | off | on / off | **跨 session 使用者判斷** *(模式／開關)* — 新 session 立即開始。將已加入參與者與近期 conversation 成員比對；只有成員重疊且無歧義時，才透過 handoff 接回原對話。不確定就不關聯。外部雲端服務：人臉裁切會離開這台裝置。 |
| `reid_consent` | off | on / off | **我已告知使用者並取得同意** — 必須取得同意：每個 conversation 最多三位參與者的臉部裁切，只保存在伺服器記憶體，並送至第三方雲端比對。不寫入磁碟或 log；開啟顯示照片時才傳送縮圖至此瀏覽器。 |
| `reid_show_crops` | on | on / off | **顯示比對照片** — 在「使用者比對」卡片顯示這次送去比對的臉和比對到的參考臉（小縮圖，約 112 px）。縮圖只經這個連線送到這個瀏覽器顯示，不寫入磁碟、log、除錯事件或錄音；「清除參考臉」或重設時一併清除。 |
| `reid_provider` | `jev` | `jev` | **雲端服務** *(進階)* — 比對用的雲端後端（可抽換）。目前提供 Jev（Typesafe）。 |
| `reid_model` | `jev-latest` | `jev-latest` / `jev-1.13.0` | **模型** — jev-latest 會跟著服務更新；固定 jev-1.13.0 可確保結果可重現。 |
| `reid_threshold` | 0.9 | 0.5 – 0.99 | **同一人門檻** — 分數達到這個值才視為同一人並接續對話。認錯人的代價較高，預設 0.9（LFW 評測：FAR 0%、FRR 14%）。 |
| `reid_uncertain_low` | 0.4 | 0 – 0.95 | **不確定下限** *(進階)* — 分數介於這個值和門檻之間時，再取一張新畫面比對一次；仍未達門檻就開新對話。 |
| `reid_ttl` | 600 | 30 – 600 | **參考臉保存時間** — 已結束對話的成員參考保存在記憶體多久，從最後一次正常 session END 起算。重連不延長期限，到期自動清除。 |
| `reid_max_refs` | 3 | 1 – 5 | **近期對話數** — 最多保留幾個已結束的 conversation，每個最多三位成員參考；目前活躍對話另外保留。會比對所有候選以排除歧義。 |
| `reid_min_face` | 0.06 | 0.02 – 0.5 | **最小臉寬** *(進階)* — 臉的寬度至少佔畫面這個比例才拿來比對。 |
| `reid_face_wait` | 3 | 0.5 – 15 | **等待清楚的臉** *(進階)* — 第一位參與者出現後，最多等這麼久拿到一張清楚的臉；等不到就開新對話。 |
| `reid_decision_timeout` | 10 | 0.5 – 30 | **最長等待時間** *(進階)* — 從 session 開始起算的實際時間期限，包含收集影像與雲端呼叫；未能決定就不關聯，AWAY 不暫停計時。 |
| `reid_timeout` | 2 | 0.3 – 10 | **單次呼叫逾時** *(進階)* — 雲端呼叫超過這個時間就放棄並開新對話。 |
| `reid_cooldown` | 5 | 0 – 120 | **冷卻時間** *(進階)* — 兩次判斷之間至少間隔這麼久；session 快速重開時直接開新對話，不呼叫。 |
| `reid_cap_per_hour` | 30 | 1 – 600 | **每小時上限** — 每小時最多呼叫次數，達到後一律開新對話。 |
| `reid_cap_total` | 200 | 1 – 10000 | **每連線上限** *(進階)* — 這個連線最多呼叫次數。 |
| `reid_cost_per_call` | 0 | 0 – 10 | **每次呼叫單價** — 自行填入的單價，只用來估算成本（0＝只顯示 tokens）。 |

### 效能

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `motion_gate` | on | on / off | **移動偵測閘門** — 畫面幾乎沒變化、狀態也穩定時，跳過人臉偵測，沿用上一次的結果。 |
| `motion_threshold` | 0.015 | 0 – 0.2 | **變化門檻** *(進階)* — 縮小灰階畫面的平均差異（0–1）低於這個值就視為沒變化。有雜訊的鏡頭要調高。 |
| `force_detect_seconds` | 2 | 0.5 – 30 | **強制偵測間隔** *(進階)* — 即使畫面沒變化，最久每隔幾秒也要完整偵測一次。 |
| `embed_throttle` | on | on / off | **Embedding 節流** — 人臉沒移動時沿用上次的 embedding；新的人臉、移動或縮放超過門檻、相似度偏低，或超過間隔時間時才重新計算。 |
| `embed_interval` | 3 | 0 – 30 | **Embedding 重算間隔** *(進階)* — 每張臉最久每隔幾秒重算一次 embedding。 |
| `embed_move_thresh` | 0.15 | 0 – 1 | **移動門檻** *(進階)* — 中心位移或大小變化超過臉框的這個比例就重算。 |
| `adaptive_fps` | on | on / off | **自適應 fps** — 由伺服器依狀態建議送幀速率：穩定時用低 fps，有變化或 MAYBE_AWAY 時用高 fps。開啟時會取代下方的「每秒送出張數」。 |
| `fps_low` | 0.5 | 0.2 – 5 | **穩定時 fps** — 狀態穩定（ACTIVE/IDLE/ENDED、沒有新人臉）時的建議 fps。 |
| `fps_high` | 2 | 0.2 – 5 | **變化時 fps** — 有新人臉、參與者消失、MAYBE_AWAY 時的建議 fps。 |
| `roi_crop` | on | on / off | **ROI 裁切＋縮小** — 只把 ROI（加邊界）裁下來並縮小後再偵測，偵測框會換算回原畫面座標。 |
| `crop_margin` | 0.1 | 0 – 0.5 | **裁切邊界** *(進階)* — ROI 四周多保留的範圍（畫面比例）。 |
| `max_width` | 640 | 96 – 1920 | **偵測最大寬度** — 送進偵測器的影像最大寬度。越小越快，但遠處的小臉可能偵測不到。 |

### 顯示（只影響這個瀏覽器）

| 鍵 | 預設 | 範圍 | 說明 |
|---|---|---|---|
| `mirror_view` | on | — | **鏡像顯示** *(只在瀏覽器)* — 像照鏡子一樣左右翻轉預覽畫面與標示框（自拍視角），比較直覺。只影響這個瀏覽器的顯示：送到伺服器的影像不翻轉，和機器人相機看到的一樣；ROI 數值也一律用原始相機座標儲存（鏡像時畫面上的左邊＝x 值大的一側）。 |
| `show_face_roi` | on | — | **顯示臉部 ROI** *(只在瀏覽器)* — 在影像上顯示青色的臉部 ROI 框（只影響顯示）。 |
| `show_pose_roi` | on | — | **顯示身體 ROI** *(只在瀏覽器)* — 在影像上顯示紫色的身體 ROI 框（只影響顯示）。 |

<!-- END GENERATED -->
