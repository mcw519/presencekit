[English](../audio.md) · **繁體中文**

# 音訊：PureSound 人聲分離 + 影像交叉確認

音訊回答「使用者在說話嗎？」。有兩種管線（`audio_pipeline`）：

| 管線 | 在哪裡 | 怎麼判斷「說話」 |
|---|---|---|
| `energy_vad`（預設） | 瀏覽器 | 音量超過自適應噪音底（`vad_k`、`vad_min`、`vad_hang`）。不需要安裝。 |
| `puresound` | 伺服器 | 串流人聲分離，對分離後的聲音做 VAD，再逐塊用影像交叉確認。 |

音量偵測分不出使用者、電視、音樂或房間另一頭的人。`puresound` 管線用兩步解決：聲學模型主要保留**近距離**的人聲，
再由鏡頭否決聲學模型判斷錯的部分。

```mermaid
flowchart TD
  MIC[麥克風 20 ms 一塊] --> RAW[原始 VAD]
  MIC --> ISO[PureSound 分離] --> CV[乾淨 VAD]
  CAM[影像快照，保持 0.75 s] --> XC{交叉確認}
  RAW --> XC
  CV --> XC
  XC -- 一致／安靜 --> KEEP[不變]
  XC -- 漏音／誤收 --> ATT[−40 dB，不算說話]
  XC -- 漏收 --> MIX[30% 分離 + 70% 原聲，算說話]
  XC -- 只看聲音 --> KEEP
```

![聲音交叉確認](../img/audio_crosscheck.zh-TW.png)

## 安裝

PureSound 是選用相依套件（公開的 [PureSound](https://github.com/mcw519/puresound) repo 中的獨立 SDK，Apache-2.0，
固定在某個 commit），需要 Python ≥ 3.12。

```bash
uv sync --extra puresound
uv run python scripts/fetch_puresound_model.py --clone     # 或 --from /path/to/puresound
```

模型（`dpcrn_curriculum_v2`，4.9 MB + manifest）只有在 sha256 和 PureSound 目錄一致時才會複製到 `models/`。
注意：不帶 `--extra puresound` 的 `uv sync` 會再把它移除。沒有 SDK 或模型時，選 `puresound` 會在音訊面板顯示原因，
狀態機繼續用瀏覽器 VAD。

## 第 1 步：分離與原聲混音

瀏覽器用 `AudioWorklet` 取麥克風聲音，重取樣到 16 kHz 單聲道，每 20 ms（320 個樣本）經 WebSocket 送出一塊。
PureSound 使用 STFT 512、10 ms hop 與 30 ms 前瞻。預設 `iso_dry_blend = 0.9`，輸出 `0.9 × 增強 + 0.1 × 對齊的原聲`，
與上游預設逐樣本一致。演算法延遲 52–62 ms。

在 **設定 → 詳細設定 → 音訊** 選擇 **PureSound** 後，可把 **人聲分離比例（dry blend）** 調整為 0.01–1.00。
新值會用於影像交叉確認前的分離輸出（`vi.wav`），套用時會短暫重置分離串流。

## 第 2 步：乾淨 VAD

模型沒有 VAD 輸出頭，所以**乾淨 VAD** 是對*分離後*訊號做能量偵測：10 ms 一幀、自適應噪音底（下降立即、上升約 3 dB/s），
音量超過 `max(clean_vad_min_db, 噪音底 + clean_vad_margin_db)` 持續 60 ms 算說話，尾音 `clean_vad_hang`（0.3 s）。
乾淨 VAD 讀取與 `vi.wav` 相同的音軌，位置在影像判斷混回原音與調整輸出增益之前。
同樣的偵測跑在原始輸入上就是**原始 VAD**，只用來發現漏收的說話。

## 第 3 步：影像交叉確認

預設自適應影像每秒 0.5–2 張，聲音每秒 50 塊。每塊音訊都讀取最新的追蹤快照。「ROI 內有人」「有人在 `near_range` 內」
「所有人都在 `near_range` 外」三個事實各自**保持 `xcheck_hold`**（0.75 s）。「所有人都在範圍外」只有在這段時間內
沒有看到近距離的人時才成立，以保護近距離說話者的第一個音節。

![判斷表](../img/audio_decisions.zh-TW.png)

| 乾淨 VAD | 原始 VAD | 影像 | 判斷 | 輸出 | 算說話？ |
|---|---|---|---|---|---|
| 有 | – | ROI 內沒人 | `leak` 漏音 | 壓到 `xcheck_floor_db`（−40 dB） | 否 |
| 有 | – | 所有人都在 `near_range` 外 | `false_accept` 誤收 | 壓低 | 否 |
| 有 | – | 有人在 `near_range` 內 | `agree` 一致 | 不變 | 是 |
| 無 | 有 | 有人在 `near_range` 內 | `false_reject` 漏收 | 目前模式允許救回時混入對齊的原聲 | 是，純音訊模式拒絕起音時為否 |
| 其他 | | | `silence` 安靜 | 不變 | 否 |
| 任何 | | 影像不確定 | `acoustic_only` 只看聲音 | PureSound 原樣輸出 | = 乾淨 VAD |

- **影像不確定**：還沒開始、決策用影像年齡超過 `xcheck_stale`（2 s）、只有低信心偵測，或關閉交叉確認。
- **距離**只在開啟距離估計時使用。沒開時「近距離」就是「在 ROI 內」，不會出現 `false_accept`。
  `xcheck_require_calib` 還要求先做距離校正。
- 增益變化淡入 10 ms（`xcheck_attack`）、淡出 50 ms（`xcheck_release`），避免爆音。

音訊面板保留「狀態進入次數」，另顯示每種判斷的持續秒數與占全部已處理 PCM 音訊的比例，包含
`silence` 與 `acoustic_only`。這些是判斷統計，不是辨識正確率。`decision_duration_s`、`decision_ratio`、
`classified_audio_s` 在同一個 `AudioStream` 內累積；模型 reset 不清除，新串流則從零開始。丟棄的輸入與沒有
音訊時的牆鐘等待不計入分母。

穩定狀態為 0.5 FPS 時，每兩秒才取一張圖，最近影像比音訊早接近兩秒可能單純來自取樣間隔。影像對齊欄因此
同時顯示目前取樣間隔、畫面完成後經過多久、server 接收到完成的耗時及其中的佇列等待。`vision_timing` 數值單位為
毫秒；抓取時間差使用共同的瀏覽器時鐘，server 耗時使用 server monotonic 時鐘，兩者不直接相減。過期判斷取
「server 收到影像後的年齡」與「非負的音訊／影像抓取時間差」較大者，避免慢畫面剛處理完就被當成新鮮。
缺少 client 時間戳時，server 收到後的年齡只是下界；JPEG 編碼與網路延遲尚未獨立量測。每批音訊固定使用實際
決策的影像時間戳與上下文，不受處理途中完成的新影像影響。

例子：電視開著，裝置前沒有人。PureSound 讓一部分電視人聲通過（聽起來夠近）。乾淨 VAD 說「有說話」，影像說「ROI 內沒人」
→ `leak`：這一塊被壓低 40 dB，不算說話，session 不會開始。

### 漏收起音救回模式

開啟 PureSound 和影像交叉確認後，到 **設定 → 詳細設定 → 音訊 → 漏收起音救回** 選擇模式。

- **原本的混回規則**（`legacy`，預設）。每次判為 `false_reject` 時都允許混回。
- **純音訊起音救回（實驗）**（`audio`）。符合音訊起音條件才混回原音。影像仍用來確認近處有人，可使用一般低幀率。
- **嘴部動作與音訊救回（實驗）**（`audiovisual`）。音訊條件或近場者的新鮮嘴部動作都可放行。

SDK 可設定 `Config(audio_pipeline="puresound", xcheck_recovery_mode="audio")`，並使用同一個單調時鐘送入影像與音訊。
舊設定的 `xcheck_mouth_recovery=true` 會對應到 `audiovisual`。

兩種實驗模式都調整 `false_reject`（raw VAD 有反應、乾淨 VAD 無反應、近處有人）時的額外原音比例。
乾淨 VAD 讀取混回前的 `vi`。影像年齡超過 `xcheck_stale` 時，兩版都回到 `acoustic_only`，
停止放行新的原音混回，混音逐步回到設定的 `vi` 比例，`speech` 以乾淨 VAD 為準。

- **嘴部動作。** YuNet 提供雙眼與嘴角。先以雙眼把相鄰影格的灰階臉部對齊，減少頭部平移、畫面內旋轉與縮放的干擾。
  再用 OpenCV 稠密光流估計像素位移，扣掉臉頰位移向量的中位數。嘴部剩餘位移量的第 80 百分位減去臉頰的
  第 80 百分位，差值最低取零，再除以嘴寬。當前近場追蹤對象最近三次連續分數的中位數須大於約 0.00459。
  相鄰影格的人臉須能透過重疊位置唯一配對。影格跳過、臉部遺失、配對不明確或影格間隔過長，都會中斷證據序列。
- **音訊起音。** raw VAD 確認起音後，取頭 60 ms 與前 200 ms 比較。起音的 RMS 須上升至少 8 dB，且
  1.8–6 kHz 能量須比 150 Hz–1.8 kHz 低至少 12 dB。與起音後頭 120 ms 重疊的輸出區塊可放行混回，並沿用
  既有的混音平滑。這一步只用已收到的 raw 樣本，利用 PureSound 輸出較早音訊的時間差。

兩條線索都不足時，混音會逐步回到設定的 `vi` 比例。PureSound SDK 的 `iso_dry_blend` 和內建起音保護都會影響
這條 `vi`。要對照離線實驗，設定 `iso_dry_blend=1` 並關閉內建起音保護。純音訊模式下，
`false_reject` 只有在起音條件允許救回的區塊才向 session 和依說話旗標取樣的 ASR 回報 `speech=True`。
被擋下的區塊保留 `false_reject` 診斷標籤，回報 `speech=False`，因此不更新 session 的最後說話時間。
安靜的參與者仍可依設定的在場規則維持 session。`legacy` 與 `audiovisual` 模式的 `false_reject` 回報
`speech=True`。嘴部動作是時間線索，無聲動嘴也可能有分數。音訊門檻來自同一份近遠場錄音上的實驗。

**嘴部線索的影像更新率和延遲。** 三次光流需要四張連續影格，任兩張間隔最多 85 ms。WebSocket 也要求影音有可比對的
時間戳，且做音訊判斷時影像有效年齡在 85 ms 內。預設 0.5–2 fps 取樣太疏，這時救回靠音訊線索。
要測嘴部線索，可關閉自適應 fps，把固定影像 fps 調高至 24。實驗會在有時間戳的畫面上執行 YuNet 更新 landmarks，
穩定場景也會更新，因此增加影像運算與傳輸。音訊使用當下已完成的證據，不等待下一張畫面，也不增加 PureSound 推論輪數。

音訊面板會顯示放行原因、嘴部動作分數、影像年齡及影格間隔，錄音 JSON 也會取樣保存 `recovery` 診斷。
比較實際處理幀率、效能面板的 `mouth` 花費，以及音訊處理和排隊 P95，就能看到高幀率有沒有造成積壓。
可在專案環境執行 `python experiments/puresound_mix/realtime.py BUNDLE OUTPUT.json`，
以錄音即時重播相同的 WebSocket 流程，量測伺服器端的處理能力。

一次 9 秒的本機重播中，0.5 和 2 fps 都沒有可用的嘴部動作分數。15 和 24 fps 分別在 60 筆音訊診斷中取得
55 筆分數、在 62 筆中取得 59 筆。固定 24 fps 時，開啟實驗讓影像處理 P95 從 2.45 升至 13.88 ms。
音訊處理 P95 在關閉與開啟時分別是 6.33 和 6.31 ms。正式部署時仍需在實際裝置上量測瀏覽器取樣、編碼和網路的影響。

## 第 4 步：狀態機看到什麼

PureSound 管線運作時，**修正後**的說話旗標取代瀏覽器 VAD：

- `onset_entry_only`（開）：**只有**修正後的說話開頭，加上最近 1.5 秒內看到一位在 `leave_distance` 內、信心足夠的人，
  才會開始 session。路人、只有聲音都不會。
- `onset_needs_vision`（開）：從 AWAY 用說話回來也要看得到人。
- `speech_join = auto` 在任何優先模式下，都把乾淨的說話歸給唯一的候選人。
- 音訊串流停止超過 1 秒，自動改回瀏覽器 VAD。

人換了的時候（NEW_GROUP、新 session、參與者換成另一組），會重置分離狀態（`iso_reset_on_change`），約 32 ms 沒有輸出。

## 監聽與錄影

- **監聽**：音訊面板可以播放乾淨或原始音軌給你聽，請戴耳機。監聽會多一個小的 jitter buffer（約 80 ms）和網路往返時間；
  判斷在伺服器上完成，不受影響。
- **錄影**：啟動相機後按「錄影」，會錄下鏡頭、臉部與身體偵測框、骨架，以及畫面上顯示的 ROI。瀏覽器 VAD 模式使用麥克風音訊，
  PureSound 模式使用交叉確認後的最終輸出。按「停止錄影」可在頁面直接播放，或下載瀏覽器支援的 MP4 或 WebM。
  每段最長 10 分鐘，預覽影片留在頁面記憶體。
- **影片與音軌下載**：PureSound 錄影結束後，可按「下載影片與音軌（ZIP）」。壓縮檔包含同一部影片、三軌樣本對齊的
  16 kHz WAV（`raw` 麥克風原聲、`vi` 人聲分離後、`out` 交叉確認後的最終輸出），以及 JSON 判斷時間軸。
  三軌 WAV 使用同一取樣時鐘，影片使用瀏覽器時間。

## 效能

8 核 x86 測試伺服器，單執行緒 ONNX Runtime：

| 項目 | 數值 |
|---|---|
| 分離即時率（RTF） | 0.22–0.27（每 20 ms 一塊約 5 ms） |
| 演算法延遲 | 52–62 ms |
| WebSocket + 佇列 + 計算，p50 / p95 | 6.0 / 7.8 ms |
| 端到端估計 | 約 85–100 ms |
| 伺服器 CPU，2 fps 影像 + 一張臉 | 只有影像 3.8% → 影像 + 音訊 31.4%（一顆核心 = 100%） |

音訊比影像貴得多，請先在目標裝置上量測再開啟。

## 已知限制

- `near_range`、`xcheck_hold` 和乾淨 VAD 的音量門檻還沒有在很多房間調過；−50 dBFS 的門檻和麥克風增益有關。
- 瀏覽器本身的回音消除和降噪會在 PureSound 之前執行。
- 電視在 ROI 內而且比 `near_range` 還近時，分不出它和人。
- 裝置附近有好幾個人時，交叉確認只知道*有近距離的人*在說話，不知道*是誰*。
- 距離誤差（例如鏡頭角度很低）可能造成錯誤的 `false_accept`，請先校正距離。

## 授權

PureSound 程式碼、SDK 和權重是 Apache-2.0。訓練資料不在這個授權範圍內，商用前請確認。見[授權](licenses.md)。
