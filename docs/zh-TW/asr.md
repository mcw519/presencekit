[English](../asr.md) · **繁體中文**

# 語音辨識（選用，雲端）

> **外部雲端服務：開啟後聲音會離開裝置**，送到 AssemblyAI。預設關閉（`asr_provider = none`）。
> 使用前請確認使用者同意、當地個資法規，以及服務條款和資料保留政策。

[PureSound 管線](audio.md)修正後的音訊可以串流到 AssemblyAI Universal-Streaming v3，產生即時字幕。逐字稿顯示在
**語音辨識**面板，也會以字幕列疊在影像上。

```
麥克風 → PureSound + 影像交叉確認 → 修正後乾淨音軌 ──(只在 session 進行中)──▶ AssemblyAI (wss)
瀏覽器面板 ◀── WS {"type": "asr"} ◀── 伺服器（只轉送，不保存） ◀── turn（partial / final）
```

## 運作方式

- 需要 `audio_pipeline = puresound`。連線由**伺服器**建立，用 `Authorization` header 帶 key，key 不會到瀏覽器。
- **只在 session 進行中**（狀態不是 ENDED）連線。ENDED、停止、斷線或換新 session 時送出 `Terminate` 並關閉。
- `asr_source`：`clean`（預設：修正後的輸出，漏音已被壓低）或 `raw`。
- `asr_feed`：`speech`（預設：只有算說話的區塊送真正的聲音，其他送靜音以維持即時節奏，內容不會離開裝置）或
  `session`（session 中全部送出）。
- 20 ms 的區塊累積成 100 ms 一則訊息。網路卡住時最多排隊 5 秒，更舊的丟掉。錯誤時以指數退避重新連線（最多 3 次，
  收到 `Begin` 後重設）。
- 每個 turn 帶有相對 session 開始的 `start_s` / `end_s`。
- 逐字稿只存在網頁記憶體（最近 100 句）。伺服器不保存、不記錄任何文字。

## API key

在[伺服器的 .env](configuration.md#使用-env-啟動) 設定 `ASSEMBLYAI_API_KEY`，或在 **設定 → 語音辨識** 手動輸入
（`POST /api/asr_key`，受 `PRESENCE_TOKEN` 保護）。手動輸入會覆蓋目前程序的啟動值。
`GET /api/asr_status` 只回報是否已設定與來源（`env`、`ui` 或 `null`）。
key 由所有連線共用，demo 對外開放時要設定 `PRESENCE_TOKEN`。

## 參數

| 參數 | 預設 | 送出的參數 | 適用模型 |
|---|---|---|---|
| `asr_speech_model` | `universal-3-6-pro` | `speech_model` | 全部 |
| `asr_language_codes` | `zh,en` | `language_codes` + `language_detection` | Universal-3.x Pro |
| `asr_keyterms` | 空 | `keyterms_prompt`（≤ 100 個） | 全部 |
| `asr_format_turns` | 開 | `format_turns` | Universal-Streaming |
| `asr_eot_confidence` | 0.4 | `end_of_turn_confidence_threshold` | Universal-Streaming |
| `asr_min_turn_silence` | 400 ms | `min_turn_silence` | 全部 |
| `asr_max_turn_silence` | 1280 ms | `max_turn_silence` | 全部 |
| `asr_region` | `global` | 端點：global / `streaming.us` / `streaming.eu` | 全部 |

只送所選模型支援的參數。依服務文件，Universal-3.6 Pro 支援 32 種語言、原生混語，包含普通話（`zh`）和粵語（`yue`）。
台灣口音的辨識品質和輸出字形（繁體或簡體）**沒有實測**。

## 測試

`tests/test_asr.py` 對本機的 mock WebSocket 伺服器測試：header 和參數、100 ms 分塊、`Terminate`、partial / final 解析、
key 端點不回傳 key、日誌裡沒有 key。只有設定 `ASSEMBLYAI_API_KEY` 時才會執行實際連線測試。

AssemblyAI 是付費服務，依串流時間計費。它不是開源元件；本專案只透過公開 API 使用。
