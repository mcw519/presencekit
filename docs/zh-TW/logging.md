[English](../logging.md) · **繁體中文**

# 日誌與診斷

伺服器輸出結構化、注重隱私的日誌，格式如下：

```
2026-10-15 12:00:00,000 INFO presence event=heartbeat conn=3f9a1c frames_rx=120 frames_done=118 ...
```

每筆紀錄也會放進記憶體中的環形緩衝，供介面的**事件紀錄**面板和 `/debug/events` 端點使用。除了你設定的日誌檔之外，不會保存任何東西。

## 從不記錄的內容

- 鏡頭畫面、聲音樣本，或任何可以還原它們的衍生資料。
- 逐字稿文字（只記錄 ASR 狀態和錯誤代碼）。
- Re-ID 臉部裁切（只在程序記憶體中，見 [Re-ID](reid.md)）；只記錄分數和計數。
- API key、存取權杖、密碼或 authorization header。名為 `key`、`token`、`secret`、`password`、`authorization` 的欄位
  會遮蔽成 `***`；網址中的 `token=` / `key=` 由套用在所有 handler（包含 uvicorn）的過濾器遮蔽。

## 設定（環境變數）

| 變數 | 預設 | 意思 |
| --- | --- | --- |
| `PRESENCE_LOG_LEVEL` | `INFO` | `DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `PRESENCE_LOG_FILE` | 未設定（stderr） | 輪替日誌檔路徑 |
| `PRESENCE_LOG_MAX_BYTES` | `5000000` | 超過這個大小就輪替 |
| `PRESENCE_LOG_BACKUPS` | `3` | 保留的輪替檔數 |
| `PRESENCE_EVENT_BUFFER` | `500` | 記憶體中保留給介面／端點的事件數 |
| `PRESENCE_VISION_LOCK_TIMEOUT` | `10` | 等待共用偵測器多少秒後回報卡住 |

## 事件

事件名稱和欄位是程式介面，所以維持英文；完整清單和說明見[英文版](../logging.md#events)。常用的幾個：

| 事件 | 等級 | 時機 |
| --- | --- | --- |
| `server_start`、`module_init` | info | 程序啟動、模型載入（`module`、`name`、`ms`） |
| `conn_open` / `conn_ready` / `conn_close` | info | 連線生命週期：id、來源、關閉原因、時長、計數 |
| `config`、`config_warnings` | info / warning | 設定差異（機密已遮蔽）、不會如預期運作的設定組合 |
| `state`、`session_open`、`session_event` | info | 狀態機轉換（`frm`、`to`、`reason`…）、session 開始／結束 |
| `track_new`、`track_lost`、`track_recovered`、`track_dropped` | info | track 生命週期 |
| `face_miss_summary` | info / warning | 每 30 秒：偵測次數、命中率（低於 0.6 為 warning） |
| `reid_*` | info / warning | re-ID：比對、分數、延遲、決定、等待原因；從不包含影像或 key |
| `asr`、`asr_skip` | info / warning | ASR 狀態和錯誤（從不包含文字） |
| `heartbeat` | info | 每個連線每 10 秒：收到／處理／丟棄的畫面、音訊區塊、積壓、迴圈延遲、交叉確認計數 |
| `slow_frame`、`loop_lag`、`audio_backlog`、`send_backlog` | warning | 單張超過 500 ms、事件迴圈被卡住、音訊佇列 > 200 ms、送往瀏覽器的佇列變長 |
| `no_frames`、`no_audio`、`host_suspend` | warning | 執行中但瀏覽器停止傳送、主機曾經休眠 |
| `bad_frame` | warning | 影像封包格式正確，但內容無法解碼成 JPEG（`frames_bad` 為累計張數；最多每 30 秒一次）。這張會略過，不算影像停滯 |
| `vision_stalled`、`audio_stalled`、`vision_lock_timeout` | error | watchdog：worker 5 秒沒有進展（附所有執行緒的 stack dump，重啟 worker，介面顯示錯誤橫幅） |
| `task_crash`、`task_gave_up` | error | worker 發生例外（附 traceback、退避後重啟）、重啟太多次而放棄 |

## 端點與介面

- `GET /debug/events?n=200&conn=<id>` 以 JSON 回傳最近的事件，和其他端點一樣受 `PRESENCE_TOKEN` 保護。
- 介面在可收合的**事件紀錄**面板（右下）顯示這個連線的事件；watchdog 或 worker 當機時會顯示錯誤橫幅。

## 排查卡住

啟動時會開啟 `faulthandler`。對執行中的程序：

```
py-spy dump --pid <PID>
```

watchdog 偵測到卡住時，也會把所有執行緒的 Python stack dump 寫進日誌。
