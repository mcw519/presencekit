# Changelog

> From 0.12.0 on, entries are written in English. Older entries are kept as written (Traditional Chinese) and may
> mention the previous upper-case doc names (e.g. `docs/REID.md` is now `docs/reid.md`).

## 0.14.0 — unreleased

Renamed the project, distribution, Python package and CLI to **PresenceKit** (`presencekit`). Update imports
from `presence_demo` to `presencekit` and launch the web demo with `uv run presencekit`. The `PRESENCE_*`
environment variables and browser settings are unchanged. Existing per-user `presence-demo/models` caches
are reused when the new `presencekit/models` cache does not exist. The GitHub repository is now `https://github.com/mcw519/presencekit`.

SDK release: every neural model runs on ONNX Runtime, and the detection logic can be embedded without the web
demo. Tracking, presence and state-machine behaviour, wire protocol and settings are unchanged (exceptions under
*Fixes*). See [docs/sdk.md](docs/sdk.md).

- **`presencekit.inference`** (numpy + onnxruntime only): `OrtOptions` (execution providers with aliases and
  `auto`, threads, graph optimisation; `PRESENCE_ORT_PROVIDERS` / `PRESENCE_ORT_THREADS` / `PRESENCE_ORT_GRAPH_OPT`),
  `load_model` with one shared session per (file, options) and an automatic CPU fallback when an accelerator fails,
  in-memory graph sanitising (weights listed as graph inputs are removed so ORT can fold constants; no `onnx`
  dependency), and one model catalogue + `ModelStore` (sha256-verified, atomic downloads; the root honours
  `PRESENCE_MODEL_DIR` everywhere, defaults to the per-user cache outside a source checkout).
- **Detectors on ONNX Runtime**: `OrtYuNetDetector` (YuNet-s dynamic-input export, same weights; reproduces
  OpenCV `FaceDetectorYN` padding, decoding and NMS within 1e-3 px), `OrtSFace` (fp32 and the official int8;
  cosine 1.0 vs OpenCV), `OrtMoveNetDetector` (local export by the new `scripts/export_movenet_onnx.py`, validated
  against LiteRT; LiteRT stays the fallback when the ONNX file is absent). PureSound sessions are created through
  the same layer. Measured on Apple M-series CPU: YuNet 4.6 → 2.7 ms, SFace 5.4 → 3.0 ms; MoveNet 3.2 → 5.2 ms
  (slower than LiteRT's XNNPACK). OpenCV DNN, LiteRT and Haar remain as fallbacks; registry names `yunet`, `sface`,
  `movenet` now mean the ORT implementations (`yunet-opencv`, `movenet-tflite` added).
- **`presencekit.sdk.PresenceEngine`**: frames (BGR or JPEG), host VAD or PureSound PCM in, `Step` out; settings,
  distance calibration, reset and diagnostics; no web framework, asyncio or threads (import is checked to load
  neither the web stack nor psutil).
- **Packaging**: `onnxruntime` is a core dependency; the web demo moved to the `web` extra and LiteRT to the
  `litert` extra (`uv sync` still installs both for development). `presencekit` without the web extra exits with
  an install hint.
- **Web UI / diagnostics**: Resources shows an ONNX Runtime row (version, accelerator, providers);
  `VisionPipeline.info()` adds `onnxruntime` and `ort_providers`.
- **Fixes** (from a code review): model loads and first-use downloads (pose, int8 SFace) no longer run while the
  process-wide vision lock is held, and downloads have a timeout; `audio.MODELS_DIR` ignored `PRESENCE_MODEL_DIR`;
  the YuNet OpenCV and MoveNet TFLite files are now checksum-verified; Haar-fallback rows reported a fully profile
  face (`frontal = 0`), which blocked joining with `intent_model = score` (now unknown → neutral default); an
  infinite value for an integer setting raised `OverflowError` instead of a warning; dwell and intent smoothing
  treated a timestamp of exactly 0 as missing (virtual clocks); a non-ASCII `?token=` caused HTTP 500 instead of a
  refusal. `calibrate` moved to `presencekit.body` (re-exported from `runtime.resources` and `server`).
- **State machine fixes**: with `priority_mode = vision` and `present_max_silence > 0`, a silent person in view made
  the state flip AWAY ↔ ACTIVE on every evaluation and the session never ended (rules and hmm). Continued silent
  presence is no longer a return from that AWAY: speech from someone in view, a re-entry or a new person is (the
  silence clock restarts on the return). Otherwise the session ends after `end_after_away`, and the same silent
  person does not restart it until they speak or leave the ROI. A NEW_GROUP timer still pending when a session
  ended survived into ENDED and blocked every later join, so with `intent_model = score` no session could start
  again; `_end()` now clears it.
- **Runtime fixes**: `AssemblyAIStream.close()` awaited a full queue that nobody drained once the stream had given
  up (no key, retries exhausted), hanging the audio worker and the connection close; it no longer waits. 0x03
  recording packets were posted outside the outbox back-pressure (unbounded growth with a slow client); they now
  share the audio bound, so a slow client gets gaps in its recording instead. Frames that pass the envelope check
  but do not decode no longer make the watchdog restart the vision worker every 5 s (new `bad_frame` warning).
  `FrameProcessor.reset()` from the event loop while a worker thread is inside `process()` is now deferred until
  that frame finishes (the in-flight frame cannot refill the caches) and never waits for inference; its counters
  are zeroed in place, so the stats keep a valid reference after a watchdog restart.

## 0.13.0 — 2026-10-30

Architecture release: the code is split into layers with a one-way dependency direction. No change in tracking,
presence, audio, re-ID behaviour, wire protocol, settings or defaults; no new features or models. See
[docs/architecture.md](docs/architecture.md#layers-v013).

- **`presence_demo.core`** (numpy only): typed values (`Observation`, `VideoFrame`, `PresenceEvent`, `Step`), the
  tracker split into `tracks` / `geometry` / `tracking` / `state_machine` / `tracker`, and `PresenceCore`, a typed
  facade (`update(Observation) -> Step`) that yields exactly the tracker's steps.
- **`presence_demo.detectors`**: `FaceDetector`, `Embedder`, `PoseDetector` protocols, the built-ins (YuNet, Haar,
  SFace, histogram, MoveNet) and a small name registry (`detectors.registry`). Settings do not select detectors by
  name yet.
- **`presence_demo.identity`**: re-ID split into crops, in-memory reference store, key handling, `IdentityMatcher`
  protocol + Jev client, call guard (caps, breaker), decision and `ContinuationDriver`. Crops are still kept in
  memory only.
- **`presence_demo.runtime`** (no FastAPI/Starlette/uvicorn): `PresenceRuntime` holds all per-connection logic that
  was inside the `/ws` handler; `AppContext` lists the process-wide dependencies it receives (references only;
  ownership unchanged); protocol (now with named binary tags and `CLIENT_MESSAGE_TYPES`), alignment, page continuity
  and process resources moved here. `web/session.py` is now the transport only.
- **Grouped settings view**: `Config.group(name)` / `Config.as_groups()`. Flat keys remain the wire and preset format
  and are fully supported; a deprecation of flat keys in Python code is planned for the next version (no warnings
  in 0.13).
- **Compatibility**: the 0.12 import paths (`presence_demo.tracker`, `vision`, `reid`, `web.runtime`,
  `web.protocol`, `web.alignment`, `web.continuity`) re-export the moved objects. The pipeline singleton now lives
  in `runtime.resources`; assign `runtime.resources._pipeline` instead of `web.runtime._pipeline` (internal).
- **Tests**: `tests/test_layering.py` enforces the import rules (static, plus a subprocess check that the lower
  layers never load the web framework); new `test_core.py`, `test_runtime.py` and detector/identity tests.

## 0.12.0 — 2026-10-16

Clean-up release: no change in tracking, presence or audio behaviour.

- **English UI with i18n.** Every visible string (labels, settings and help texts, module names, state
  descriptions, panel captions, buttons, event-log labels, presets) comes from string tables in
  `static/i18n/en.js` and `static/i18n/zh-TW.js`. A language switch (EN / 中文) in the sidebar; the choice is kept in
  `localStorage`, defaults to the browser language and falls back to English; `?lang=` overrides it.
  `tests/test_i18n.py` checks identical keys and placeholders, that every used key exists, and that no text is
  hard-coded in the scripts. The recording sidecar JSON is now in English.
- **Code structure.** `config.py` (Config, RANGES, State) split out of `tracker.py`; the web server split into
  `server.py` (app and routes) and `web/` (`runtime.py`, `outbox.py`, `session.py`); the front end split into
  `index.html` (markup only) and `static/js/*.js`. Dead code removed; tests renamed by topic instead of version.
- **Documentation rewritten** as a bilingual set (English in `docs/`, Traditional Chinese in `docs/zh-TW/`) with an
  index, plain-language algorithm pages (detection, tracking, intent, presence, state machine, audio, re-ID) and
  new figures generated from the real code (`scripts/make_figures.py`, `scripts/render_diagrams.sh`). The parameter
  reference is generated from the code and string tables (`scripts/gen_config_docs.py`) and checked by
  `tests/test_docs.py`. README rewritten as a project landing page, plus `README.zh-TW.md`.
- **Community files**: English `CONTRIBUTING.md` (+ zh-TW), `CODE_OF_CONDUCT.md`, `SECURITY.md`.

## 0.11.2 — 2026-10-16

- **新的「使用者比對」卡片**（在狀態機卡片下方）：每次跨 session 判斷後，並排顯示這次送去比對的臉和比對到的參考臉，加上分數、門檻、決策（↩ 接續 #n／＋ 新對話）、原因、時間；保留最近 4 筆。
  - **隱私規則變更（使用者明確要求）**：縮圖（112×112 JPEG）只經這個連線的 WebSocket 送到瀏覽器顯示，由新設定 `reid_show_crops`「顯示比對照片」控制（re-ID 啟用時預設開）。不寫磁碟、log、診斷事件或錄音；「清除參考臉」、reset、關閉功能時一併清除。`docs/REID.md` 的例外說明已更新。
- **設定頁版面**：API key、距離校正這類額外列改放進模組內容區，左右內距和其他欄位一致，不再貼齊卡片邊緣；按鈕一律用 PureSound 樣式（語音辨識和使用者判斷相同）；key 狀態改成徽章（已設定＝綠色）。
- 測試：會檢查「沒寫檔」的測試現在略過 `logs/`（執行中的開發伺服器會寫這個資料夾），避免誤判。

## 0.11.1 — 2026-10-16

依 2026-10-16 實測記錄分析的修正。
- **預設值調整**：`min_conf` 0.6 → 0.4、`leave_halflife` 0.6 → 1.5 s、`force_detect_seconds` 3 → 2 s。
- **預設組合**：「推薦」改成實測所用的設定（kalman＋hmm＋score、姿勢與距離開、PureSound＋交叉確認、near_range 1.6 m 與上述調整）；單人／多人圍觀／吵雜環境改成建立在「推薦」之上；新增「低角度鏡頭」。每個組合都寫在 `docs/CONFIGURATION.md`。
- **re-ID 等待**：新增 `reid_face_wait`（3 s），從第一位參與者出現才開始計時；`reid_decision_timeout` 改成整體上限 10 s；MAYBE_AWAY／AWAY 期間暫停計時（修正：上次 session 2 在 AWAY 時就逾時了）。
- **記錄**：新增 `session_open`（開啟原因）、`track_*` 生命週期、`face_miss_summary`、`reid_wait`、`asr_skip`（沒有 key）、`client`（瀏覽器回報分頁隱藏、鏡頭／麥克風中斷、重連、休眠）、`host_suspend`、`sender_closed`。見 `docs/LOGGING.md`。
- **瀏覽器自動重連並續跑**：8 s 沒收到訊息就判定連線失效並重連；鏡頭或麥克風中斷時自動重啟擷取。
- 瀏覽器斷線視為正常關閉，不再記成 `task_crash`。
- 修正偶爾失敗的測試 `test_ws_record_streams_three_tracks_and_writes_no_files`：改成檢查私有暫存目錄，不受其他程序在 /tmp 寫檔影響。

## 0.11.0 — 2026-10-16

- **新功能：跨 session 使用者判斷（雲端 re-ID，選用，預設關閉）**。新 session 開始時，把第一張清楚的臉和最近幾位使用者的參考臉送到可抽換的雲端後端比對（內建 Jev／Typesafe images API）。
  - 結果：分數 ≥ 0.9 → 接續上一段對話（沿用 `conversation_id`）；介於 0.4 和 0.9 之間 → 再比一張；其他情況或任何失敗 → 開新對話。
  - session 進行中不呼叫；沒有未過期的參考臉時直接開新對話，不呼叫。
  - 新的 WebSocket 事件 `handoff`（`conversation_id`、`decision`、`resumed_from_session`、`score`、`reason`），以及每秒的 `reid` 狀態。presence-demo 本身不做對話記憶。
  - 成本控制：冷卻、每小時與每連線上限、斷路器；呼叫數、tokens、估計成本顯示在狀態機卡片，並寫進 heartbeat 與 `reid_*` 記錄事件。
  - **隱私例外（明文）**：參考臉（每位一張 320×320 裁切）只在伺服器記憶體保存，最長 10 分鐘；到期、關閉、清除、reset、`POST /api/reid_reset`、斷線、關機時清除。不寫磁碟或 log，也不傳回瀏覽器。需要同時開啟 `reid_enabled` 和 `reid_consent`。
  - key 來自 `TYPESAFE_API_KEY` 或設定頁，只存在記憶體。
  - 文件：`docs/REID.md`。商用條款與 danger-zone header 的意義列為待確認事項。
  - 測試：`tests/test_reid.py`，用 mock 伺服器涵蓋決策、重試、TTL、上限、逾時、斷路器、清除、key 不外洩、不寫檔；另有選用的實機冒煙測試（`REID_LIVE=1`）。

## 0.10.8 — 2026-10-16

- **監看舞台重新設計**，參考 PureSound playground 的 CompareDeck（`deck.css`）：深色外框有內距；影像是圓角內嵌元素；音軌放在圓角內嵌面板裡，每條音軌左側有固定寬度的標籤欄（色點、名稱「原始／輸出」、noisy／output、即時電平 dB），右側是圓角的波形／頻譜 canvas，兩軌之間有淡分隔線。影像與音軌左右邊緣對齊，兩條音軌的時間軸對齊。
- 「波形／頻譜／兩者」切換移到舞台上方的「監看」標題列（在外框之外）。
- 三種模式在 1280×800 與 1440×900 都不用捲動。

## 0.10.7 — 2026-10-16

- **音軌改為上下堆疊**：監看舞台裡的原始 noisy（上）與輸出 output（下）改成全寬上下排列，兩軌時間軸對齊，方便上下比對；標籤疊在左上角，不佔寬度。
- **音軌顯示切換**：影像下方的工具列新增「波形／頻譜／兩者」segmented 控制，選擇存在 `localStorage`（`presence.trackView`）。隱藏的 canvas 不會重畫。
- 每種模式各自設定音軌高度（兩者模式最精簡），舞台寬度依視窗高度扣掉該模式的音軌高度計算，所以三種模式在 1280×800 與 1440×900 都不用捲動。

## 0.10.6 — 2026-10-16

- **監看舞台（monitor stage）**：相機影像（含字幕）和兩條音軌（原始 noisy／輸出 output，各為波形＋頻譜）合成一個獨立的深色框，音軌直接接在影像下方、中間沒有間隔。舞台寬度依視窗高度計算（`100vh − 250px`，維持 4:3），在 1280×800 與 1440×900 的筆電畫面不用捲動就能同時看到影像和兩條音軌。
- 其他內容移出舞台：ROI 編輯與說明放在舞台下方的「相機與 ROI」卡片；監聽、錄音、30 秒時間軸在「監聽・錄音・時間軸」卡片；狀態機、語音辨識、倒數等維持在右欄。
- 新增測試：舞台只包含影像、字幕和兩條音軌；腳本裡用到的每個元素 id 都存在（避免版面搬移時漏掉元素）。

## 0.10.5 — 2026-10-16

- **音軌移到相機正下方**：「乾淨音軌」面板（原始 noisy／輸出 output 兩組波形＋頻譜、時間軸、錄音與監聽）改放在左欄相機下方，右欄只留狀態、語音辨識、在場機率、倒數等，左欄不再留白。
- **語音辨識即時字幕**：開啟語音辨識時，在影像底部疊一行字幕。partial 逐字原地更新（灰字），句尾 final 變白字，新的一句淡入，final 後停留 2.5 秒淡出（只有 partial 沒有後續時 7.5 秒後淡出）。字幕放在 canvas 外，鏡像預覽時文字不會反過來；過長時保留最新的字。關閉語音辨識或 ASR 斷線時隱藏。原本的語音辨識面板保留。字幕以 `textContent` 寫入，不解析 HTML。

## 0.10.4 — 2026-10-16

- **狀態機改成軸上的節點**：ACTIVE → IDLE → MAYBE AWAY → AWAY → ENDED 排成一條軸，目前狀態的節點高亮（狀態色），最近一次轉換的來源節點以虛線標示、經過的線段以強調色加箭頭（往回轉換箭頭反向）。軸下方列出最近 6 次轉換（時間、來源 → 目標，最後一筆附原因）。
- **音軌依訊號分組**：「原始 noisy」一組＝波形＋頻譜，「輸出 output」一組＝波形＋頻譜，兩組分開並以 mint／periwinkle 色條區分；頻譜改為全寬。
- **臉部與身體 ROI 都能直接在影像上拖曳設定**：相機上方新增「編輯 ROI：臉部／身體／鎖定」選擇與「重設此 ROI」。只有選中的 ROI 可拖曳（避免重疊時抓錯框）：框內拖曳移動、四角縮放、框外拖曳畫新框；正在編輯的身體 ROI 即使姿勢偵測關閉也會顯示並提示。鏡像顯示時座標一律換算回原始相機座標儲存（新增純函式 `roiMath` 與測試）。數字輸入仍保留在設定中。
- 修正 `app.css` 裡 `--line:var(--line)` 的自我參照（讓 `--line` 變成無效值，邊框退回 currentColor），現在直接使用上游 token。

## 0.10.3 — 2026-10-15

- **真正貼齊 PureSound playground 的設計，不只是字體與色票。** HTML 改用 vendored 的 PureSound 元件 class（`.button` / `.button-primary` / `.panel.panel-pad` / `.panel-title` / `.metric` / `.segmented` / `.chip` / `.stat` / `.table` / `.toast` / `.toggle-row` / `.control` / `.range` / `.inspector` / `.eyebrow` / `.empty-state` / `.status-dot`），並載入完整的 `shell.css` 與 `screens-shared.css`。自訂樣式收到 `app.css`，只留相機、狀態機、倒數、scope 底色、追蹤名單、設定模組手風琴等 presence 專用佈局，不再重寫 button/panel/token。
- 版面：playground 式 page-header（標題＋說明＋actions）、workbench（相機 panel ＋右側 panel 堆疊）、設定改為 Inspector 抽屜（eyebrow 分節、segmented 預設、toggle 進階、field/control/range）。監聽改 segmented；指標改 metric；連線改 status-dot；toast / 錯誤橫幅用 playground toast 樣式。
- `docs/DESIGN.md` 改寫為元件對照表。測試確認 index 使用共用 class、且 `app.css` 不重新定義上游 token／component。

## 0.10.2 — 2026-10-15

- **UI 改用 PureSound playground 的設計系統**：直接載入 PureSound web 的 `tokens.css`、`base.css`、`components.css` 與 IBM Plex 字型（釘在與 `puresound-streaming` 相同的 revision `121bf62`，放在 `presence_demo/static/vendor/puresound/`，Apache-2.0／OFL，授權檔一併附上），不再維護自己的一份色票。App 自己的 CSS 只把 presence 的角色對應到上游 token。
- 版面改成 playground 的 shell：深色側欄（導覽、連線狀態、主題 自動／淺色／深色、版本）＋頁首（標題、說明、主要動作）＋右側面板欄。主要按鈕用 ink 色、強調色（橘）只用於焦點、選取和進行中；原始音訊＝mint、乾淨音訊＝periwinkle，與 playground 同義；波形、頻譜、時間軸、sparkline 放在常駐深色底上（同 playground 播放器）。設定抽屜、事件紀錄、錯誤橫幅、toast 套用 Inspector／toast 樣式。
- 主題預設改為「自動」（跟隨系統），仍可固定淺色／深色。
- `/static/vendor/` 下的公開 CSS／字型不需 token，其餘路徑仍受 token 保護；首頁顯示版本號。
- `docs/DESIGN.md` 改為記錄與 PureSound token 的對應；NOTICE 補上第三方 CSS／字型。
- 新增 `tests/test_design_tokens.py`（vendored 檔案與釘選 checksum、未重新定義上游 token、vendor 公開但 app 受保護）。
- 無頭 Chrome：深／淺色主畫面與設定頁沒有 JS 錯誤，文字對比全部符合 WCAG AA。

## 0.10.1 — 2026-10-15

- **修正：啟用姿勢後臉部追蹤與音訊停止**。伺服器原本在影像／音訊處理工作中直接 `await` 傳送給瀏覽器；慢的或經過轉送的連線（每個連線約 50 kB/s，兩個分頁加倍）會讓處理工作卡在傳送上，最後連線因 keepalive 被關閉；工作中的例外也不會被記錄。現在所有傳送改走每連線一個非阻塞的傳送佇列（Outbox：過舊的音訊與可丟的訊息會被丟棄並計數），state 訊息只在設定變更時附上設定。
- 工作（ticker、影像、音訊）由監督器執行：例外會記錄完整 traceback 並自動重啟；watchdog 偵測處理停滯，記錄所有執行緒的 stack dump、重啟工作，並在 UI 顯示錯誤橫幅。
- 共用偵測器鎖加上逾時（`PRESENCE_VISION_LOCK_TIMEOUT`），逾時會記錄 stack dump。
- **結構化日誌**（`docs/LOGGING.md`）：連線生命週期、設定變更（密鑰遮罩）、模組載入時間、每 10 秒心跳計數、狀態轉換與 session 事件、交叉確認計數、門檻警告。`PRESENCE_LOG_LEVEL`、`PRESENCE_LOG_FILE`（輪替）。不記錄影像、音訊、逐字稿或密鑰。
- UI 新增「事件紀錄」面板；`/debug/events`（需 token）。
- 新增 `tests/test_stall_0101.py`：音訊串流中途啟用姿勢、兩個同時連線、工作崩潰後重啟、慢連線不阻塞、`/debug/events` 保護、日誌遮罩。

## 0.10.0 — 2026-10-15

- **選用的錄音（評估效果用）**：「乾淨音軌」面板的錄音按鈕，預設關閉。三軌 raw（已補償 PureSound 延遲）／vi（PureSound 輸出）／out（影像交叉確認後）逐樣本對齊、同長度，16 kHz mono PCM16 WAV，加上 JSON（每 100 ms 判斷、狀態時間軸、設定快照），打包成一個 zip 下載到使用者的電腦。最長 10 分鐘，標題列有錄音中指示。錄音只在瀏覽器記憶體；伺服器只在錄音時多串流 vi 與對齊的 raw（WS `0x03`），不寫任何檔案。影像從不錄製。
- **設定頁重新設計**：依資料流分成 12 個模組（輸入 → 視覺偵測 → 姿勢 → 距離 → 追蹤 → 互動意圖 → 在場判斷 → 狀態機／逾時 → 音訊 → 語音辨識 → 效能 → 顯示）。上方是可點擊的資料流狀態條；每個模組最上面是啟用／模式，相依參數縮排且只在相關時顯示（例如 Kalman 參數只在 Kalman 模式），不適用的參數變灰並說明原因（例如距離關閉時跳過情況 3）。基本／進階切換（進階預設隱藏，改過的仍顯示）、每個模組的重設、新的預設組合「推薦新組合」（Kalman + 機率 + 意圖分數）與「低功耗」。設定鍵與匯出／匯入格式不變。
- **整體 UI 改版**：新的設計規範 `docs/DESIGN.md`（DESIGN.md 格式）；深色為主、可切換淺色；所有顏色改用 CSS 變數 token，圖表也讀同一組 token；數字用等寬字與 tabular figures；焦點框、減少動態效果。無頭 Chrome 檢查深／淺色的主畫面與設定頁：沒有 JS 錯誤、文字對比全部符合 WCAG AA。
- `docs/CONFIGURATION.md` 依模組重新整理，加上模組相依表。
- 新增 `tests/test_recording.py`、`tests/test_settings_ui.py`。


## 0.9.1 — 2026-10-15

修正遠端（經 HTTPS 反向代理）實測回報的兩個問題。

- **監聽延遲很高**：根因在瀏覽器的播放器。舊的播放佇列最多累積 25 塊（500 ms）才砍回 5 塊，網路一次卡頓後湧入的聲音就變成**永久延遲**（之後送達與播放同速，佇列不會消化），麥克風與喇叭的時脈漂移也會慢慢累積。
  - 新播放器：固定小 jitter buffer（累積 60 ms 才開始、目標 80 ms），超過 160 ms 立刻丟最舊的聲音回到目標，超過目標 + 20 ms 持續 0.5 s 就丟一塊（慢速追趕）；斷音時重新預先緩衝；線性內插重取樣；`AudioContext({latencyHint:'interactive'})`。
  - 伺服器：人聲分離改在專用的音訊執行緒池（不再和影像共用預設 thread pool），每塊只跳一次執行緒；積壓時一次處理整批；佇列上限從 1 s 降到 500 ms。
  - UI 新增「監聽延遲（擷取→播放）」：擷取 20 + 演算法 + 往返 p50/p95 + 播放緩衝 + 輸出裝置延遲，以及斷音次數、追趕丟棄量、伺服器佇列。
  - 模擬（60 ms 單程網路、每 20 s 一次 400 ms 卡頓、0.2 % 時脈差）：擷取→播放（不含演算法／輸出）p50 479 → 179 ms，最大 480 → 180 ms；在 node 中執行真正的 worklet 程式碼，卡頓後緩衝最多 117 ms（舊版 ≥ 400 ms）。伺服器端（本機 WS）往返 p50 6.5 → 6.9 ms、最大 23.8 → 11.6 ms，原本就不是瓶頸。
- **常常偵測不到臉**，三個根因：
  1. **`max_width` 預設 320 太小**：ROI 裁切後（640×480 畫面約 512 px 寬）縮到 320 px，臉高約 80 px 以下（約 1.5 m 以外的筆電鏡頭）就偵測不到。預設改成 640（實際不縮小，偵測 2.4 → 5 ms）；瀏覽器裡存過的 320 會自動升級一次。
  2. **多個連線共用同一個 YuNet 偵測器但沒有同步**：`cv2.FaceDetectorYN` 的輸入尺寸是可變狀態、不是 thread-safe，兩個分頁／裝置同時偵測會得到錯誤結果，甚至讓程序崩潰（重現：兩個執行緒同時偵測不同尺寸 → `double free or corruption`）。所有影像推論改成經過同一把鎖。
  3. **移動閘門會沿用「沒有臉」的結果**：畫面變化小（小臉或慢慢走進來）時最多 3 s 都不重新偵測。現在只有上一次**有**偵測到臉時才會跳過。
  - 已確認**不是**鏡像問題：送出的畫面不翻轉（鏡像只在 CSS），ROI 一律以原始相機座標儲存，加了回歸測試。
  - 資源監控新增「偵測率」（有臉的偵測次數／偵測次數，記憶體中的計數器）。
- 安全：uvicorn 的 WebSocket 握手 log 會印出 `?token=`，現在會遮成 `token=***`。
- 新增 `tests/test_face_misses_091.py`、`tests/test_player_js.py`（有 node 才執行）。


## 0.9.0 — 2026-10-15

- **新增 PureSound 串流人聲分離**（`audio_pipeline = puresound`；預設仍是 `energy_vad`，行為不變）。詳見 `docs/AUDIO.md`。
  - PureSound 可攜式 SDK（`sdk/python`，Apache-2.0）以**選用相依**安裝：`uv sync --extra puresound`（uv git 來源，公開 repo github.com/mcw519/puresound，固定 commit `121bf62`）。沒有複製任何 PureSound 程式碼，只有 `presence_demo/audio.py` 的 glue。
  - 模型 `dpcrn_curriculum_v2`（ONNX + manifest）用 `scripts/fetch_puresound_model.py` 從 PureSound clone 複製到 gitignored `models/`，並檢查 sha256。
  - 瀏覽器 AudioWorklet 擷取麥克風 → 16 kHz PCM16 → 每 20 ms 一個 WS binary 訊框；伺服器回傳修正後的乾淨音軌（binary）和每 100 ms 的 `audio` 中繼資料。
  - 乾淨 VAD（v2 沒有 VAD 輸出頭，用分離後訊號的能量 + 尾音）。
  - **影像交叉確認**每 20 ms 一次：漏音（有乾淨說話但沒人）、聲學誤判-超出範圍（所有人都 > `near_range` 1.0 m）→ 壓低、不算說話；聲學誤判-漏收（近距離有人、原始有說話、模型沒聲音）→ 多放原聲、算說話；影像不確定時只看聲音。
  - **進場規則**：乾淨音源下，session 只在「修正後說話開頭 + 近距離有人」時開始（`onset_entry_only`）。
  - NEW_GROUP／新 session／參與者換人時重置分離串流狀態（重新啟動 onset guard）。
  - UI：「乾淨音軌」面板（監聽 不監聽／原始／乾淨，預設靜音，耳機警告；原始與乾淨波形、電平；判斷標籤與次數；RTF、延遲），設定群組「乾淨音軌（PureSound）」。
- **選用的雲端語音辨識**（`asr_provider = assemblyai`，預設 `none`）：AssemblyAI Universal-Streaming v3 WebSocket，由伺服器連線；只在 session 進行中、預設只送校正後算說話的片段（其餘送靜音）；ENDED 時斷線；自動重連。API key 來自環境變數或 UI 密碼欄位，只存在伺服器記憶體、不寫 log、不回傳瀏覽器。逐字稿（partial / final、相對 session 的時間）只顯示在 UI。詳見 `docs/ASR.md`。**聲音會離開裝置**。
- UI：原始／乾淨頻譜、30 秒時間軸（原始 VAD／乾淨 VAD／算說話／判斷／狀態 + P(group)）、閘門狀態、影像與音訊的時間對齊（畫面訊息帶 `performance.now()`）、資源監控中的人聲分離 RTF／延遲。
- **開源準備**：改為 Apache-2.0（新增 `LICENSE`、`NOTICE`），文件改成中性用語，移除個人／私有環境的描述；PureSound 描述為公開的外部相依。
- 效能（x86 測試伺服器，單執行緒）：分離 RTF 0.22–0.27，演算法延遲 52–62 ms，WS + 計算 p50 6 ms；伺服器 CPU 影像 3.8 % → 影像 + 音訊 31.4 %。
- 輸出驗證：glue vs SDK、SDK vs PureSound `StreamingOrt` 最大差 0.0。
- 新增 `tests/test_asr.py`（mock AssemblyAI 伺服器、key 不外洩）。
- 新增 `tests/test_audio.py`（SDK smoke 與分塊不變性〔沒有 SDK 時跳過〕、四種交叉確認情況、影像不確定的退回、距離關閉、保持時間、NEW_GROUP 重置、乾淨 VAD、進場規則、WS 音訊來回不寫檔）。
- 新文件 `docs/AUDIO.md`；更新 ALGORITHM、CONFIGURATION、PERFORMANCE、LICENSES。

## 0.8.2 — 2026-10-15

- **修正**：balanced／audio priority + rules 在場模型下，在場但沉默的人讓狀態每 0.5 s 在 AWAY 和 ACTIVE 之間來回跳、session 永遠不會結束。
  - 從 AWAY 回來現在需要**新的證據**：說話開始、確認沒人後再出現，或進入 AWAY 時不在畫面上的新人。人一直待著不算。回來時重設靜默計時。
  - 因 AWAY 逾時而 ENDED 後，同一個人一直待著不會立刻開始新的 session（rules 和 hmm 都適用；hmm 以前會這樣不斷重開）。要說話，或離開後再出現。
  - 新的時間（有人在場但沉默，預設值）：balanced rules 7.5 s MAYBE／15 s AWAY／20 s ENDED；audio rules 15／30／35 s；balanced hmm 約 14.5／20.5／25.5 s；audio hmm 約 11.5／16.5／21.5 s。
  - vision 模式行為不變。
- 新增 `tests/test_away_return.py`：3 種 priority × 2 種在場模型的回歸測試，加上 score 意圖模式。

## 未發布（文件）

- `docs/ALGORITHM.md` 改寫成演算法架構總文件（v0.8.1）：端到端資料流、各層的公式和預設值、priority_mode 的影響、模式矩陣與建議預設、時間表、效能／隱私／授權、已知限制。所有數字都已經和程式碼預設值核對，時間表用模擬實測。
- 新增 mermaid 原始檔和 PNG：`docs/img/dataflow`、`state_rules`、`state_hmm`。
- 修正 `docs/PRESENCE_MODEL.md` 中 balanced／audio 靜默時的時間（依實測：audio 約 16.5 s AWAY，不是 20 s）。
- 記錄已知問題：balanced／audio + rules 模式下，在場但沉默的人會讓狀態在 AWAY 和 ACTIVE 之間來回跳（程式碼尚未修改）。

## 0.8.1 — 2026-10-14

- 網頁預覽新增「鏡像顯示」（預設開啟，存在 localStorage）：影像與疊圖（人臉框、身體框、關鍵點、ROI）一起左右翻轉成自拍視角；疊圖是換算座標後再畫，文字不會反過來。
- 送到伺服器的影像不翻轉，和機器人相機一致；ROI 一律以原始相機座標儲存，拖曳時由網頁換算。
- 已用 headless Chrome 以合成框驗證疊圖與影像對齊、拖曳座標換算正確。

## 0.8.0 — 2026-10-14

- **SFace 嵌入後端**：新增 `embed_backend`（`fp32`（預設）／`int8`／`npu`）、`npu_precision`（`int8`／`fp16`）和 `cos_match_int8`。
  - `int8`：opencv_zoo 官方的 `face_recognition_sface_2021dec_int8.onnx`，下載時驗證 sha256。
  - `npu`：用 LiteRT 執行 TFLite 模型，可以掛外部 delegate。
    - delegate 路徑只能用環境變數 `PRESENCE_NPU_DELEGATE` 設定，WebSocket 無法修改。
    - delegate 無法使用時改用 CPU XNNPACK，並在狀態中顯示實際使用的加速器。
- 新增 `presence_demo/embed_backends.py`（Umeyama 5 點對齊、OpenCV 和 TFLite 兩種後端）。
- `ensure_model`：支援 sha256 驗證，先下載到 `.part` 暫存檔再換名。
- `VisionPipeline(allow_download=False)` 會使用已經下載的模型。
- 新增 `scripts/export_sface_tflite.py`：可重現的 fp16／全 INT8 TFLite 匯出。
  - PReLU 合併成單一 op；
  - 用 CC0 人像在記憶體中校正；
  - onnx2tf 會改寫輸入檔，所以轉換複本。
- 新增 `scripts/faces_cc0.py`（只在記憶體中處理的 CC0 人臉）和 `scripts/eval_embed_backends.py`（各後端和 fp32 的精度一致性）。
- `benchmark.py --latency`：新增嵌入後端比較和模型本身的延遲。
- 網頁：新增「嵌入後端」設定；「資源監控」顯示實際使用的後端和加速器。
- 文件：新增 [`docs/NPU.md`](docs/NPU.md)；更新 LICENSES、CONFIGURATION、PERFORMANCE。

### 先前未發布的變更

- `scripts/benchmark.py` 新增兩種模式：
  - `--latency`：各階段 p50／p95／max（設定 × 輸入）；
  - `--ws`：本機 WebSocket 端到端延遲（暫時啟動伺服器，量完自動關閉）。
- `PresenceTracker.timings`：記錄最近一次 update 的 tracker／intent／presence 耗時（ms）。
- PERFORMANCE.md 新增 v0.7.0 的每幀延遲結果。

## 0.7.0 — 2026-10-14

- **Layer 3 互動意圖**：新增 `intent_model`（`rules`／`score`，預設 `rules`，行為不變）。
  - `score`：每個 track 有一個 0–1 的意圖分數，用實際 dt 做 EMA。線索有：
    - 面向（YuNet 五點的正臉分數＋肩膀方向）；
    - 對話距離（距離估計關閉時自動略過）；
    - 停留時間；
    - 走近／走遠；
    - 是否在橫向路過；
    - 是否接近 ROI 中央；
    - 可歸屬的說話（弱，權重依 priority_mode）。
  - 分數還會經過一個「面向」注意力閘門。
  - 加入：意圖 ≥ `join_threshold` 持續 `join_seconds`。session 也要有人達到這個條件才開始。
  - 降為旁觀者：意圖 < `drop_threshold` 持續 `drop_seconds`；最後一位參與者不會被降級。
  - NEW_GROUP 的新候選人也要達到加入條件。
  - hmm 的群組 P 只看參與者；可以選擇依意圖加權（`intent_weighted_presence`）。
- 新增 `presence_demo/intent.py`。
- `Detection.frontal`：視覺管線從 YuNet 五點計算正臉分數。
- state 每個 track 新增 `intent`、`intent_cues`、`role`（`participant`／`bystander`）。
- 網頁：每個人的意圖條和「參與者／旁觀者」標籤；新增「互動意圖」設定分組。
- 評估：新增 6 個情境，以及「誤加入」「加入時間」指標；比較版本加入 `kalman+hmm+score`。
- 新增 [`docs/INTENT_MODEL.md`](docs/INTENT_MODEL.md)。

## 0.6.0 — 2026-10-14

- **Layer 2 在場機率**：新增 `presence_model`（`rules`／`hmm`，預設 `rules`，行為不變）。
  - `hmm`：每個 track 一個兩狀態（在場／離開）Bayes 濾波器，用對數勝算加「每秒證據 × 實際 dt」計算，和 fps 無關。
  - 證據來源：偵測信心、kalman 預測（較弱）、低信心幀（接近中立）、轉身＋遠離、超出距離。
  - 聲音是全域的弱證據，權重依 `priority_mode` 不同，不會歸到任何人身上。
  - 群組 P 取參與者中的最大值，再用遲滯門檻決定 ACTIVE／MAYBE_AWAY／AWAY。
  - hmm 模式的 NEW_GROUP 條件：舊參與者 P < `p_away`，而且新候選人的 P 穩定在 `p_active` 以上。
  - 新增參數：`leave_halflife`、`detect_gain`、`lowconf_factor`、`p_active`、`p_maybe`、`p_away`、`away_hold_seconds`。
- 新增 `presence_demo/presence_model.py`。
- state 新增 `p_group`、`p_tracks`；每個 track 新增 `p`。
- 網頁：新增「在場機率」卡片（群組 P 曲線和門檻線）、每個人的機率條，以及「在場機率」設定分組。
- `scripts/eval_tracking.py`（別名 `scripts/eval_presence.py`）：
  - 改為比較 legacy+rules、kalman+rules、kalman+hmm 三個版本；
  - 新增 6 個情境：低信心閃爍、ROI 空＋電視聲、長時間安靜、慢慢離開、短暫走開、遠處路人；
  - 新增「狀態切換/分」指標。
- 新增 [`docs/PRESENCE_MODEL.md`](docs/PRESENCE_MODEL.md)。

## 0.5.0 — 2026-10-14

- **Layer 4 追蹤**：新增 `tracker_mode`（`legacy`／`kalman`，預設 `legacy`，行為不變）。
  - `kalman`：每個 track 一個等速 Kalman 濾波器（用實際時間戳記計算 dt），Mahalanobis 門檻篩選，綜合 IoU、移動、臉部 cosine 和衣服顏色的分數，用匈牙利演算法配對。
  - 短暫遮擋時沿預測位置延續（`coast_seconds`、`coast_present_conf`）。
  - 新增參數：`gate_mahalanobis`、`appearance_min_sim`、`appearance_ema`。
- 新增 `presence_demo/kalman.py`（Kalman、匈牙利演算法、Bhattacharyya 相似度，純 numpy，不需要 SciPy）和 `presence_demo/appearance.py`（軀幹 HSV 直方圖，不需要模型權重，只在記憶體中計算）。
- snapshot 每個 track 新增 `visible_kind`、`vis_conf`、`dwell_seconds`、`velocity`。
- 網頁：新增「追蹤」設定分組；預測中的框用虛線畫，並顯示可見信心、停留時間和速度。
- 新增 `scripts/eval_tracking.py`（在記憶體中產生合成情境，比較兩種模式）和 [`docs/TRACKING.md`](docs/TRACKING.md)。
- 新增 `tests/test_kalman.py`；更新 ALGORITHM、CONFIGURATION。

## 0.4.1 — 2026-10-14

修正 [`docs/ALGORITHM.md`](docs/ALGORITHM.md) 稽核中發現的問題（每項都有測試，見 `tests/test_fixes_041.py`）：

- `empty_since` 記錄第一張空幀的時間，不再假設 1 fps（自適應 fps 下 `empty_timeout` 不會再提早或延後）。
- 新增 `body_min_conf`（預設 0.4）：只有身體的偵測用這個門檻判斷「有信心」，看不到臉時也能算有人。
- 超出 `leave_distance` 的人不會從 ENDED 開始 session，也不會觸發 NEW_GROUP（修正每 10 秒重新開始的循環）。
- 新增 `speech_join`（`auto`／`off`／`single_candidate`／`all`）：vision 模式下說話預設不會讓人立即加入；balanced／audio 只在唯一候選人（臉在 ROI、正面、在對話區內）時立即加入。
- vision 模式：單靠說話不會從 AWAY 恢復（包括影像不確定時），和文件一致。
- 修正 `present_timeout` 的說明（有人時的靜默逾時；預設 15 s 比 `base_timeout` 20 s 短）。
- 小修正：新 session 重設低信心計數；只有低信心的幀不算「舊群組消失」；snapshot 新增 `face_roi`／`pose_roi`（保留 `roi` 別名）。
- 新增 `docs/ALGORITHM.md`：依程式碼整理的完整偵測邏輯。

## 0.4.0 — 2026-10-13

- **授權稽核**：新增 [`docs/LICENSES.md`](docs/LICENSES.md)，列出每個模型、套件和測試資料的授權、來源和商用風險。
- **姿勢模型改成 MoveNet MultiPose Lightning**（Google，Apache-2.0，TFLite float16，用 `ai-edge-litert` 執行），最多 6 人；移除 YOLOv8n-pose（AGPL-3.0）和相關下載程式。姿勢成本約為原來的 1/5。
- **測試圖片**：`tests/data/face.jpg` 從 lena.jpg（授權不明）換成 Cleveland Museum of Art 釋出的 CC0 圖片（Julia Margaret Cameron，1867）。
- 新增環境變數 `PRESENCE_FACE_EMBEDDER=hist`：不載入 SFace，改用 HSV 直方圖（SFace 訓練資料的商用風險見 LICENSES.md）。
- 新增 `tests/test_movenet.py`；重新量測基準測試。

## 0.3.0 — 2026-10-13

- **姿勢偵測（選用，`pose_enabled`，預設關閉）**：可替換的 `PoseDetector` 介面；預設是 YOLOv8n-pose（ONNX，OpenCV DNN CPU，第一次使用時自動下載）。輸出身體框、關鍵點（鼻、眼、耳、肩、髖）、朝向（正面／側身／背對）和信心值。
- **融合**：臉和身體配對；看不到臉時用身體判斷有人，並用 IoU（含身體框）延續身分；轉身而且正在遠離時，提早進入 MAYBE_AWAY（`turned_away_early`、`turned_away_seconds`、`recede_ratio`）。
- **距離估計（選用，`distance_enabled`）**：用臉寬或肩寬加上鏡頭視角估計距離，EMA 平滑；對話區（`zone_min`／`zone_max`）、離開距離 `leave_distance`、靠近／遠離趨勢；網頁一鍵校正（WS `calibrate`）。
- **分開的 ROI**：`face_roi`（取代 `roi`，舊設定自動轉換）和 `pose_roi`，判定規則 `roi_rule`；兩個 ROI 都能在畫面上拖曳（藍色／紫色），也能各自隱藏。
- 網頁：新增「姿勢」「距離」設定分組；畫面上畫出骨架、身體框、距離和趨勢標籤、對話區提示；資源監控顯示姿勢耗時、呼叫次數和模型。
- 基準測試新增姿勢和距離的比較；更新 `docs/PERFORMANCE.md`。
- 新增 `presence_demo/body.py` 和 `tests/test_pose_distance.py`。

## 0.2.0 — 2026-10-13

- 新增效能／省電最佳化（`presence_demo/optimizer.py`），每項都可在設定的「效能」分組個別開關：
  - 移動偵測閘門（`motion_gate`、`motion_threshold`、`force_detect_seconds`）
  - Embedding 節流（`embed_throttle`、`embed_interval`、`embed_move_thresh`）
  - 自適應 fps（`adaptive_fps`、`fps_low`、`fps_high`），伺服器在 state 訊息中回傳 `recommended_fps`
  - ROI 裁切＋縮小（`roi_crop`、`crop_margin`、`max_width`）
- 資源監控新增：跳過比例、embedding 計算／重用次數、建議 fps、移動偵測耗時。
- 新增 `scripts/benchmark.py`（記憶體中合成畫面，不寫入影像）和 `docs/PERFORMANCE.md`。全部開啟時 CPU 約減少 63–69%。
- 設定驗證支援布林值；`fps_low` 不會大於 `fps_high`。
- 隱私：解碼改用 `decode_jpeg()`，在記憶體中完成，處理後立即釋放。

## 0.1.0 — 2026-10-13

首個版本。

- 核心：`PresenceTracker` 群組追蹤（IoU + cosine greedy 配對、EMA 模板、參與者加入與寬限期）、遲滯與低信心降權、狀態機 ACTIVE/IDLE/MAYBE_AWAY/AWAY/ENDED、NEW_GROUP。
- 判斷優先 `priority_mode`：vision（預設）／balanced／audio，以及 `present_max_silence`。
- 影像：OpenCV YuNet + SFace，備援 Haar + HSV 直方圖；模型自動下載。
- 伺服器：FastAPI + WebSocket，每個連線獨立 tracker；設定驗證與範圍限制；`PRESENCE_TOKEN` 權杖；最新畫面槽位（丟棄舊畫面）；資源與延遲統計。
- 網頁：開始／停止（關閉鏡頭與麥克風、切換分頁自動暫停）、ROI 拖曳、設定抽屜（分組、說明、預設組合、localStorage、JSON 匯出入）、狀態機、倒數、參與者、轉換紀錄、資源監控與 sparkline；支援手機。
- 預設值：`present_timeout` 15 s、`empty_timeout` 5 s。
- 工具：uv（pyproject + uv.lock）、ruff、pytest。
