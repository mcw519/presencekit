[English](../real-acceptance.md) · **繁體中文**

# 真實 Chrome 裝置與 Jev 驗收 runbook

本 runbook 用於在 Apple M4、Chrome、本機 server 與實體鏡頭／麥克風上執行有人參與的驗收，並在另行完成知情同意與費用確認後，驗證 Jev 跨 session re-ID。它是操作流程，不代表任何真實裝置或 Jev 案例已通過。合成測試、mock、headless Chrome 和合成音訊結果不能代替本 runbook。

**目前交付狀態：**此文件已備妥；本輪沒有啟動裝置或呼叫 Jev。使用正式離線準備工具建立驗收資料夾：

## 1. 執行前準備

### 離線準備資料夾

在 repo 根目錄執行：

```bash
.venv/bin/python scripts/prepare_real_acceptance.py --output-dir artifacts/acceptance/real-preparation
```

檢查產生的 `profile.json`、`preflight.json`、`manual-results.json` 和 `README.md`，不要把任何媒體或可識別資料放入報告資料夾。確認有可用的本機模型及 PureSound runtime；準備工具應離線執行且不得下載模型或連到 Jev。若工具不可用、模型缺少或輸出欄位不完整，記為 `not_run` 並先停止，不要手動假稱完成準備。

### 固定環境與操作限制

- 記錄驗收日期、macOS 版本、M4 機型／記憶體、Chrome 版本、repo commit、Python／SDK 版本、模型版本、PureSound 版本、設定 profile 版本及 server 啟動方式。這些欄位由操作人填寫；不要猜測。
- 一次只使用一條 active WebSocket 連線；Chrome 開一個驗收分頁，其他人不要連入同一 server。場景為 1–3 位已同意參與者。
- server 綁定 loopback：

  ```bash
  uv run presencekit --host 127.0.0.1 --port 8000
  ```

  這是 README 所列 `presencekit` entry point 與 `server.main` 的正式參數；預設也是 `127.0.0.1:8000`。不要改綁 `0.0.0.0`。先確認 port 沒被其他 server 使用；若需更換 port，記錄實際命令與 URL。
- 使用 Chrome 開啟 `http://127.0.0.1:8000`，人工確認頁面顯示的版本符合本輪 repo。按「開始」時才由操作人核准瀏覽器相機與麥克風權限。
- 準備匿名場景卡：參與者只用 `A`、`B`、`C`；記下操作時間點與預期，不記姓名、臉部描述、聲音特徵或原始 conversation/session/track ID。
- 不錄製／匯出影片、臉部裁切、音訊、逐字稿、原始 `/debug/events` 或含識別碼的訊息。需要證據時只記匿名案例代碼、人工操作時間、通過／失敗／未執行、允許的數值摘要與錯誤類別。

### 載入固定 presence profile

準備工具應產生可由設定抽屜匯入的 profile。於開始前用「設定 → 匯入 JSON」載入，核對以下設定鍵；版本不同時以匯入後實際值為準並記錄差異。這組 PoC 驗收 profile 與「推薦」預設組合不同，不能用推薦組合替代。

| 設定鍵 | 驗收值 |
|---|---|
| `tracker_mode` / `presence_model` / `intent_model` | `kalman` / `hmm` / `score` |
| `join_seconds` / `grace_seconds` | `1.5` / `10` |
| `face_roi` / `pose_roi` / `pose_enabled` | `[0.08, 0.02, 0.92, 0.98]` / `[0, 0, 1, 1]` / `true` |
| `fps` / `adaptive_fps` / `fps_low` / `fps_high` / `pose_interval` | `2` / `true` / `0.5` / `2` / `2` 秒 |
| `audio_pipeline` / `asr_provider` | `puresound` / `none` |
| `reid_enabled` / `reid_consent` / `reid_show_crops` | `false` / `false` / `false` |
| `reid_ttl` / `reid_max_refs` / `reid_decision_timeout` | `600` 秒 / `3` / `10` 秒 |

「匯入 JSON」會載入其餘設定；另逐項確認未列出的效能、顯示、ASR 與 Re-ID 參數沒有改變本輪預期。特別確認 `adaptive_fps`、`fps_low`、`fps_high` 的實際 profile 值，並寫入紀錄，不要憑表格推定。停用 re-ID 時仍保留其 TTL、容量和期限設定，供之後 Jev 階段核對。

## 2. 第一階段：裝置驗收，雲端保持關閉

確認 `reid_enabled=false`、`reid_consent=false`、`asr_provider=none`；不得在此階段開啟雲端。用 `GET /api/reid_status` 確認 key 狀態；若已設定，從 UI 清除目前程序中的 key，再開始裝置案例。重新整理之前先匯入 profile，然後啟動一條連線並等待模型暖機。每個案例都從乾淨、已記錄的起始狀態開始；案例不符合預期時記 `failed`，不得為了通過而放寬門檻或改參數。設定一旦更動，本輪即記為新 profile，受影響案例需重跑。

| 案例 | 操作與計時 | 預期結果 | 證據／停止條件 |
|---|---|---|---|
| `device.quiet_120s`、`device.quiet_120s_2`、`device.quiet_120s_3` | 分別讓 1、2、3 位參與者加入並保持可見，接著全員安靜 120 秒。 | 沉默本身不會令 session `ENDED`。 | 每種人數各記開始／結束時間及狀態。提前 `ENDED` 即停止該案例並記 `failed`。 |
| `device.occlusion_3s`、`device.turned_head_3s` | A 已加入後，分別遮住臉或轉開約 3 秒，再恢復。 | 不因短暫不可辨識而誤結束已加入參與者。 | 記案例、時長、結果；3 秒內誤結束即 `failed`。 |
| `device.last_leave_20s` | A 已加入後完全離開鏡頭及互動區，記下離開時點。 | tracker 在 20 秒內到 `ENDED`。 | 記離開與 `ENDED` 的人工時間差；超過 20 秒或未結束即 `failed`。 |
| `device.remaining_1`、`device.remaining_2`、`device.remaining_3` | 依序測 1、2、3 位參與者；至少 2 人的案例先全員加入，再逐一讓人離開，同時至少留一位已加入者。 | 只要有任一已加入參與者留下，session 就不結束。單人案例確認一位在場時不結束。 | 每個人數與離開順序各記一列；剩餘者仍在但 session `ENDED` 即 `failed`。 |
| `device.bystander_not_extend`、`device.background_speech_not_extend` | 已加入參與者離開後，分別讓未加入路人經過，以及播放／提供背景 speech。 | 路人或背景 speech 單獨存在，不得把未加入者當成參與者或延續 conversation；最後一位離開後符合 `ENDED` 時限。 | 只記匿名場景和狀態；錯誤加入／延續即 `failed`。PoC 不具 TV 視覺語意分類，不能把「辨認出電視」列為通過。 |
| `device.sensor_interruption` | A 已加入時停止影格輸入或中斷瀏覽器媒體軌 25 秒，再恢復。 | sensor interruption 與人離開分開，不記成正常參與者離場。恢復媒體後依傳輸生命週期可能開始新 session。 | 記中斷及恢復時間、sensor 狀態；中斷被當作人已離開即 `failed`。 |
| `device.same_page_reconnect` | 保持同一載入中的分頁，造成 WebSocket 傳輸中斷後自動重連；觀察 session 計數。 | 恢復為新 session；有效頁面記憶體可供後續 re-ID，且不把斷線本身記成人離開。 | 記是否重連、session 是否更新、狀態連續性。不可檢視或複製 page capability。 |

### 30 分鐘效能 gate

執行 `device.normal_load_30min`，在固定 profile、已暖機、持續正常互動下量測 30 分鐘。必須同時符合：每秒取得的新鮮 `rtf_recent` 樣本之近 5 秒 RTF p95 ≤ `0.5`、audio dropped `0`、未捕捉 server error `0`、音訊佇列不持續成長、最後 10 分鐘平均 RSS 減最初 10 分鐘平均 RSS ≤ `30 MiB`。

目前 UI 沒有量測按鈕；recorder 由操作人手動載入，讀取既有 WebSocket 數值訊息，不呼叫 `getUserMedia`、不送網路請求、不修改設定，也不保存音訊、圖像、conversation ID 或 raw events。正式頁面先按「開始」並確認裝置串流後，在同源頁面的 Chrome DevTools Console 輸入：

```js
await import('/static/js/real-acceptance-recorder.js')
RealAcceptanceRecorder.start({seconds:1800,warmup:10})
```

暖機 10 秒後開始每秒取樣，量 1,800 秒後自動停止。不要將 Chrome 分頁切至背景；若斷線、capture stop 或計數器重設，continuity gate 會失敗。完成後在 Console 執行 `RealAcceptanceRecorder.report()` 檢視數值報告，並以 `RealAcceptanceRecorder.download()` 下載 `real-acceptance-metrics.json`。提前停止可用 `RealAcceptanceRecorder.stop()` 保留數值紀錄，但不能通過 30 分鐘 gate。開始下一次量測前先下載本輪結果，避免記憶體中的紀錄被取代。樣本覆蓋率需 ≥95%，每個首尾 RSS 10 分鐘視窗至少 570 筆樣本。

Recorder 的 `observed_errors` 是瀏覽器可見的錯誤數，不等於 server 未捕捉錯誤的權威稽核。`manual_checks.server_uncaught_errors` 會維持 `unverified`；操作人須只核對專用 server console 的錯誤數，將數值手動填入紀錄表。沒有獨立證據時該 gate 保持 `pending_real`。Jev event 中的呼叫、token、latency 與 model 只是觀察資料，不是完整帳單或全域預算。報告頂層狀態一律是 `pending_real`，須人工核對；全部數值 gates、人工 server error 核對與參與者 ground truth 都符合，才可驗收；`thirty_minute_coverage` 單項通過只代表量測時間與連續性，不能視為整體通過。

若 RTF 超標、音訊丟包、錯誤增加、佇列持續成長或裝置過熱／失穩，立即停止效能案例，安全停止 media，再記匿名指標與最後狀態。不要保留音訊／影像來「之後再分析」。

## 3. Jev 前置核准與設定

只有裝置階段完成且負責人核准後才進入。操作人應在任何臉部裁切可能送出前：

1. 向每位參與者說明此為跨 session 臉部比對，裁切會送到第三方 Jev（Typesafe）服務，並取得適用於這次驗收的明確知情同意；也須處理旁觀者可能入鏡的情況。
2. 由服務／法務／隱私負責人確認 Typesafe/Jev 適用的資料處理、保存、刪除與使用條款；runbook 不推定供應商保留時間、刪除保證或辨識 FAR。未確認條款時不得送出影像。
3. 由預算負責人填寫每次服務呼叫價格、幣別、核准上限與負責人。價格未知或沒有核准上限時，不得啟用雲端。
4. 先確認本機 server 僅有本次驗收的連線。`POST /api/reid_reset` 會清除**所有頁面／連線**的參考臉；只有確認沒有其他使用者 session 後才能呼叫。一般清理優先用 UI 的「清除參考臉」，其本身也會清除伺服器所有連線的 refs。
5. 僅在本機 server 設定面板輸入 Jev key；不要貼到 terminal、shell 命令、聊天、紀錄表或截圖。`GET /api/reid_status` 只回報 `set` 與來源資訊，不回傳 key，可用來確認是否已設定。若 key 來自 `TYPESAFE_API_KEY` 環境變數，UI 清除不會移除環境變數，結束時必須停止使用該環境變數的專用 server。
6. 於設定匯入同一 presence profile 後，設定 `reid_provider=jev`、`reid_enabled=true`、`reid_consent=true`、`reid_show_crops=false`、`reid_ttl=600`、`reid_max_refs=3`、`reid_decision_timeout=10`。`reid_consent` 只是應用程式開關，不是同意紀錄；真人同意須由操作人依組織流程另行確認。確認 `asr_provider=none`。

最多 3 段歷史 conversation、每段最多 3 位參與者參考、每個新 session 最多 3 位查詢者，理論上最多 27 次 pairwise 呼叫／輪；不確定分數的重試可能將上界提高至 54 次。實作為序列比較且整體期限 10 秒，實際能否完成須在已核准預算與正常服務條件下驗證。這些限制是每個 driver／連線的限制，重連會重設，不能當成跨 session 或全域 hard cap。測試全程由操作人逐筆累計呼叫數與估算費用；達到事先核准上限立即停用 Re-ID 並停止測試，不能提高 timeout 來掩蓋延遲。

## 4. Jev 驗收矩陣

每個案例用匿名 A/B/C 與受控動作執行。每個新 conversation 都由新 session 立即開始；只能在明確、唯一的辨識結果下 handoff。每案記 setup、開始時間、結果時間／耗時、handoff 布林、呼叫數、分數（若 UI／紀錄格式提供）、費用估算、失敗類別。不得記錄或匯出裁切、臉、姓名、原始 ID 或 raw events。

| 案例 | Setup 與操作 | 通過條件 | 必須停止／失敗條件 |
|---|---|---|---|
| `jev.same_person` | A 建立並正常結束歷史 conversation；同頁保持在 600 秒內，新 session 由 A 單獨加入。 | 新 session 先建立 provisional conversation；明確唯一匹配須在 10 秒內 handoff 回 A 的歷史 conversation。 | 已確認 A 在場且影像可辨識，但 10 秒內未 handoff，記 `failed`；服務不可用或缺少可靠證據記 `not_run`。錯誤／歧義 handoff 立即停雲端並記 `failed`。不可延長 deadline。 |
| `jev.different_person` | 歷史只含 A；在 TTL 內用 B 開新 session。 | 不 handoff，B 留在新的未關聯 conversation。 | B 接回 A 的歷史 conversation 即 `failed`，立刻停止該輪。 |
| `jev.member_overlap` | 歷史 conversation 由 A+B 組成；新 session 為 B+C。 | 唯一辨認出 B 時，handoff 回 A+B 的 conversation；這不代表 C 已被辨識。 | handoff 到其他 conversation、將 C 當作已辨識、或唯一結果後 10 秒仍無決定，分別記失敗／逾時。 |
| `jev.ambiguous_overlap` | 建立至少兩個歷史 conversation，使新 session 中的查詢者分別對應不同歷史；新 session 同時包含這些人。 | 無 handoff，保持 provisional conversation。 | 只要發生任何 handoff 就是錯誤關聯，立即停止，記 `failed`。 |
| `jev.unknown_timeout` | 在同意範圍內使用低品質姿勢或無可比對參考，記錄從 session 開始到最終決定的真實時間；不得以陌生人的臉冒充已同意樣本。 | 10 秒內 fail closed：無 handoff，session/audio 繼續。 | 無清楚可同意樣本、操作不安全或疑似第三方影像時立即停止；任何不確定結果 handoff 或 10 秒後仍阻塞即 `failed`。 |
| `jev.service_error` | 由服務負責人核准後，以可控方式驗證錯誤／逾時 fail closed；不以破壞 production key 模擬。記錄真實耗時。 | 10 秒 deadline 內錯誤或逾時後無 handoff，session/audio 不阻塞。 | 10 秒後仍等待、影音阻塞或錯誤時 handoff，記 `failed` 並停止後續呼叫。不可調大 timeout。不可安全觸發或服務不可用時記 `not_run`。 |
| `jev.revoke` | 一個已同意案例比較中，參與者撤回同意；以 UI 關閉 `reid_consent`，確認比對取消並清除顯示，再檢查延遲結果。 | 臉部 refs／縮圖清除；撤回後即使結果晚到也不得 handoff；session 繼續。 | 無法確認停止、仍有舊 crop／handoff，或需要保存媒體才能確認時，立即停用並記 `failed`。 |
| `jev.ttl_600s` | 以已同意 A 建歷史 conversation，正常結束後等候 600 秒再以 A 開新 session。 | 舊 refs 到期；不應 handoff 到過期歷史。 | 600 秒後仍續接舊歷史即 `failed`；勿用 reset 前資料混跑。 |
| `jev.max_3_history` | 依序建立並正常結束 A→c1、B→c2、C→c3。c1–c4 是操作人臨時匿名標籤，不是 repo conversation IDs。接著以 A+B 開新 session：兩人分別命中不同歷史，應保持歧義並建立 provisional c4；正常結束 c4 後，歷史上限應淘汰最舊 c1，留下 3 個歷史。 | 用核准介面提供的安全數值確認 `history_conversations=3`，並確認 c4 建立前沒有 handoff。A 或 B 之後可能匹配保留的 c4；這不能用來判斷 c1 是否仍在。若沒有安全的歷史數量／淘汰證據，記 `not_run`。 | 過程中任何歧義 handoff、歷史數量超過 3、或 active refs 被逐出，記 `failed`。不可把 A 後續匹配 c4 解讀成 c1 未淘汰，也不可用 A 未匹配證明已淘汰。 |

### 取得同意後的清理案例

`device.refresh_clears` 與 `device.server_restart_clears` 不能在雲端關閉、參考歷史為空時執行，空集合變空集合不是清理證據。先完成 Jev 前置核准、取得參與者同意並建立**非空**參考歷史，再分別重新載入頁面及重啟專用 server。預期有效 refs 被清除，記錄匿名布林結果，不保存 IDs 或影像。`manual-results.json` 的 prerequisite 未滿足時保留 `pending_real`／`not_run`；不能在裝置階段標成 pass。

至少 Jev 案例符合固定條件，且每次雲端呼叫都在費用核准內，才可將該項控制流程標記為通過。這仍不能推論一般使用者的辨識準確率、FAR/FRR 或供應商資料處理品質；那些需要另行定義、具同意且統計設計充分的評估。

### 服務商與計費來源

以下 Typesafe 官方頁面於 **2026-10-23** 確認；實跑前須重新核對頁面與適用帳戶合約。[Images API](https://docs.typesafe.ai/images) 標示為實驗性／Dreamer-ready，且需要 `Danger-Zone` header；這不是正式產品品質保證。[Models 與 rates](https://docs.typesafe.ai/models) 說明依 input tokens 計費及組織 rate limit；實際價格與限額須查看操作帳戶 console 和合約。[System One API](https://docs.typesafe.ai/api/system-one) 回應提供 `model` 與 `usage.input_tokens`，可作為這次實際模型及 input 使用量的依據。[提交資料如何使用或保存](https://help.typesafe.ai/articles/5974925716-how-is-submitted-data-used-or-retained) 和 [Typesafe 隱私政策](https://typesafe.ai/legal/privacy-policy) 中「不拿來訓練」不等同「不保存」或零資料保存保證；公開頁面未提供固定保存天數。請由負責人確認當前合約與條款。

## 5. 中止與清理

出現任何錯誤 handoff、同意撤回未生效、費用達上限、第三方影像、服務條款不清、key 暴露、媒體／session 阻塞或裝置失穩時，立即關閉 `reid_enabled` 與 `reid_consent`，停止測試並停止專用 server。確認無其他使用者連線後，再用設定中的「清除參考臉」或 `POST /api/reid_reset` 清除所有連線 refs；API 回傳計數不應被當成身分資料保存。清除 UI key 會重設目前程序中的 key，包括由環境變數載入的 key，但不會移除啟動環境中的 `TYPESAFE_API_KEY`；停止該程序，並確保之後需保持無 key 的 server 啟動環境已移除該變數。最後撤銷 Chrome 的本機相機／麥克風網站權限，關閉驗收分頁。

只保留匿名數值摘要及每案 `passed`、`failed`、`not_run`、`pending_real` 狀態。任何未量到數值、未做、工具欄位缺失或證據不足者都留在 `not_run`／`pending_real`；不可用合成結果替代。驗收結論須由負責人檢視數值、設定 profile 和同意紀錄後確認。

## 記錄表範本

```text
日期／時區：
操作者代碼：
repo commit：                 profile hash／版本：
macOS／M4／記憶體：          Chrome：
Python／PureSound SDK／模型：
server 啟動參數／URL：        連線數：
場景代碼（只用 A/B/C）：
案例 ID：                     結果：passed / failed / not_run / pending_real
setup 與預期：                 實際結果摘要：
起始／結束時間或耗時：        允許的數值：
RTF recent-5s p95：           drops：           errors：
queue 趨勢：                  RSS 首 10 分平均：   RSS 末 10 分平均：
Jev 呼叫數／估算費用／核准上限（僅 Jev 階段）：
停止原因／待處理：
```

延伸閱讀：[PoC 驗收總規格](acceptance.md)、[Re-ID 設計與保存方式](reid.md)、[隱私說明](privacy.md)、[設定參數](configuration.md)。
