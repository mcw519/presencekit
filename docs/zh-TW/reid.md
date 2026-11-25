[English](../reid.md) · **繁體中文**

# 跨 session 重新辨識

Re-ID 可將新 session 關聯到最近的一段 conversation。此功能為選用，必須同時開啟 `reid_enabled` 並取得明確的 `reid_consent`。目前 PoC 以一條本機 M4＋Chrome 連線及最多三位參與者為目標。Jev 服務會比對人臉裁切；使用真實後端時，生物特徵影像會離開裝置。

## Session 與 handoff 行為

每個新 tracker session 都會立即取得暫用的隨機 `conversation_id`，正常執行在場與音訊處理。Re-ID 在背景比對。明確且無歧義的比對成功後，發出 `handoff`，提供原暫用 conversation ID 與選定的歷史 conversation ID。原 session ID 和事件歷史都會保留。比對失敗、不確定、有歧義、過期或服務不可用時，新 conversation 保持未關聯，session 繼續正常運作。

每個 session 最多保留三位已加入參與者中品質最佳的人臉裁切。正常進入 `ENDED` 時，這些參考臉會寫入已結束的 conversation 記憶項目。下一個 session 開始後，系統會把目前可用的參與者臉部逐一和所有符合條件的歷史參考比對。只有命中結果指向唯一一個歷史 conversation 才能 handoff。若符合門檻的命中分屬多個 conversation，就判為有歧義，不 handoff。只認出一位舊成員可以延續群組 conversation；這不代表新 session 的所有參與者都已辨識。

從新 session 開始起算，整體比對期限是**真實時間 10 秒**。`AWAY` 等在場狀態不會暫停期限。系統最多等待 `reid_face_wait` 取得第一張清楚的人臉；分數不確定時可用新畫面重試一次，但仍受整體期限限制。逾時、沒有清楚人臉、不確定分數、呼叫次數上限、缺少金鑰、斷路器開啟或服務錯誤，都會讓新 conversation 保持未關聯。Jev 的辨識品質與延遲，仍須使用取得同意的受測者獨立驗收；mock 只驗證應用程式流程。

可選擇 `reid_model=jev-1.14.0` 固定模型版本。開啟 `reid_batch` 後，每張當前人臉會用一次 HTTP 請求比對所有符合條件的參考臉。預設每對人臉各發一次請求。兩種模式都要收到完整的有效分數才會做出續接判斷。每個請求計入一次呼叫額度，每筆回覆只計一次 input tokens。批次失敗時，暫用 conversation 保持未關聯。

伺服器送出的 `handoff` 含有 `session`、`conversation_id`、`previous_conversation_id`、`decision`、`resumed_from_session`、`score`、`reason` 和 `calls`。`conversation_id` 是不透明的隨機識別碼；本程式不保存對話內容。

## 頁面延續與到期

瀏覽器在 JavaScript 記憶體中建立密碼學安全的 128-bit 隨機頁面 capability。開啟 WebSocket 時，會和 `presence-v1` 一起透過 `page-<32 個小寫十六進位字元>` subprotocol 傳送；同一個已載入頁面的重連會沿用。capability 不放在 URL，也不寫入瀏覽器持久儲存。若它出現在 WebSocket `Sec-WebSocket-Protocol` 除錯標頭，診斷遮蔽器會將它遮罩。`pagehide` 時，瀏覽器會透過盡力送出的 `sendBeacon`，把 capability 放在 JSON body 請求清除。重新載入頁面會產生新的 capability，無法取回上一頁的參考臉。

伺服器只在程序記憶體保存頁面狀態與人臉裁切。正常 `ENDED` 時開始計算歷史 conversation 的 **600 秒 TTL**。重連和比對失敗不會刷新歷史 TTL；已續接的 conversation 再次中斷，也沿用原本正常 `ENDED` 的到期時間。從未正常進入 `ENDED` 的活躍對話，會在第一次中斷時取得一次 600 秒參考臉租約；反覆重連或中斷都不會延長。`pagehide` 會盡力送出清除請求。由應用程式擁有的清理工作每秒清除頁面和參考臉租約，即使最後一條 WebSocket 已斷線仍會執行；server 關閉時也會清除所有頁面 store。撤回同意和手動重設同樣會清除參考臉。

每條 WebSocket 連線都會取得頁面 generation。替換傳輸時，系統先讓舊工作失效，再把共用頁面 store 交給新連線；舊 generation 的 worker 和 callback 不能改動新連線狀態。Re-ID 工作也會檢查 store generation 與 session；若候選參考臉在 handoff 前已到期，就拒絕續接。傳輸中斷不是正常 `ENDED`：符合條件的活躍參考會在有限租約內保留給同頁重連。

相機停止輸入時，在場觀察時鐘會停在**最後一張完成處理的畫面之後三秒**，而不是延遲醒來的 ticker 當下。這可避免主機睡眠／喚醒期間被算成離開時間。新畫面完成處理後時鐘恢復，且時間不倒退。

PoC 最多保留三個未結束的歷史 conversation，每個最多三張參與者裁切。超過容量時淘汰最久未結束的歷史 conversation，不會為了符合歷史容量而移除活躍 conversation。頁面 registry 另有獨立的頁面容量上限；已到期且中斷的頁面可被清理。

## 隱私與服務呼叫

| 資料 | 處理方式 |
|---|---|
| 參考臉／探測臉 | 只在伺服器記憶體保存 320 × 320 JPEG 人臉裁切；不保存整張比對畫面。 |
| 保存期限 | 正常 `ENDED` 後最多 600 秒；頁面中斷也有 600 秒硬性租約。不寫入磁碟。 |
| 服務 | 設定後且取得同意時，Typesafe Jev 每次收到一張當前人臉與一張參考臉。批次模式會在同一請求送出所有符合條件的參考臉。合成驗收不會呼叫服務。 |
| 同意 | 必須同時開啟功能與同意旗標；關閉功能會清除參考臉。 |
| API key | 在[伺服器的 .env](configuration.md#使用-env-啟動) 設定 `TYPESAFE_API_KEY`，或在設定頁手動輸入。手動輸入會覆蓋目前程序的啟動值。 |
| 瀏覽器縮圖 | 啟用 `reid_show_crops` 時，選用的 112 px 比對縮圖只送到此頁面，不會放入診斷事件。 |

啟用雲端後端前，應告知參與者、取得必要同意，並確認服務提供者對影像的處理與保存條款。合成驗收不可使用真實人臉資料。

## 驗收狀態

Chrome 與離線模型可用時，`browser-reid` 涵蓋九個命名案例：`consent_required`、`provisional_conversation`、`same_page_handoff`、`nonblocking_comparison`、`consent_revocation`、`late_result_discarded`、`nonblocking_media`、`cleanup`、`server_errors`。它會將 provisional ID 和原 conversation 比較；成功 handoff 時精確核對原 session 與 conversation。撤回同意後，具抗取消行為的 mock 必須在取消後增加 `completed_after_cancel` 並回傳 `.99`，案例再確認 handoff／resume 計數沒有增加，且 UI 維持清空。Production `state` 訊息帶有 `session`；新 session 或 WebSocket 重連、撤回同意、停用功能及 wipe 參考資料時，UI 會清除舊 handoff 狀態與縮圖，並忽略較舊 session 的 re-ID 訊息。

`functional` adapter 使用不含真實人臉的 mock 身分值和虛擬時間，驗證 re-ID 判斷規則與延續性契約。Browser adapter 另透過合成 Chrome 媒體、production 設定與 crop callback、`CrossSessionReid` 和明確 handoff 驗證控制流程；mock `CloudReid` 只注入隔離的 child server。測試 consent 只是測試設定，不代表真人已同意。Crop 和 page capability 留在記憶體；報告不保存識別碼或影像資料。測試不呼叫 Jev，不能證明生物特徵辨識準確度或服務提供者行為。各案例狀態請查看產生的報告。完整規格見 [PoC 驗收計畫](acceptance.md) 和[架構](architecture.md)。
