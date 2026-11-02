[English](../acceptance.md) · **繁體中文**

# PoC 驗收

本計畫的目標環境是 **Apple M4＋Chrome＋本機 server、單一使用中的連線、同場 1–3 位參與者**。判斷時優先避免已加入對話的人被誤判離開。合成測試通過只代表受測的程式路徑符合結果，不能證明實體鏡頭效能或身分辨識準確度。

## 執行測試套件

runner 預設將 `acceptance.json` 與 `acceptance.md` 寫入 `artifacts/acceptance`。報告記錄允許清單中的主機與套件資訊、案例結果、門檻和限制；不會傾印環境變數或媒體內容。

```bash
uv run python scripts/run_acceptance.py --suite functional
uv run python scripts/run_acceptance.py --suite pipeline --pipeline-seconds 10
uv run python scripts/run_acceptance.py --suite browser --browser-seconds 10
uv run python scripts/run_acceptance.py --suite browser-reid --browser-seconds 10
.venv/bin/python scripts/run_acceptance.py --suite browser-reid --browser-seconds 10
uv run python scripts/run_acceptance.py --suite soak --soak-seconds 1800
uv run python scripts/run_acceptance.py --suite all --pipeline-seconds 10 --browser-seconds 10 --soak-seconds 1800
```

`functional` 使用注入時鐘、合成追蹤器偵測資料和記憶體中的身分 mock。`pipeline` 會啟動隔離的本機 server、載入本機模型，再透過 loopback 重播合成 PCM 和影像；預設暖機 5 秒。`browser` 和 `browser-reid` 共用隔離的 headless Chrome、暫存 profile、loopback server 和合成媒體 fixture。fixture 經由 app 的 media 路徑提供合成 canvas 影像與音訊，不呼叫原生 `getUserMedia`；兩者都會走過實際擷取計時器、AudioWorklet、binary JPEG／PCM 傳輸和 WebSocket 生命週期。`browser` 關閉 re-ID，檢查媒體與連線延續。`browser-reid` 使用 `legacy` tracker、`rules` presence／intent、`vision`、full-face ROI、`leave_distance=10m`、2 fps，並停用 pose；它不使用 functional 套件的 HMM／score `PROFILE_CONFIG`。它透過 production 設定流程啟用及撤回同意、擷取 production crop，並進入 production `CrossSessionReid` 流程；新 session 先取得 provisional conversation，再明確 handoff 到原 conversation。只有隔離 child server 注入本機 mock `CloudReid`，正式 server 沒有測試 hook 或 mock 選項。它也驗證比對等待中撤回同意、拒絕延遲回覆、JPEG／PCM 持續處理與清理。測試中的 consent 旗標只是測試設定，不代表真人已同意。Crop bytes 與 page capability 留在記憶體；報告只保存允許的布林與數值，不含識別碼或影像資料。`--browser-seconds` 同時控制兩個 browser 套件的穩定串流時間；生命週期檢查會額外花時間。兩個套件都不驗在場準確度、實體裝置、Jev 辨識品質或 RTF，也不呼叫 Jev／雲端 ASR。`soak` 預設暖機 10 秒，再以真實時間量測 1,800 秒；較短的冒煙測試不能通過 30 分鐘門檻。依套件而定，pipeline、browser 和 soak 需要本機 runtime 檔案；runner 不會下載缺少資產。

`browser-reid` 可用 `--suite browser-reid` 執行，並和 `browser` 共用 `--browser-seconds`。Python 介面 `run_acceptance(..., suite="browser-reid", browser_seconds=10)` 會呼叫 `presencekit.acceptance_browser_reid.run_browser_reid(seeds=20, seconds=10)`；Chrome fixture 位於 `scripts/acceptance_browser_reid_fixture.js`。

`all` 依序執行 `functional`、`pipeline`、`browser`、`browser-reid`、`soak`，至少需要 30 分鐘的 soak 實測時間；開發迭代時可指定單一套件。`--seeds` 控制可重現的功能情境；pipeline 使用固定 seed。`--output-dir` 可更改報告位置。加入 `--strict` 後，只要仍有 `not_run` 或 `pending_real`，結束碼為 2；案例失敗時為 1。非 strict 模式只有案例失敗才回傳 1，涵蓋不完整則結束碼為 0，但報告總狀態仍是 `incomplete`。

若啟動套件前找不到 Chrome 或必要的本機模型，對應案例標示 `not_run`。browser 開始執行後發生斷言或執行錯誤，則標示 `failed`。自動化使用暫存 Chrome profile 和 loopback，不會使用實體鏡頭或麥克風。`real_chrome_devices` 與 `real_jev_quality` 仍為 `pending_real`。PoC 尚未實作 TV 視覺語意分類，因此該項維持 `not_run`。

## 驗收契約

| 項目 | 必須符合的結果 |
|---|---|
| 參與者安靜 | 已加入的參與者安靜 120 秒，不得只因沉默而結束 session。 |
| 短暫遮擋 | 已加入的參與者被遮擋 3 秒，不得誤結束。 |
| 多人延續 | 最多 3 位已加入參與者中，只要還有一位在場，session 就持續。其他人離開後亦同。路人或背景音單獨出現，不得維持對話。 |
| 確實離開 | 最後一位已加入的參與者離開後，tracker 應在 20 秒內進入 `ENDED`。 |
| 新 session | 之後有人回來時先建立新 session 和暫用 conversation ID。只有 re-ID 明確且無歧義成功，才發出 `handoff` 關聯舊 conversation；原 session ID 與事件歷史保留。 |
| re-ID 歧義 | 只有明確的成員重疊且唯一對應一個歷史 conversation 才能續接。候選人分別對應多個 conversation 時，新 session 保持未關聯。不確定分數、逾時、沒有臉、撤回同意、達容量上限或服務錯誤都不能阻擋 session，也不能產生 handoff。 |
| re-ID 時限 | 從新 session 開始起算，10 秒內完成比對或放棄；以真實時間計算，`AWAY` 等在場狀態不暫停期限。 |
| 歷史參考 | 最多保留 3 個已結束的 conversation，每個最多保存 3 位參與者參考。600 秒期限從最近一次正常 `ENDED` 起算；重連和比對失敗不延長，續接後再中斷也沿用原到期時間。從未正常進入 `ENDED` 的 conversation，則從第一次中斷起取得固定 600 秒活躍參考租約。超量時淘汰最久未結束的歷史項目，不能為騰空間而移除活躍對話。 |
| 同意與保存 | 必須同時啟用 re-ID 並取得明確同意。臉部裁切只保存在記憶體；到期、頁面釋放／逾期、撤回同意、重設或 server 關閉時清除。重新載入頁面或重啟 server 後不保留參考。 |
| 傳輸中斷 | 標示感測中斷；WebSocket 或相機／麥克風傳輸中斷不代表參與者離開，也不能算正常 `ENDED`。同一頁面重連可取回仍有效的記憶體狀態。頁面 generation 會擋掉舊 worker；應用程式擁有的清理工作每秒檢查租約，最後一條 WebSocket 關閉後仍持續執行。 |
| 感測中斷 | 在場觀察時鐘停在最後完成畫面後 3 秒，而非延遲醒來的 ticker 時間，避免主機睡眠被算成參與者缺席。新畫面完成處理時恢復單調時鐘。 |

## 效能驗收

只有使用真實時間的 pipeline adapter 能驗收吞吐量與記憶體穩定性。暖機後連續量測 30 分鐘，必須同時符合：

| 指標 | 門檻 |
|---|---:|
| 每秒取樣的近 5 秒 audio RTF 之 p95 | ≤ 0.5 |
| Audio dropped | 0 |
| 量測期間未捕捉的 server 錯誤 | 0 |
| 音訊輸入佇列 | 不得持續成長；後四分之一平均值 ≤ 前四分之一平均值 + 1 block，峰值低於 25 個 block 容量 |
| RSS 成長 | 最後 10 分鐘平均 RSS − 最初 10 分鐘平均 RSS ≤ 30 MiB |
| 量測品質 | 至少量測 1,800 秒且新鮮樣本涵蓋率至少 95%；RSS 首尾區間各至少 570 筆樣本 |

pipeline adapter 使用本機合成音訊，以及空白畫面或重複的本機測試臉。它評估 server 處理與資源趨勢，不能代替真實 Chrome 擷取、相機／麥克風驅動、現場聲學或 Jev 辨識品質驗收。真人測試的 audio RTF 約 0.37 仍是獨立待查問題。請查看產生的 soak 報告確認結果；本計畫本身不代表 30 分鐘門檻已通過。

## 目前案例涵蓋範圍

實體 M4／Chrome 鏡頭與麥克風檢查、效能證據、同意流程、Jev 驗收矩陣及清理步驟，請見[真實裝置與 Jev 驗收流程](real-acceptance.md)。

Chrome 與離線模型可用時，`browser-reid` 目前回報九個案例：`consent_required`、`provisional_conversation`、`same_page_handoff`、`nonblocking_comparison`、`consent_revocation`、`late_result_discarded`、`nonblocking_media`、`cleanup`、`server_errors`。它會確認初始 provisional conversation 不同於舊 conversation，並在恢復時精確核對原 session 與 conversation。撤銷同意後，具抗取消行為的 mock 比對必須在取消後完成（`completed_after_cancel` 增加，且 mock 回傳 `.99`），才檢查 `late_result_discarded`；案例會確認 handoff／resume 計數沒有增加，且 UI 維持清空。各項實際狀態以報告為準，不固定通過數。

Production `state` 訊息包含 tracker `session`。進入新 session 或 WebSocket 重連時，UI 會清掉舊 re-ID 狀態和縮圖；撤回同意、停用 re-ID 或清除參考資料時也會立即清除。舊 session 的 re-ID 狀態與 handoff 訊息會被忽略。這些 browser 驗收使用合成媒體與 child-only mock，不驗在場準確率、實體相機／麥克風、Jev 身分品質或 RTF 門檻；fixture 支援 pagehide 清理。
目前 functional adapter 涵蓋安靜 120 秒、遮擋 3 秒、仍有參與者在場、離開期限、同一人、不同人、成員重疊、重疊歧義、不確定結果、後端錯誤、歷史到期、re-ID 期限、同頁重連、頁面隔離、舊租約拒絕、歷史 TTL 保留／到期、參考臉清除和長時間感測中斷。適用情境使用虛擬時間和身分 mock。20-seed 功能測試為 **290 passed、0 failed**；這只代表功能套件。測試有使用背景 speech flags 和低意圖路人情境，但不能證明系統能判斷聲音來源是電視。PoC 尚無 TV 視覺語意分類能力，因此該項明確標示 `not_run`。缺少模型檔案或 pipeline 樣本過期時，也會回報 `not_run`，不算通過。各案例與 soak 門檻的目前狀態以產生的報告為準。

每個案例使用 `passed`、`failed`、`not_run`、`pending_real` 其中一種狀態。合成／mock 案例通過時，整體 PoC 仍可能未完成。真實 Chrome 相機／麥克風，以及取得同意素材的 Jev 辨識品質與延遲測試仍為 `pending_real`；mock 只能驗證控制流程，不能代表生物特徵辨識準確率。另見[Re-ID](reid.md)、[架構](architecture.md)與[報告格式實作](../../presencekit/acceptance.py)。
