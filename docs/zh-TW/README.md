[English](../README.md) · **繁體中文**

# 文件

PresenceKit 替對話裝置回答一個問題：**正在跟我說話的人（或這群人）還在嗎？**
這裡說明它怎麼運作、怎麼執行、怎麼調整。第一次看的話，請從「運作原理」開始。

## 從這裡開始

| 頁面 | 內容 |
|---|---|
| [快速開始](getting-started.md) | 安裝、執行、打開 demo、選用功能 |
| [運作原理](how-it-works.md) | 用白話說明整條管線，附一個完整例子 |

## 演算法（每一層一頁）

| 頁面 | 層 | 回答的問題 |
|---|---|---|
| [偵測](detection.md) | 1 · 感知 | 臉和身體在哪裡、多遠、誰是誰？ |
| [追蹤](tracking.md) | 2 · 追蹤 | 每個偵測結果屬於哪個人？（legacy 與 Kalman） |
| [互動意圖](intent.md) | 3 · 意圖 | 誰真的想說話，誰只是路過？ |
| [在場判斷](presence.md) | 4 · 在場 | 這群人還在嗎？（rules 與 HMM） |
| [狀態機](state-machine.md) | 5 · 決策 | ACTIVE / IDLE / MAYBE_AWAY / AWAY / ENDED、逾時、NEW_GROUP |
| [音訊](audio.md) | 交叉確認 | PureSound 人聲分離，再用影像確認 |
| [Re-ID](reid.md) | 選用 | 新的 session 是不是上一位使用者？ |
| [PoC 驗收](acceptance.md) | 驗證 | 合成測試、管線量測與實體裝置驗收代表什麼 |
| [真實裝置與 Jev 驗收流程](real-acceptance.md) | 操作 | M4＋Chrome 實體裝置檢查、同意門檻、Jev 案例矩陣與清理 |
| [語音辨識](asr.md) | 選用 | 乾淨音軌的串流語音辨識 |

## 使用與調整

| 頁面 | 內容 |
|---|---|
| [設定參數](configuration.md) | 所有參數（由程式產生）、預設組合、環境變數 |
| [調參指南](tuning.md) | 新場地、新鏡頭的調整步驟 |
| [效能](performance.md) | 省電機制、各階段延遲、embedding 後端與 NPU |
| [網頁介面](ui.md) | 版面、語言切換、設計規範、新增文字的方法 |

## 專案與維運

| 頁面 | 內容 |
|---|---|
| [架構](architecture.md) | 程式結構、執行緒、WebSocket 協定 |
| [SDK 與 ONNX Runtime](sdk.md) | 嵌入 `PresenceEngine`、執行提供者、模型檔、移植 |
| [PoC 驗收](acceptance.md) | 目標環境、門檻、runner 指令與待實測項目 |
| [真實裝置與 Jev 驗收流程](real-acceptance.md) | 實體鏡頭／麥克風及取得同意後 Jev 驗收的操作程序 |
| [隱私](privacy.md) | 處理什麼、記憶體裡保存什麼、什麼可能離開裝置 |
| [日誌](logging.md) | 結構化事件、`/debug/events`、排查卡住 |
| [授權](licenses.md) | 第三方模型、套件和資料 |

## 圖片

所有圖都在 [`../img/`](../img)。流程圖用 [Mermaid](https://mermaid.js.org) 撰寫（`img/*.mmd`），也直接嵌在各頁；
曲線圖和表格圖是用真正的程式在合成情境上畫出來的。中文版圖檔的檔名是 `*.zh-TW.png`。

```bash
uv run --with matplotlib python scripts/make_figures.py      # 曲線圖與表格圖
MMDC=./node_modules/.bin/mmdc scripts/render_diagrams.sh      # Mermaid → PNG（需要 mermaid-cli）
uv run python scripts/gen_config_docs.py                      # configuration.md 的參數表
```

每一頁英文文件在這個資料夾都有對應的繁體中文版。修改其中一個時，請一起更新另一個（或開 issue 請其他人協助）。
