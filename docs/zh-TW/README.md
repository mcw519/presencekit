[English](../README.md) · **繁體中文**

# 文件

PresenceKit 替對話裝置回答一個問題：**正在跟我說話的人（或這群人）還在嗎？**
本文件集分為**兩個主要入口**：

1. **新手與專案說明入口** — 適合剛接觸專案的人：快速安裝執行、嵌入 SDK、理解程式碼分層架構與設定方式。
2. **演算法詳細設計入口** — 深入定義各層的資料結構、數學模型與決策規則，並完整說明演算法如何組裝出最終成果。

---

## 入口一：新手與專案說明

| 頁面 | 內容 |
|---|---|
| [快速開始](getting-started.md) | 安裝、執行網頁 demo、選擇輸入裝置、選用功能 |
| [架構](architecture.md) | 程式碼分層（`core`、`inference`、`detectors`、`identity`、`runtime`、`sdk`、`web`）、執行緒與 WebSocket/HTTP 協定 |
| [SDK 與 ONNX Runtime](sdk.md) | 嵌入 `PresenceEngine`、`OrtOptions`、執行提供者、模型目錄與移植 |
| [設定參數](configuration.md) | 所有參數（由程式產生）、預設組合（預設「近場優先」）、環境變數與 `.env` |
| [網頁介面](ui.md) | 版面、語言切換、設計規範、錄影匯出與新增文字 |
| [隱私](privacy.md) | 記憶體處理原則、`ENDED` 清除機制、可能離開裝置的資料 |
| [調參指南](tuning.md) | 新場地、新鏡頭的調整步驟 |
| [效能](performance.md) | 省電機制、各階段延遲、embedding 後端與 NPU |
| [PoC 驗收](acceptance.md) | 目標環境、門檻、runner 指令與待實測項目 |
| [真實裝置與 Jev 驗收流程](real-acceptance.md) | M4＋Chrome 實體裝置檢查、同意門檻、Jev 案例矩陣與清理 |
| [日誌](logging.md) | 結構化事件、`/debug/events`、排查卡住 |
| [授權](licenses.md) | 第三方模型、套件和資料 |

---

## 入口二：演算法詳細設計

| 頁面 | 層級／範圍 | 定義與內容 |
|---|---|---|
| **[演算法詳細設計與組裝](algorithms.md)** | **入口 · 完整管線** | **資料契約、數學模型，以及 7 層演算法如何組裝成 `Step` 與乾淨音訊** |
| [運作原理](how-it-works.md) | 概念導覽 | 用白話說明整條管線，附一個完整例子 |
| [偵測](detection.md) | 1 · 感知 | YuNet、SFace、MoveNet、朝向、距離估計與嘴部動作 |
| [追蹤](tracking.md) | 2 · 追蹤 | `legacy` 貪婪配對與 `kalman`（Kalman＋匈牙利＋HSV 外觀＋滑行延續） |
| [互動意圖](intent.md) | 3 · 意圖 | `rules` 停留計時與 `score`（6 項線索＋朝向閘門＋EMA＋`speech_join`） |
| [在場判斷](presence.md) | 4 · 在場 | `rules` 連續幀與 `hmm` 連續時間對數勝算貝氏濾波器＋音訊偏移 |
| [狀態機](state-machine.md) | 5 · 決策 | `ACTIVE` / `IDLE` / `MAYBE_AWAY` / `AWAY` / `ENDED`、優先權模式、`NEW_GROUP` |
| [音訊](audio.md) | 6 · 交叉確認 | PureSound 人聲分離、雙路 CleanVAD、視覺交叉確認與起音救回 |
| [Re-ID](reid.md) | 7 · 選用 | 跨 session 身分延續（`ContinuationDriver`、`CallGuard`、Jev） |
| [語音辨識](asr.md) | 選用 | 乾淨音軌的串流語音辨識（AssemblyAI） |

## 圖片

所有圖都在 [`../img/`](../img)。流程圖用 [Mermaid](https://mermaid.js.org) 撰寫（`img/*.mmd`），也直接嵌在各頁；
曲線圖和表格圖是用真正的程式在合成情境上畫出來的。中文版圖檔的檔名是 `*.zh-TW.png`。

```bash
uv run --with matplotlib python scripts/make_figures.py      # 曲線圖與表格圖
MMDC=./node_modules/.bin/mmdc scripts/render_diagrams.sh      # Mermaid → PNG（需要 mermaid-cli）
uv run python scripts/gen_config_docs.py                      # configuration.md 的參數表
```

每一頁英文文件在這個資料夾都有對應的繁體中文版。修改其中一個時，請一起更新另一個（或開 issue 請其他人協助）。
