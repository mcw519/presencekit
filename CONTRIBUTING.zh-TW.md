[English](CONTRIBUTING.md) · **繁體中文**

# 參與貢獻

謝謝你的協助！歡迎回報錯誤、描述難處理的實際情境（請用文字描述，不要上傳錄影）、修正文件或貢獻程式碼。
參與本專案即表示同意遵守[行為準則](CODE_OF_CONDUCT.md)。

## 環境

```bash
uv sync                          # 執行期 + 開發相依套件（pytest、httpx、ruff、websockets）
uv sync --extra puresound        # 選用：PureSound 音訊管線（Python ≥ 3.12）
uv run presencekit --port 8000
```

## 開 pull request 之前

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

三項都必須通過。pull request 請保持聚焦，說明改了*什麼*和*為什麼*。行為改變需要測試，並在 [CHANGELOG.md](CHANGELOG.md) 加一行。

## 測試

測試依主題分組，不依版本：

| 檔案 | 涵蓋內容 |
|---|---|
| `test_tracker*.py`、`test_presence_model.py`、`test_intent.py`、`test_kalman.py` | 狀態機和演算法情境，注入時鐘和合成 embedding（不需要模型） |
| `test_ws_pipeline.py`、`test_watchdog.py` | WebSocket 端到端；能下載模型就下載，否則用備援偵測器 |
| `test_privacy.py`、`test_audio.py`、`test_recording.py` | 不寫入任何檔案 |
| `test_settings_ui.py`、`test_ui_*.py`、`test_design_tokens.py`、`test_i18n.py` | 網頁介面結構、設計規範和翻譯 |
| `test_docs.py` | 文件：語言連結、相對連結和圖片都存在、產生的參數表是最新的 |

## 程式風格

- Python ≥ 3.10，用 ruff 格式化與檢查（行寬 120）。公開函式要有型別標註和 docstring。
- 核心（`tracker.py`、`kalman.py`、`presence_model.py`、`intent.py` …）維持純邏輯：沒有 I/O、可注入時鐘、可單元測試。
- 網頁介面用一般 script，沒有 build 步驟，也不使用第三方 JS。使用 PureSound 元件 class；不要修改
  `presencekit/static/vendor/puresound/`，也不要在 `app.css` 重新定義上游元件或 token（見 [docs/zh-TW/ui.md](docs/zh-TW/ui.md)）。

## 新增參數

1. 在 `presencekit/config.py` 的 `Config` 加欄位、在 `RANGES` 加範圍（`test_ranges_cover_all_fields` 會檢查）。
2. 在 `presencekit/static/js/settings.js` 的 `DEF` 加預設值，並把鍵加進 `MODULES` 的某個模組；`META` 只放結構（範圍、間距、選項）。
3. 在 `static/i18n/en.js` **和** `static/i18n/zh-TW.js` 加上名稱 `s.<key>`、說明 `s.<key>.h` 和選項名稱 `s.<key>.o.<value>`
   （`test_i18n.py` 會檢查）。
4. 重新產生參數表：`uv run python scripts/gen_config_docs.py`。

## 介面文字與翻譯

不要寫死看得到的文字。在兩份字串表加上鍵，HTML 用 `data-i18n`，JS 用 `t('key')`。如果無法翻譯，請先把英文複製到另一份表，
並在 pull request 中註明。

## 文件

英文頁面在 `docs/`，繁體中文版在 `docs/zh-TW/`，可以的話請兩邊一起更新。圖片：
`uv run --with matplotlib python scripts/make_figures.py`（曲線圖）和 `scripts/render_diagrams.sh`（Mermaid）。
文件只描述已經實作的功能；想法請開 issue。

## 隱私與機密

- 不要提交權杖、`.env`、`models/` 或 `logs/`（已列入 `.gitignore`）。
- 伺服器不可以把鏡頭畫面或聲音寫入磁碟，也不可以記錄畫面、聲音、逐字稿或 key。
- 不要在 issue 或文件中附上真實的鏡頭畫面或人的錄音。
