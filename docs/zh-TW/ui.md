---
version: alpha
name: presencekit-console
description: "Presence-demo's console is a PureSound playground screen: same shell, same components, same tokens. This file records how presence maps its content onto those components; it does not invent a second design system."
source:
  repository: https://github.com/mcw519/puresound
  path: puresound/web/static
  revision: 121bf623b491cb7f18805bec65099543ecc5323f
  vendored: presencekit/static/vendor/puresound
  license: Apache-2.0 (CSS), SIL OFL 1.1 (IBM Plex)
---

[English](../ui.md) · **繁體中文**

# 網頁介面

demo 頁面是用 PureSound 設計系統做成的單一畫面。這一頁說明版面、語言切換，以及修改介面時的設計規範。
上方的 YAML 標頭採用 [DESIGN.md](https://github.com/google-labs-code/design.md) 格式。

## 版面

- **側邊欄**：連線狀態、語言切換（EN / 中文）、主題（自動／淺色／深色）、導覽。
- **主欄**：鏡頭畫面和疊圖（臉與身體框、骨架、ROI、字幕）、音訊軌（波形／頻譜）、ROI 編輯器、監聽／錄音。
- **右欄**：狀態機、語音辨識、在場機率圖、倒數條、含意圖條的人員清單、使用者比對卡片、資源和事件紀錄。
- **設定抽屜**：預設組合，接著每個管線模組一區（輸入 → 偵測 → 姿勢 → 距離 → 追蹤 → 意圖 → 在場 → 狀態 → 音訊 →
  語音辨識 → re-ID → 效能 → 顯示）。進階參數要開啟「進階參數」才顯示；不適用的參數會變灰並說明原因。

## 語言（i18n）

所有看得到的文字都來自字串表，每種語言一個檔案：

- `static/i18n/en.js` 和 `static/i18n/zh-TW.js` 定義 `I18N["en"]` / `I18N["zh-TW"]`，鍵完全相同。
- HTML 用 `data-i18n="key"`（文字）和 `data-i18n-title`、`data-i18n-aria-label`、`data-i18n-placeholder`（屬性）。
  script 呼叫 `t("key", {name: value})`，`{name}` 佔位符會被替換。
- 設定用 `s.<key>`（名稱）、`s.<key>.h`（說明）、`s.<key>.o.<value>`（選項名稱）；模組用 `mod.<id>`。
  [設定參數](configuration.md)的參數表就是從同一份字串表產生的。
- 語言選擇順序：網址的 `?lang=`，其次是保存的選擇（`localStorage["presence-lang"]`），再來是瀏覽器語言
  （`zh*` → `zh-TW`），最後是英文。切換時會重新載入頁面。
- `tests/test_i18n.py` 檢查兩份字串表的鍵和佔位符完全相同、程式用到的鍵都存在、每個設定和模組都有翻譯，
  以及 script 裡沒有寫死的中文。

新增語言：把 `en.js` 複製成 `<lang>.js` 並翻譯，在 `js/i18n.js` 的 `LANGS` 加上代碼，在 `index.html` 的 `#langSw`
加一個按鈕，再擴充測試。

## 原則

**使用 PureSound 的設計，不只是它的字型。** HTML 優先使用 vendored stylesheet 裡的 PureSound 元件 class。
`app.css` 只加上 presence 專用的版面（鏡頭畫面、狀態機列、倒數條、示波器、人員清單、設定模組手風琴），並把少數
presence 角色對應到上游 token。它不可以重新定義 `.button`、`.panel`、`.field`、`.control`、`.segmented`、`.chip`、
`.toast`、`.inspector`，或任何 `--page` / `--ink` / `--accent` token。

## 載入的 stylesheet

| Href | 上游 | 角色 |
| --- | --- | --- |
| `/static/vendor/puresound/tokens.css` | `css/tokens.css` | 顏色、字體、密度 |
| `/static/vendor/puresound/base.css` | `css/base.css` | IBM Plex、reset、focus、減少動態 |
| `/static/vendor/puresound/components.css` | `css/components.css` | button、field、control、segmented、chip、state、toggle、range、table、stat、panel、empty-state、toast、dialog |
| `/static/vendor/puresound/shell.css` | `css/shell.css` | app-shell、sidebar、page-header、workbench、inspector |
| `/static/vendor/puresound/screens-shared.css` | `css/screens.css` 的子集 | panel-title h2、metric-row / metric |
| `/static/app.css` | 只有 presence | 鏡頭、狀態列、示波器、人員、模組手風琴 |

上游檔案（未修改）的 checksum 固定在 `tests/test_design_tokens.py`。

## 元件對應（presence → PureSound）

| Presence 介面 | PureSound class | 說明 |
| --- | --- | --- |
| Shell | `.app-shell` `.sidebar` `.main-content` `.screen` | 和 playground 相同 |
| Nav items | `.nav-link` `.nav-icon` `.nav-group-title` | 外觀像 playground 連結的按鈕 |
| Connection | `.status-dot.is-ok` / `.is-err` `.runtime` | 取代自訂的 `.dot` |
| Theme | `.nav-switch` | 自動／淺色／深色 |
| Page header | `.page-header` `.page-heading` `.page-title` `.page-desc` `.page-actions` | 固定在上方、模糊背景 |
| Primary action (Start) | `.button.button-primary` | 墨色填滿，不用強調色 |
| Secondary (Settings) | `.button.button-secondary` | |
| Stop | `.button.button-danger` | 危險色外框 |
| Ghost / icon close | `.button.button-ghost.button-icon.button-small` | |
| Camera / status / audio cards | `.panel.panel-pad` + `.panel-title` > `h2` + `.panel-help` | 不用自訂 `.card` + 大寫 `h3` |
| Speaking / silence / ROI / session | `.metric-row` > `.metric` > `span` + `strong` | playground 的結果指標 |
| State machine | `.fsm-axis` > `.fsm-node` + `.fsm-edge`; `.fsm-hist` > `.fsm-tr` (app.css) | 節點排在一條軸上；目前節點填上狀態顏色，上一次轉換的來源是虛線、路徑用強調色 |
| Audio signals | `.sig-group.raw` / `.sig-group.clean` (waveform + spectrogram) | 每個訊號一組；薄荷綠／長春花藍邊 |
| ROI editor | `.segmented#roiPick` (face / body / lock) + drag on `#view canvas` | 臉部青色虛線、身體紫色點線；只有選取的 ROI 可以拖曳 |
| Live caption | `#view > .caption` (`.is-in` fade, `.is-final` white; `--nav` 78 % backing, `--on-nav` text) | 影像底部單行，從不鏡像 |
| Cross-session re-ID line | `.reid-line#reidSt` in the state card: decision (`↩ resume #n` in `--ok`, `＋ new` + reason), short `conversation_id` (mono), calls · tokens · cost · refs (mono, right) | 功能關閉時隱藏；從不顯示人臉影像 |
| Monitor stage | `.stage-head` (title + `.segmented#trkView` wave / spectrum / both, outside the frame) + `section.panel.monitor-stage` (12 px gutter on `--nav-2`) = `#view` rounded inset + `.sig-stack` inset panel of `.sig-lane`s (`.lane-label` gutter: chip, name, dB; `.lane-plots` rounded canvases) | 參考 PureSound CompareDeck；共用的邊距讓影像邊緣和時間軸對齊；每種檢視的 `--stage-chrome` 讓它留在視窗內 |
| Layout | Left column: monitor stage, then ROI card, then monitor/record/timeline card; right column: state, ASR, P(group), countdowns | |
| Monitor off/raw/clean | `.segmented` `[role=radiogroup]` `.is-active` | |
| Live camera chip | `.chip.is-live` | 強調色邊框 + 脈動 |
| Recording indicator | `#recInd` (pill on `--bad-soft`) | |
| Resource tiles | `.stat` > `span` + `strong` | |
| Latency table | `.table` + `.num` | |
| Empty tracks | `.empty-state` | 虛線面板 |
| Toast | `.toast.is-visible` | 永遠是深色的導覽外觀 |
| Error banner | `#errBanner` (toast chrome + `--nav-bad` rule) | |
| Settings drawer | `.inspector` `.inspector-head` `.inspector-body` `.inspector-section` `.eyebrow` | 用 `.is-open` + `.inspector-backdrop` 開啟 |
| Presets | `.segmented` | |
| Advanced toggle | `.toggle-row` `.toggle-ui` | |
| Bool settings | `.toggle-row` | 由 `rowHTML` 產生 |
| Selects / text / numbers | `.control` | |
| Sliders | `.range` + `.field-label` + `output` | 強調色滑塊 |
| Hints | `.toggle-hint` / `.tip` | |
| Module strip | `.chip` in `.flow-row` | 啟用時用強調色 |
| Module blocks | `.mod` `<details>` | presence 專用的手風琴，用 inspector 密度 |
| Scopes (wave / spectrum / timeline) | always-dark `--nav` ground | 和 playground 相同 audio players |
| Raw / clean series | `--mint` / `--periwinkle` | 和 playground 相同的訊號顏色 |

## `:root` 中的角色別名（app.css）

`--ok` → `--good`、`--idle` → `--mint-ink`、`--end` → `--ink-2`、`--viz-raw` → `--mint`、`--viz-clean` → `--periwinkle`、
`--violet` → `--magenta`。鏡頭疊圖使用這些 token 的深色主題 hex 值，因為鏡頭背景永遠是深色。

## 密度與版面

- 4 px 格線、`--panel-pad` 16、`--gap` 16、控制項高度 32 / 26、圓角 4 / 6 / pill，全部來自 token。
- 工作台：主欄（鏡頭面板）+ 寬 `--inspector-w` 的右欄（≥ 1400 px 時 400 px）。
- 設定抽屜 520 px；區段之間用細線分隔；「預設組合」「資料流」「匯入」用 eyebrow 標籤。
- 每頁只有一個主要按鈕（開始）。強調色只用在 focus、選取、即時和 P(group)。

## 無障礙

儀表板和設定在淺色、深色主題下都符合 WCAG 2.1 AA（headless Chrome 檢查）。上游的第三層墨色有些地方低於 4.5:1，
所以 `app.css` 把文字用的 `--ink-3` / `--on-nav-3` 強制改成 `--ink-2` / `--on-nav-2`。focus 外框和減少動態來自 `base.css`。

## 該做與不該做

- 新增介面時先用 PureSound class；只有在沒有適合的上游元件時才自訂 class。
- `app.css` 不可以重新定義上游 token 和元件（有測試檢查）。
- 除了鏡頭疊圖（上面已說明），元件中不要寫死 hex 顏色。
- 文件截圖不要放真實的鏡頭畫面。
- 新的文字一律放進兩份字串表，不要寫死在 HTML 或 JS 裡。
