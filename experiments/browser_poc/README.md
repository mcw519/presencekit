# PresenceKit 純瀏覽器 PoC

桌面 Chromium 上用相機、麥克風和單執行緒 WASM 跑近場優先。影像固定 2 fps，音訊使用 16 kHz mono、20 ms 區塊。監看頁沿用 PresenceKit 的樣式與元件，並顯示 YuNet 距離框、原音與輸出的同尺度波形和頻譜、最近 240 ms 的音量差，以及說話、壓低和救回時間軸。監聽預設關閉，可戴耳機切換原音與輸出。按下停止後可試聽或下載已完成的 `out.wav`，每次最多試錄兩分鐘。

## 本機啟動

先用 `uv sync` 建立 PresenceKit 的 Python 開發環境，並把 PureSound checkout 放在同層的 `../PureSound`。Web SDK 原始碼與依賴對應 PresenceKit 鎖定的 commit `aba2962876c8ec8abec7971378d670677e0ec859`。以下命令準備 PureSound ONNX、manifest 與 YuNet，並啟動靜態檔案伺服器。

```sh
.venv/bin/python scripts/fetch_puresound_model.py --from ../PureSound --dest models
.venv/bin/python -c "from presencekit.inference.models import ModelStore; assert ModelStore('models').ensure('yunet')"
(cd ../PureSound/sdk/web && npm ci && npm run build)
.venv/bin/python experiments/browser_poc/prepare.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory artifacts/browser-poc/site
```

開啟 [本機介面](http://127.0.0.1:8765/)。靜態伺服器只讀取 `site/` 下的資源。兩個 Worker 分別執行 PureSound 和 YuNet，麥克風 PCM 來自 AudioWorklet。PoC 的模型與 WASM 路徑以頁面為基準，可放在專案子路徑。

左下角的 EN／中文可以在擷取中直接切換。語言選擇會記在瀏覽器，也可用 [英文入口](http://127.0.0.1:8765/?lang=en) 或 [繁體中文入口](http://127.0.0.1:8765/?lang=zh-TW) 指定。

## 對照 Python #9

以 repo 根目錄為工作目錄，使用 Python 3.12 以上並執行 `uv sync --extra puresound`。若缺少 Playwright Chromium，從 PureSound 的 `sdk/web` 執行 `npx playwright install chromium`。Python 基準使用 #9 合併的 `7ee50a2` 版本。

```sh
.venv/bin/python experiments/browser_poc/prepare.py --reference \
  "$HOME/Downloads/presence-2026-11-24T08-12-04" \
  "$HOME/Downloads/presence-2026-11-27T08-50-00"
node experiments/browser_poc/verify.mjs http://127.0.0.1:8765/ artifacts/browser-poc ../PureSound \
  presence-2026-11-24T08-12-04 presence-2026-11-27T08-50-00
.venv/bin/python experiments/browser_poc/measure.py
```

兩端使用同一批從 MP4 選出的 2 fps JPEG、原始 WAV 與錄製的相機設定。測試套用近場距離 1.3 m、純音訊救回與完整 PureSound 隔離。Python 跑原本的 FastAPI WebSocket 即時流程。瀏覽器使用與即時頁面共用的 Worker，依錄音時間送出影格與音訊。Playwright 從本機讀取私有 fixtures，產出逐區塊判斷、量測 CSV 及目標片段 WAV。相機與麥克風流程另以 Chromium 模擬裝置檢查。

靜態檔案在 `artifacts/browser-poc/site/`。測試影格、WAV、標記與結果存於 `site/` 外，僅供本機驗證。

## 移植範圍

PureSound 使用原版 Web SDK 與相同權重。瀏覽器移植了 raw/clean VAD、視覺交叉判斷、起音救回與 #9 的 150 ms 整句救回。YuNet 版估距使用臉寬與簡化的 IoU 追蹤。Python 基準還有 MoveNet、SFace、Kalman、HMM 與完整 session 狀態機。PoC 的實測結論只適用於這兩份錄音中的輸出與 `speech` 判斷。行動裝置、其他瀏覽器、真實相機和麥克風裝置的長時間表現仍待驗證。即時試錄在按下停止時匯出已產生的 PCM，串流尾端尚在處理的少量音訊可能不在檔案中。
