[English](../getting-started.md) · **繁體中文**

# 快速開始

## 需求

- [uv](https://docs.astral.sh/uv/) 和 Python ≥ 3.10（選用的 PureSound 音訊管線需要 ≥ 3.12）。
- 有鏡頭和麥克風的瀏覽器。瀏覽器只在 `localhost` 或 HTTPS 下允許使用鏡頭和麥克風。
- 不需要 GPU，全部在 CPU 上執行。

## 執行 demo

```bash
uv sync
uv run presence-demo                      # http://127.0.0.1:8000
```

打開網頁，按 **開始**，允許鏡頭和麥克風。站到鏡頭前：臉上會出現框，狀態列變成 **ACTIVE**，人員清單把你列為參與者。
走開後，可以看到狀態依序變成 **MAYBE_AWAY → AWAY → ENDED**。

第一次執行會把人臉模型（YuNet、SFace，幾 MB）下載到 `models/`，可以用 `PRESENCE_MODEL_DIR` 指定其他位置。
所有模型都跑在 ONNX Runtime 上（預設 CPU；`PRESENCE_ORT_PROVIDERS` 可以選其他執行提供者）。下載失敗時會改用簡單的
Haar + 顏色直方圖備援，demo 仍然可以啟動。

`uv sync` 會安裝 SDK、網頁 demo 和測試工具。在這個 repository 之外，`pip install "presence-demo[web]"` 安裝 demo；
單純 `pip install presence-demo` 只有 SDK（見 [SDK 與 ONNX Runtime](sdk.md)）。

## 開放到網路上

只要有其他人連得到伺服器，就要設定存取權杖。設定後，頁面、API、WebSocket 都需要 `?token=…`：

```bash
export PRESENCE_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
uv run presence-demo --host 0.0.0.0 --port 8000
# 開啟 http://<主機>:8000/?token=<你的權杖>
```

權杖只放在環境變數，不要寫進會提交的檔案。遠端使用時，請放在 HTTPS 反向代理後面，瀏覽器才會允許鏡頭。

## 語言

介面有英文和繁體中文。預設跟隨瀏覽器語言（`zh` 開頭 → 繁體中文，其他 → 英文）。可以用側邊欄的 **EN / 中文**
切換，或用 `?lang=en`、`?lang=zh-TW` 開啟。選擇會記在瀏覽器裡。

## 選用功能

| 功能 | 開啟方式 | 說明 |
|---|---|---|
| 身體姿勢（MoveNet） | 設定 → 姿勢，或「推薦」預設組合；要用 ONNX Runtime 取代 LiteRT，先執行一次 `scripts/export_movenet_onnx.py` | [偵測](detection.md)、[SDK](sdk.md#模型) |
| 距離估計 | 設定 → 距離，再做「距離校正」 | [偵測](detection.md) |
| PureSound 人聲分離 | `uv sync --extra puresound` 和 `uv run python scripts/fetch_puresound_model.py --clone` | [音訊](audio.md) |
| 雲端語音辨識 | 設定 → 語音辨識，加上 API key | [語音辨識](asr.md) |
| 跨 session 使用者判斷 | 設定 → Re-ID，明確同意，加上 API key | [Re-ID](reid.md) |

提示：設定裡的 **推薦** 預設組合會開啟較新的演算法（Kalman 追蹤、HMM 在場判斷、意圖分數、姿勢、距離、PureSound）。
一般預設值為了相容舊版，維持保守。

## 開發指令

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
uv run python scripts/benchmark.py        # 省電機制、姿勢、距離的 CPU 成本
uv run python scripts/eval_tracking.py    # 合成情境：legacy 與 kalman、rules 與 hmm、意圖
```

開發流程見 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
