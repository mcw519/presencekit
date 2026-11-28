# PresenceKit 試用頁

桌面 Chromium 上的近場優先監看頁。影像固定 2 fps，PureSound 與 YuNet 在瀏覽器內運算。

## 發布到 GitHub Pages

在 repository 的 **Settings → Pages** 將 Source 選成 **GitHub Actions**。到 **Actions → Browser preview Pages**，選 `main` 執行 **Run workflow**。 [發布後試用入口](https://mcw519.github.io/presencekit/)

## 本機啟動

需要 uv、Node.js，以及同層的 `../PureSound` checkout，版本為 `aba2962876c8ec8abec7971378d670677e0ec859`。以下指令在 PresenceKit 根目錄執行。

```sh
uv sync
.venv/bin/python scripts/fetch_puresound_model.py --from ../PureSound --dest models
.venv/bin/python -c "from presencekit.inference.models import ModelStore; assert ModelStore('models').ensure('yunet')"
(cd ../PureSound/sdk/web && npm ci && npm run build)
.venv/bin/python experiments/browser_poc/prepare.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory artifacts/browser-poc/site
```

開啟頁面後按「開始」，允許相機與麥克風存取。戴耳機可切換監聽原音或輸出。每次最多錄製兩分鐘，停止後可回放與下載 WAV。左下角可切換 EN／中文。 [開啟試用版](http://127.0.0.1:8765/) · [English](http://127.0.0.1:8765/?lang=en)

試用版使用臉部估距及簡化追蹤。完整的 session 狀態機可用 Python 版。 [Python 版快速開始](../../docs/zh-TW/getting-started.md)
