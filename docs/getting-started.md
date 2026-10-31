**English** · [繁體中文](zh-TW/getting-started.md)

# Getting started

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10 (≥ 3.12 for the optional PureSound audio pipeline).
- A browser with a camera and microphone. Browsers only allow them on `localhost` or over HTTPS.
- No GPU needed. Everything runs on the CPU.

## Run the demo

```bash
uv sync
uv run presence-demo                      # http://127.0.0.1:8000
```

Open the page, press **Start**, and allow camera and microphone. Stand in front of the camera: a box appears
around your face, the state strip turns **ACTIVE**, and the people list shows you as a participant. Walk away and
watch it go through **MAYBE_AWAY → AWAY → ENDED**.

The first run downloads the face models (YuNet and SFace, a few MB) into `models/`. Set `PRESENCE_MODEL_DIR` to
put them somewhere else. All models run on ONNX Runtime (CPU by default; `PRESENCE_ORT_PROVIDERS` picks another
execution provider). If the download fails, a simple Haar + colour-histogram fallback is used so the demo still
starts.

`uv sync` installs the SDK, the web demo and the test tools. Outside this repository, `pip install
"presence-demo[web]"` installs the demo; plain `pip install presence-demo` is the SDK only (see
[SDK and ONNX Runtime](sdk.md)).

## Exposing it on a network

When anyone other than you can reach the server, set an access token. The page, the API and the WebSocket then
all require `?token=…`:

```bash
export PRESENCE_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
uv run presence-demo --host 0.0.0.0 --port 8000
# open http://<host>:8000/?token=<your token>
```

Keep the token in the environment. Do not write it into files you commit. For remote use, put the server behind
an HTTPS reverse proxy so the browser allows the camera.

## Language

The UI is available in English and Traditional Chinese. It follows your browser language (anything starting
with `zh` → 繁體中文, otherwise English). Use the **EN / 中文** switch in the sidebar, or open the page with
`?lang=en` or `?lang=zh-TW`. The choice is remembered in the browser.

## Optional extras

| Feature | How to enable | Read more |
|---|---|---|
| Body pose (MoveNet) | Settings → Pose, or the *Recommended* preset; for ONNX Runtime instead of LiteRT, run `scripts/export_movenet_onnx.py` once | [Detection](detection.md), [SDK](sdk.md#models) |
| Distance estimate | Settings → Distance, then *Distance calibration* | [Detection](detection.md) |
| PureSound voice isolation | `uv sync --extra puresound` and `uv run python scripts/fetch_puresound_model.py --clone` | [Audio](audio.md) |
| Cloud speech recognition | Settings → Speech recognition + an API key | [ASR](asr.md) |
| Cross-session re-ID | Settings → Re-ID, explicit consent + an API key | [Re-ID](reid.md) |

Tip: the **Recommended** preset in Settings turns on the newer algorithms (Kalman tracking, HMM presence, intent
score, pose, distance and PureSound). The plain defaults are kept conservative for backwards compatibility.

## Developer commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
uv run python scripts/benchmark.py        # CPU cost of the power savers, pose and distance
uv run python scripts/eval_tracking.py    # synthetic scenes: legacy vs kalman, rules vs hmm, intent
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the workflow.
