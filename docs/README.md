**English** · [繁體中文](zh-TW/README.md)

# Documentation

PresenceKit answers one question for a conversational device: **is the person (or group) I am talking to still here?**
These pages explain how it works, how to run it and how to tune it. Start with *How it works* if you are new.

## Start here

| Page | What you will learn |
|---|---|
| [Getting started](getting-started.md) | Install, run, open the demo, optional extras |
| [How it works](how-it-works.md) | The whole pipeline in plain words, with one worked example |

## Algorithms (one page per layer)

| Page | Layer | Question it answers |
|---|---|---|
| [Detection](detection.md) | 1 · perception | Where are the faces and bodies, how far away, who is who? |
| [Tracking](tracking.md) | 2 · tracking | Which detection belongs to which person over time? (legacy vs Kalman) |
| [Intent](intent.md) | 3 · intent | Who actually wants to talk, and who is just passing by? |
| [Presence](presence.md) | 4 · presence | Is the group still here? (rules vs HMM) |
| [State machine](state-machine.md) | 5 · decision | ACTIVE / IDLE / MAYBE_AWAY / AWAY / ENDED, timeouts, NEW_GROUP |
| [Audio](audio.md) | cross-check | PureSound voice isolation checked against vision |
| [Re-ID](reid.md) | optional | Is the new session the same user as last time? |
| [PoC acceptance](acceptance.md) | validation | What synthetic, pipeline and real-device acceptance means |
| [Real-device and Jev runbook](real-acceptance.md) | operations | M4 + Chrome device checks, consent gate, Jev matrix, and cleanup |
| [Speech recognition](asr.md) | optional | Streaming ASR of the cleaned audio |

## Using and tuning

| Page | Content |
|---|---|
| [Configuration](configuration.md) | Every parameter (generated from code), presets, environment variables |
| [Tuning](tuning.md) | A step-by-step recipe for a new room or camera |
| [Performance](performance.md) | Power savers, latency per stage, embedding backends and NPU |
| [Web UI](ui.md) | Layout, language switch, design system and how to add strings |

## Project and operations

| Page | Content |
|---|---|
| [Architecture](architecture.md) | Code layout, threads, WebSocket protocol |
| [SDK and ONNX Runtime](sdk.md) | Embedding `PresenceEngine`, execution providers, model files, porting |
| [PoC acceptance](acceptance.md) | Target profile, thresholds, runner commands, and pending real tests |
| [Real-device and Jev runbook](real-acceptance.md) | Operator procedure for physical camera/microphone and consented Jev checks |
| [Privacy](privacy.md) | What is processed, what is kept in memory, what can leave the device |
| [Logging](logging.md) | Structured events, `/debug/events`, debugging a hang |
| [Licenses](licenses.md) | Third-party models, packages and data |

## Figures

All figures live in [`img/`](img). Diagrams are written in [Mermaid](https://mermaid.js.org) (`img/*.mmd`) and are
also shown inline on the pages; plots and tables are drawn from the real code on synthetic scenes.

```bash
# plots and table figures (matplotlib is only needed for this script)
uv run --with matplotlib python scripts/make_figures.py
# mermaid diagrams -> PNG (needs mermaid-cli: npm i @mermaid-js/mermaid-cli)
MMDC=./node_modules/.bin/mmdc scripts/render_diagrams.sh
# parameter reference in configuration.md
uv run python scripts/gen_config_docs.py
```

Each English page has a Traditional Chinese mirror under [`zh-TW/`](zh-TW/README.md). When you change one, please
update the other (or open an issue so someone else can).
