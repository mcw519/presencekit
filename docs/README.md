**English** · [繁體中文](zh-TW/README.md)

# Documentation

PresenceKit answers one question for a conversational device: **is the person (or group) I am talking to still here?**
The documentation is organized around **two main entry points**:

1. **Newcomer & Project Guide** — if you are new to PresenceKit, want to run the demo, embed the SDK, or understand the codebase layout.
2. **Detailed Algorithm Design** — if you want the formal definitions, equations, and how the vision, audio, tracking, intent, presence, and state machine layers assemble into the final output.

---

## Entry 1: Newcomer & Project Guide

| Page | What you will learn |
|---|---|
| [Getting started](getting-started.md) | Install, run the web demo, select input devices, optional extras |
| [Architecture](architecture.md) | Package layers (`core`, `inference`, `detectors`, `identity`, `runtime`, `sdk`, `web`), threads, WebSocket & HTTP protocol |
| [SDK and ONNX Runtime](sdk.md) | Embedding `PresenceEngine`, `OrtOptions`, execution providers, model catalogue, porting |
| [Configuration](configuration.md) | Every parameter (generated from code), presets (`Near-field priority` default), environment variables, `.env` |
| [Web UI](ui.md) | Layout, language switch, design system, recording, adding strings |
| [Privacy](privacy.md) | What is processed in memory, what is erased on `ENDED`, what can leave the device |
| [Tuning](tuning.md) | Step-by-step recipe for adapting to a new room or camera |
| [Performance](performance.md) | Power savers, latency per stage, embedding backends and NPU |
| [PoC acceptance](acceptance.md) | Target profile, thresholds, runner commands, and pending real tests |
| [Real-device and Jev runbook](real-acceptance.md) | M4 + Chrome device checks, consent gate, Jev matrix, and cleanup |
| [Logging](logging.md) | Structured events, `/debug/events`, debugging a hang |
| [Licenses](licenses.md) | Third-party models, packages and data |

---

## Entry 2: Detailed Algorithm Design

| Page | Layer / Scope | What it defines |
|---|---|---|
| **[Algorithm Design & Assembly](algorithms.md)** | **Entry · Full Pipeline** | **Data contracts, mathematical models, and how all 7 layers assemble into `Step` and cleaned audio** |
| [How it works](how-it-works.md) | Overview | The pipeline in plain words, with one worked example |
| [Detection](detection.md) | 1 · Perception | YuNet, SFace, MoveNet, orientation, distance estimation, and mouth motion |
| [Tracking](tracking.md) | 2 · Tracking | `legacy` greedy matching vs. `kalman` (Kalman + Hungarian + HSV appearance + coasting) |
| [Intent](intent.md) | 3 · Intent | `rules` dwell vs. `score` (6 cues + facing gate + EMA + `speech_join`) |
| [Presence](presence.md) | 4 · Presence | `rules` frame streaks vs. `hmm` continuous-time log-odds Bayes filter + audio offset |
| [State machine](state-machine.md) | 5 · Decision | `ACTIVE` / `IDLE` / `MAYBE_AWAY` / `AWAY` / `ENDED`, priority modes, `NEW_GROUP` |
| [Audio](audio.md) | 6 · Cross-check | PureSound isolation, dual CleanVAD, vision cross-check, and onset recovery |
| [Re-ID](reid.md) | 7 · Optional | Cross-session identity continuity (`ContinuationDriver`, `CallGuard`, Jev) |
| [Speech recognition](asr.md) | Optional | Streaming ASR of the cleaned audio (AssemblyAI) |

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
