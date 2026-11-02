**English** · [繁體中文](CONTRIBUTING.zh-TW.md)

# Contributing

Thanks for helping! Bug reports, field recordings of tricky situations (described, not uploaded), docs fixes and
code are all welcome. By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

```bash
uv sync                          # runtime + dev dependencies (pytest, httpx, ruff, websockets)
uv sync --extra puresound        # optional: PureSound audio pipeline (Python ≥ 3.12)
uv run presencekit --port 8000
```

## Before you open a pull request

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

All three must pass. Keep pull requests focused, and describe *what* changed and *why*. Behaviour changes need a
test and a line in [CHANGELOG.md](CHANGELOG.md).

## Tests

Tests are grouped by topic, not by version:

| Files | What they cover |
|---|---|
| `test_tracker*.py`, `test_presence_model.py`, `test_intent.py`, `test_kalman.py` | State machine and algorithm scenarios with an injected clock and synthetic embeddings (no models needed) |
| `test_ws_pipeline.py`, `test_watchdog.py` | WebSocket end to end; models are downloaded if possible, otherwise the fallback detector is used |
| `test_privacy.py`, `test_audio.py`, `test_recording.py` | Nothing is written to disk |
| `test_settings_ui.py`, `test_ui_*.py`, `test_design_tokens.py`, `test_i18n.py` | Web UI structure, design rules and translations |
| `test_docs.py` | Docs: language links, working relative links and images, generated parameter reference up to date |

## Code style

- Python ≥ 3.10, formatted and linted with ruff (line length 120). Type hints and docstrings on public functions.
- The core (`tracker.py`, `kalman.py`, `presence_model.py`, `intent.py`, …) stays pure logic: no I/O, an injectable
  clock, unit-testable.
- The web UI uses plain scripts, no build step and no third-party JS. Use PureSound component classes; never edit
  `presencekit/static/vendor/puresound/` and do not redefine upstream components or tokens in `app.css`
  (see [docs/ui.md](docs/ui.md)).

## Adding a parameter

1. Add the field to `Config` and its range to `RANGES` in `presencekit/config.py`
   (`test_ranges_cover_all_fields` checks this).
2. Add the default to `DEF` and the key to a module in `MODULES` (`presencekit/static/js/settings.js`); add
   `META` only for structure (range, step, options).
3. Add the label `s.<key>`, help `s.<key>.h` and option names `s.<key>.o.<value>` to **both**
   `static/i18n/en.js` and `static/i18n/zh-TW.js` (`test_i18n.py` checks this).
4. Regenerate the reference: `uv run python scripts/gen_config_docs.py`.

## UI text and translations

Never hard-code visible text. Add a key to both string tables and use `data-i18n` in HTML or `t('key')` in JS. If
you cannot translate, copy the English text into the other table and mention it in the pull request.

## Documentation

English pages live in `docs/`, Traditional Chinese mirrors in `docs/zh-TW/`. Update both when you can. Figures:
`uv run --with matplotlib python scripts/make_figures.py` (plots) and `scripts/render_diagrams.sh` (Mermaid).
Docs describe only what is built; ideas go into issues.

## Privacy and secrets

- Never commit tokens, `.env` files, `models/` or `logs/` (they are in `.gitignore`).
- Never write camera frames or audio to disk on the server, and never log them, transcripts or keys.
- Do not attach real camera images or recordings of people to issues or docs.
