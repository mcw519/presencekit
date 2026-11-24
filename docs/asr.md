**English** · [繁體中文](zh-TW/asr.md)

# Speech recognition (optional, cloud)

> **External cloud service: audio leaves the device** and is sent to AssemblyAI. Off by default
> (`asr_provider = none`). Check user consent, local privacy law and the provider's terms and retention policy
> first.

The corrected audio from the [PureSound pipeline](audio.md) can be streamed to AssemblyAI Universal-Streaming v3
for live captions. Transcripts appear in the **Speech recognition** panel and as a caption line over the video.

```
mic → PureSound + vision cross-check → corrected clean audio ──(only during a session)──▶ AssemblyAI (wss)
browser panel ◀── WS {"type": "asr"} ◀── server (forwards only, stores nothing) ◀── turns (partial / final)
```

## How it behaves

- Requires `audio_pipeline = puresound`. The **server** opens the connection and sends the key in the
  `Authorization` header, so the key never reaches the browser.
- Connected **only during a session** (state is not ENDED). On ENDED, stop, disconnect or a new session it sends
  `Terminate` and closes.
- `asr_source`: `clean` (default: the corrected output, with leaks attenuated) or `raw`.
- `asr_feed`: `speech` (default: only chunks that count as speech carry real audio; others are sent as silence to
  keep real-time pacing, so their content never leaves the device) or `session` (everything during a session).
- 20 ms chunks are batched into 100 ms messages. Up to 5 s are queued during a network hiccup; older audio is
  dropped. Errors reconnect with exponential backoff (at most 3 tries; reset after `Begin`).
- Each turn carries `start_s` / `end_s` relative to the session start.
- Transcripts live only in the page's memory (last 100 lines). The server keeps no text and logs none.

## API key

Set `ASSEMBLYAI_API_KEY` in [the server's .env file](configuration.md#loading-settings-from-env), or type it into
**Settings → Speech recognition** (`POST /api/asr_key`, protected by `PRESENCE_TOKEN`). A manual entry overrides the
startup key for the current process. `GET /api/asr_status` returns only the set flag and the source
(`env`, `ui`, or `null`). The key is shared by all connections, so set `PRESENCE_TOKEN` when others can reach the demo.

## Parameters

| Parameter | Default | Sent as | Models |
|---|---|---|---|
| `asr_speech_model` | `universal-3-6-pro` | `speech_model` | all |
| `asr_language_codes` | `zh,en` | `language_codes` + `language_detection` | Universal-3.x Pro |
| `asr_keyterms` | empty | `keyterms_prompt` (≤ 100 terms) | all |
| `asr_format_turns` | on | `format_turns` | Universal-Streaming |
| `asr_eot_confidence` | 0.4 | `end_of_turn_confidence_threshold` | Universal-Streaming |
| `asr_min_turn_silence` | 400 ms | `min_turn_silence` | all |
| `asr_max_turn_silence` | 1280 ms | `max_turn_silence` | all |
| `asr_region` | `global` | endpoint: global / `streaming.us` / `streaming.eu` | all |

Only parameters the chosen model supports are sent. According to the provider's documentation Universal-3.6 Pro
covers 32 languages with native code-switching, including Mandarin (`zh`) and Cantonese (`yue`). Recognition
quality for Taiwanese accents and the output script (traditional or simplified) have not been measured.

## Tests

`tests/test_asr.py` runs against a local mock WebSocket server: header and query parameters, 100 ms framing,
`Terminate`, partial / final parsing, the key endpoints never returning the key, and no key in logs. A live test
runs only when `ASSEMBLYAI_API_KEY` is set.

AssemblyAI is a paid service, billed by streamed time. It is not an open-source component; this project only uses
its public API.
