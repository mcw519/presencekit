**English** · [繁體中文](zh-TW/logging.md)

# Logging and diagnostics

The server emits structured, privacy-safe log lines of the form

```
2026-10-15 12:00:00,000 INFO presence event=heartbeat conn=3f9a1c frames_rx=120 frames_done=118 ...
```

Every record is also kept in an in-memory ring buffer that feeds the UI **Event log** panel and the
`/debug/events` endpoint. Nothing is persisted except the log file you configure.

## What is never logged

- Camera frames, audio samples, or anything derived from them that could reconstruct them.
- Transcript text (only ASR status / error codes).
- Re-ID face crops (kept in process memory only, see [Re-ID](reid.md)); only scores and counts are logged.
- API keys, the access token, passwords or authorization headers. Keys named `key`, `token`, `secret`,
  `password` or `authorization` are masked as `***`, and `token=` / `key=` in URLs are masked by a
  log filter applied to every handler (including uvicorn's).

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PRESENCE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `PRESENCE_LOG_FILE` | unset (stderr) | Path of a rotating log file |
| `PRESENCE_LOG_MAX_BYTES` | `5000000` | Rotate after this size |
| `PRESENCE_LOG_BACKUPS` | `3` | Rotated files kept |
| `PRESENCE_EVENT_BUFFER` | `500` | Events kept in memory for the UI / endpoint |
| `PRESENCE_VISION_LOCK_TIMEOUT` | `10` | Seconds to wait for the shared detector before reporting a stall |

## Events

| Event | Level | When |
| --- | --- | --- |
| `server_start` | info | Process start (version, log settings) |
| `module_init` | info | A model loaded (`module`, `name`, `ms`) |
| `pose_load_failed`, `audio_unavailable` | warning | Optional module could not load |
| `conn_open` / `conn_ready` / `conn_close` | info | Connection lifecycle: id, peer, close reason, duration, counters |
| `config` | info | Config diff (`changed` keys and new values; secrets masked) |
| `config_warnings` | warning | Config combination that will not behave as expected |
| `start`, `stop`, `calibrate`, `record_stream` | info | User actions |
| `state` | info | State-machine transition (`frm`, `to`, `session`, participant count, `reason`, `faces`, `speech`) |
| `session_event` | info | Session start / end |
| `audio_init` | info | Audio isolation ready (`ms`) |
| `asr` | info/warning | ASR status changes and errors (never text) |
| `reid_start`, `reid_call`, `reid_retry`, `reid_decision` | info | Cross-session re-ID ([Re-ID](reid.md)): refs compared, score, latency, model, tokens, running call count and cost estimate, final decision and `conversation_id`. Never images, base64 or keys |
| `reid_skip`, `reid_error`, `reid_breaker` | info/warning | Re-ID skipped (cap, cooldown, no key, breaker) or failed (timeout, HTTP code); breaker open after 3 errors |
| `reid_store` | info | In-memory reference faces added / expired / wiped (count only) |
| `session_open` | info | A session started: `reason` (the tracker transition reason), participants, faces, tracks, `speech` |
| `track_new`, `track_lost`, `track_recovered`, `track_dropped` | info | Track lifecycle (id, age, last confidence) |
| `face_miss_summary` | info/warning | Every 30 s while running: detection attempts, hits, hit rate (warning below 0.6) |
| `reid_wait` | info | Re-ID wait ended: `result`, `waited_s`, `participant_seen`, `why` (no_participant / face_too_small / facing_away / no_face / predicted / low_quality), `face_w` |
| `asr_skip` | warning | ASR enabled but no key (`reason=no_key`, at most once a minute) |
| `client` | info/warning | Browser reports: `what` = visibility, autopause, media_ended (camera/mic track ended), reconnect (`tries`, `down_s`), stale_socket, client_suspend. Fields are clamped to 64 chars |
| `host_suspend` | warning | Wall clock jumped ahead of the monotonic clock by > 2 s (the host was suspended); `gap_s` |
| `sender_closed` | info | The browser went away while a message was being sent; treated as a normal close (no `task_crash`) |
| `heartbeat` | info | Every 10 s per connection: frames received/processed/dropped, audio chunks, backlog, outbox drops, max send time, loop lag, cross-check counters, `reid_calls` / `reid_tokens` / `reid_cost_est` / `reid_refs` |
| `slow_frame` | warning | A frame took > 500 ms (per-stage timings) |
| `loop_lag` | warning | The event loop was blocked |
| `audio_backlog` | warning | Audio queue > 200 ms |
| `send_backlog` | warning | Outbound queue to the browser is growing (slow client or network) |
| `no_frames`, `no_audio` | warning | Running, but the browser stopped sending |
| `bad_frame` | warning | A video frame passed the envelope check but did not decode as JPEG (`frames_bad` so far; at most every 30 s). The frame is skipped; it does not count as a vision stall |
| `vision_stalled`, `audio_stalled` | error | Watchdog: a worker made no progress for 5 s while input was arriving. A stack dump of all threads is logged, the worker is restarted and the UI shows an error banner |
| `vision_lock_timeout` | error | The shared detector was held for longer than the timeout (stack dump attached) |
| `task_crash` | error | A worker raised; the full traceback is logged and the worker restarts with backoff |
| `task_gave_up` | error | A worker crashed too often and was not restarted |

## Endpoints and UI

- `GET /debug/events?n=200&conn=<id>` returns recent events as JSON. It is protected by the same token as
  the rest of the server (`PRESENCE_TOKEN`).
- The UI shows this connection's events in the collapsible **Event log** panel (bottom right) and shows an
  error banner when the watchdog or a worker crash reports a problem.

## Debugging a hang

`faulthandler` is enabled at start. For a live process:

```
py-spy dump --pid <PID>
```

The watchdog also writes a Python stack dump for all threads into the log when it detects a stall.
