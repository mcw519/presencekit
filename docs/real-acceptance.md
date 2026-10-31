**English** · [繁體中文](zh-TW/real-acceptance.md)

# Real Chrome device and Jev acceptance runbook

This runbook is for human-supervised acceptance on an Apple M4, Chrome, the local server, and physical camera and microphone. Jev cross-session re-identification is a later phase and may start only after informed-consent, provider-terms, and budget checks. This document is an operating procedure; it does not mean any real device or Jev case has passed. Synthetic tests, mocks, headless Chrome, and synthetic audio do not replace these steps.

**Delivery status:** the runbook is prepared; no device was started and no Jev request was made in this turn. Use the offline preparation tool to create the acceptance folder:

## 1. Prepare the run

### Offline preparation bundle

Run from the repository root:

```bash
.venv/bin/python scripts/prepare_real_acceptance.py --output-dir artifacts/acceptance/real-preparation
```

Review `profile.json`, `preflight.json`, `manual-results.json`, and `README.md`. Keep media and identifying material out of the report directory. Model file existence and Chrome install metadata in `preflight.json` do not establish runtime readiness; manually verify the running server, effective model directory, Chrome runtime version, and PureSound status. The preparation tool runs offline and does not read environment variables or credentials, or connect to devices/services. If a model is missing or output fields are incomplete, mark this step `not_run` and stop preparation rather than claiming it is complete.

### Fixed environment and operating limits

- Record the acceptance date, macOS version, M4 model and memory, Chrome version, repository commit, Python/SDK versions, model versions, PureSound version, profile version, and server launch command. The operator fills these values in; do not guess.
- Use one active WebSocket connection at a time: one Chrome acceptance tab, with no other users connected to the same server. The scene has 1–3 participants who have agreed to take part.
- Bind the server to loopback:

  ```bash
  uv run presencekit --host 127.0.0.1 --port 8000
  ```

  This is the documented `presencekit` entry point and `server.main` arguments; the defaults are also `127.0.0.1:8000`. Do not bind to `0.0.0.0`. Check that the port is free first. If another port is needed, record the exact command and URL.
- Open `http://127.0.0.1:8000` in Chrome. Manually confirm the displayed app version matches the repository under test. The operator grants camera and microphone access only when pressing **Start**.
- Prepare anonymous scene cards using only `A`, `B`, and `C`. Record action times and expected outcomes, not names, face descriptions, voice traits, or raw conversation/session/track IDs.
- Do not record or export video, face crops, audio, transcripts, raw `/debug/events`, or messages containing identifiers. Evidence is limited to an anonymous case code, operator action times, pass/fail/not-run, approved numeric summaries, and error category.

### Load the fixed presence profile

The preparation tool should produce a profile that can be imported in the Settings drawer. Before starting, use **Settings → Import JSON** and verify the keys below. Use the actual imported values as authoritative and record any differences. This PoC acceptance profile differs from the **Recommended** preset; do not substitute Recommended.

| Setting key | Acceptance value |
|---|---|
| `tracker_mode` / `presence_model` / `intent_model` | `kalman` / `hmm` / `score` |
| `join_seconds` / `grace_seconds` | `1.5` / `10` |
| `face_roi` / `pose_roi` / `pose_enabled` | `[0.08, 0.02, 0.92, 0.98]` / `[0, 0, 1, 1]` / `true` |
| `fps` / `adaptive_fps` / `fps_low` / `fps_high` / `pose_interval` | `2` / `true` / `0.5` / `2` / `2` seconds |
| `audio_pipeline` / `asr_provider` | `puresound` / `none` |
| `reid_enabled` / `reid_consent` / `reid_show_crops` | `false` / `false` / `false` |
| `reid_ttl` / `reid_max_refs` / `reid_decision_timeout` | `600` seconds / `3` / `10` seconds |

Import JSON loads the remaining settings too. Verify that omitted performance, display, ASR, and Re-ID values do not change the intended profile. In particular, inspect and record the actual `adaptive_fps`, `fps_low`, and `fps_high` values; do not infer them from this table. Keep Re-ID TTL, capacity, and deadline settings in place while Re-ID is disabled so they can be checked in the Jev phase.

## 2. Phase one: device acceptance with cloud services off

Confirm `reid_enabled=false`, `reid_consent=false`, and `asr_provider=none`; keep cloud services off for this phase. Use `GET /api/reid_status` to check key status; if set, clear the current process key from the UI before the device cases. Import the profile before starting or refreshing, start one connection, and allow the models to warm up. Begin every case from a clean, recorded state. Mark a mismatch `failed`; do not relax thresholds to pass. A settings change creates a new profile and requires affected cases to be rerun.

| Case | Setup and timing | Expected result | Evidence / stop condition |
|---|---|---|---|
| `device.quiet_120s`, `device.quiet_120s_2`, `device.quiet_120s_3` | In separate runs, 1, 2, and 3 participants join and stay visible, then all remain silent for 120 seconds. | Silence alone does not set the session to `ENDED`. | Record start/end time and state for each group size. An early `ENDED` is `failed`; stop that case. |
| `device.occlusion_3s`, `device.turned_head_3s` | After A joins, separately cover the face and turn away for about 3 seconds, then recover. | A short loss of visibility does not falsely end a joined participant. | Record case and duration. An `ENDED` within the 3-second event is `failed`. |
| `device.last_leave_20s` | After A joins, A leaves the camera and interaction area. Record the departure time. | Tracker reaches `ENDED` within 20 seconds. | Record the manual time to `ENDED`; over 20 seconds or no end is `failed`. |
| `device.remaining_1`, `device.remaining_2`, `device.remaining_3` | Test 1, 2, and 3 joined participants in sequence. For groups of at least 2, have everyone join, then remove participants one at a time while at least one joined participant remains. | The session stays active while any joined participant remains. The one-person case confirms that a sole participant present does not end. | Record each group size and departure order. `ENDED` while someone remains is `failed`. |
| `device.bystander_not_extend`, `device.background_speech_not_extend` | After joined participants leave, separately have an unjoined passer-by cross the scene and provide background speech. | A passer-by or background speech alone does not join someone or keep the conversation alive; the final departure meets the `ENDED` deadline. | Record only anonymous scene and state. False join/continuation is `failed`. The PoC has no TV visual-semantic classifier, so “TV correctly identified” cannot pass. |
| `device.sensor_interruption` | With A joined, interrupt camera frames or the browser media track for 25 seconds, then restore it. | Sensor interruption is distinct from a person leaving and is not recorded as a normal participant departure. Resuming media may begin a new session according to the transport lifecycle. | Record interruption/recovery time and sensor state. Treating the gap as a person leaving is `failed`. |
| `device.same_page_reconnect` | Keep the same loaded tab and interrupt the WebSocket transport so it reconnects. Observe the session counter. | Recovery creates a new session; still-valid in-memory page state is available for a later re-ID step. A transport drop is not a person leaving. | Record reconnect, session change, and state continuity. Do not inspect or copy the page capability. |

### 30-minute performance gate

Run `device.normal_load_30min` for 30 minutes after warm-up, using the fixed profile and normal ongoing interaction. All gates must pass: p95 of fresh, once-per-second recent-5-second audio RTF samples ≤ `0.5`; audio dropped `0`; uncaught server errors `0`; the audio queue does not keep growing; and mean RSS over the final 10 minutes minus mean RSS over the first 10 minutes ≤ `30 MiB`.

The UI has no measurement button; the operator manually loads a passive recorder that reads numeric messages from the existing WebSocket. It does not call `getUserMedia`, make network requests, change settings, or retain audio, images, conversation IDs, or raw events. On the production page, press **Start** and confirm the device stream first. Then enter this in the same-origin page’s Chrome DevTools Console:

```js
await import('/static/js/real-acceptance-recorder.js')
RealAcceptanceRecorder.start({seconds:1800,warmup:10})
```

After a 10-second warm-up, the recorder samples once per second for 1,800 seconds, then stops automatically. Keep the Chrome tab in the foreground; a disconnect, capture stop, or counter reset fails the continuity gate. At completion, run `RealAcceptanceRecorder.report()` in the Console to inspect the numeric report and `RealAcceptanceRecorder.download()` to download `real-acceptance-metrics.json`. An early `RealAcceptanceRecorder.stop()` retains numeric evidence but cannot pass the 30-minute gate. Download each run before starting another measurement, which replaces the in-memory record. Require at least 95% sample coverage and at least 570 samples in each endpoint 10-minute RSS window.

Recorder `observed_errors` counts browser-visible errors; it is not an authoritative audit of uncaught server errors. `manual_checks.server_uncaught_errors` remains `unverified`; the operator must inspect only the dedicated server console’s error count and enter that number in the worksheet. Keep the gate `pending_real` without independent evidence. Observed Jev event calls, tokens, latency, and model are not a complete bill or global budget. The report’s top-level status always stays `pending_real` and needs human review. All numeric gates, the manual server-error audit, and participant ground truth must pass; `thirty_minute_coverage` alone establishes duration and continuity, not overall acceptance.

If RTF exceeds the limit, audio drops, errors increase, the queue keeps growing, or the device overheats or becomes unstable, stop the performance case, stop media safely, and record only anonymous metrics and the final state. Do not retain audio/video for later analysis.

## 3. Jev approval and setup

Proceed only after device acceptance and owner approval. Before any face crop can be sent:

1. Explain to every participant that cross-session face comparison is being tested and that crops are sent to the third-party Jev (Typesafe) service. Obtain explicit informed consent applicable to this test, and account for bystanders who may enter the frame.
2. Have the service/legal/privacy owner confirm the applicable Typesafe/Jev data-processing, retention, deletion, and use terms. This runbook makes no assumption about provider retention periods, deletion guarantees, or recognition FAR. Do not send images until the terms are confirmed.
3. Have the budget owner enter the per-call price, currency, approved limit, and accountable owner. Do not enable the cloud service while price or an approved cap is unknown.
4. Confirm the local server has only this acceptance connection. `POST /api/reid_reset` wipes references across **all pages/connections**; call it only after confirming no other user session is active. Prefer the UI’s **Clear reference faces** action for normal cleanup; it also clears refs across server connections.
5. Enter the Jev key only in the local server settings panel. Do not paste it into a terminal, shell command, chat, worksheet, or screenshot. `GET /api/reid_status` reports only whether a key is set and its source; it never returns the key. Clearing the key resets the current process key, including when it came from the environment, but does not remove `TYPESAFE_API_KEY` from the launch environment. Stop the dedicated server and remove that environment value before any later server launch that must remain keyless.
6. Import the same presence profile, then set `reid_provider=jev`, `reid_model=jev-1.13.0`, `reid_enabled=true`, `reid_consent=true`, `reid_show_crops=false`, `reid_ttl=600`, `reid_max_refs=3`, and `reid_decision_timeout=10`. The current UI offers `jev-latest` and pinned `jev-1.13.0`; prefer the pinned option for a reproducible run. Do not type a model value unsupported by the UI, which may clamp it to a different value. Record the actual `response.model` returned by Jev. `reid_consent` is an application switch, not a record of human consent; the operator must verify consent separately under the organization’s process. Confirm `asr_provider=none`.

With up to 3 historical conversations, 3 participant references per conversation, and 3 query participants per new session, the theoretical maximum is 27 pairwise calls per round. An uncertain-score retry may increase the upper bound to 54. Comparisons are sequential with a 10-second overall deadline, so whether they finish must be measured under approved budget and normal service conditions. These caps are per driver/connection and reset on reconnect; they are not a cross-session or global hard cap. The operator must tally calls and estimated cost throughout the run. Stop Re-ID and testing at the pre-approved limit. Do not increase the timeout to hide latency.

## 4. Jev acceptance matrix

Use anonymous A/B/C and controlled actions. Every new conversation starts immediately with a new session; handoff is allowed only for an explicit, unique recognition result. For each case record setup, start/result time or duration, handoff boolean, call count, score if exposed by the UI/report, estimated cost, and failure category. Do not record/export crops, faces, names, raw IDs, or raw events.

| Case | Setup and action | Pass condition | Stop / failure condition |
|---|---|---|---|
| `jev.same_person` | A creates and normally ends a historical conversation. Within the 600-second same-page window, start a new session with A alone. | A provisional conversation starts first. A clear unique match hands off to A’s historical conversation within 10 seconds. | If A is confirmed present and the image is usable but no handoff occurs within 10 seconds, mark `failed`. If the service is unavailable or evidence is inadequate, mark `not_run`. A wrong/ambiguous handoff is `failed`; do not extend the deadline. |
| `jev.different_person` | History contains A only. Start a new session with B before TTL expiry. | No handoff; B stays in a new unlinked conversation. | Linking B to A’s history is `failed`; stop the round. |
| `jev.member_overlap` | Historical conversation contains A+B; the new session has B+C. | If B is uniquely recognized, hand off to the A+B conversation. This does not mean C was recognized. | A handoff to another conversation, treating C as recognized, or failure to decide within 10 seconds after a unique result is a failure/timeout. |
| `jev.ambiguous_overlap` | Create at least two historical conversations so query participants in the new session correspond to different histories; include those people in the new session. | No handoff; keep the provisional conversation. | Any handoff is a wrong association: stop immediately and mark `failed`. |
| `jev.unknown_timeout` | Within the scope of consent, use a low-quality face view or no matching reference and measure real elapsed time from session start to final decision. Never use a non-consenting stranger’s face as a substitute. | Fail closed within 10 seconds: no handoff; session/audio continues. | Stop if no clear consented sample is available, the setup is unsafe, or a bystander may be captured. Any handoff on an uncertain result or blockage past 10 seconds is `failed`. |
| `jev.service_error` | With service-owner approval, use a controlled method to verify fail-closed behavior on error/timeout; do not simulate this by damaging a production key. Record real elapsed time. | Error/timeout resolves within the 10-second deadline without handoff or blocking the session/audio. | Waiting past 10 seconds, blocked media, or handoff on error is `failed`; stop further calls. Do not raise the timeout. If it cannot be triggered safely or the service is unavailable, mark `not_run`. |
| `jev.revoke` | During a comparison, a participant withdraws consent. Turn off `reid_consent` in the UI, confirm cancellation/display clearing, then check that a late result cannot be used. | Face refs/thumbnails clear; even a late result cannot hand off; the session continues. | If revocation cannot be confirmed, old crops/handoff remain, or confirmation would require retaining media, disable the feature and mark `failed`. |
| `jev.ttl_600s` | Create and normally end A’s historical conversation with consent. Wait 600 seconds, then start a session with A. | Old refs expire; no handoff to the expired history. | Reconnecting to the expired history after 600 seconds is `failed`. Do not mix pre-reset data into this case. |
| `jev.max_3_history` | Create and normally end A→c1, B→c2, and C→c3 in sequence. Then start A+B: each matches a different history, so the result must remain ambiguous and create provisional c4. End c4 normally; the oldest c1 should be evicted, leaving 3 histories. | Use an approved interface’s safe numeric value to confirm `history_conversations=3` and confirm there was no handoff before c4 was created. A or B may later match retained c4; that does not show whether c1 remains. If no safe history-count/eviction evidence is available, mark `not_run`. | Any ambiguous handoff, more than 3 histories, or eviction of active refs is `failed`. Do not interpret a later A→c4 match as c1 remaining, or a failed A match as proof of c1 eviction. |

### Cleanup cases after consent approval

`device.refresh_clears` and `device.server_restart_clears` cannot be run with cloud disabled and an empty reference history; empty-to-empty is not evidence of cleanup. First complete Jev approval, obtain participant consent, and establish a **nonempty** reference history. Then separately reload the page and restart the dedicated server. Expect valid refs to be cleared and record only an anonymous boolean, never IDs or images. If prerequisites in `manual-results.json` are unmet, leave the cases `pending_real`/`not_run`; they cannot pass in the device-only phase.

Only mark an applicable Jev control-flow case passed when its fixed conditions are met and every cloud call stays within the approved budget. This does not establish recognition accuracy, FAR/FRR for general users, or provider data-handling quality. Those require a separately designed, consented evaluation with adequate statistical power.

### Provider and billing references

Official Typesafe pages checked on **2026-10-23**; recheck them and the applicable account contract before the run. The [Images API](https://docs.typesafe.ai/images) is marked experimental/Dreamer-ready and requires the `Danger-Zone` header; this is not a production-quality guarantee. The [models and rates page](https://docs.typesafe.ai/models) describes rates by input tokens and organization rate limits; check the actual account’s console and contract for price and limits. The [System One API](https://docs.typesafe.ai/api/system-one) response exposes `model` and `usage.input_tokens`, which are the run’s evidence for actual model and input usage. The [submitted-data use and retention article](https://help.typesafe.ai/articles/5974925716-how-is-submitted-data-used-or-retained) and [Typesafe privacy policy](https://typesafe.ai/legal/privacy-policy) do not make “not used for training” equivalent to “not retained” or a zero-data-retention guarantee; the public pages do not establish a fixed retention period. Confirm the current contract and terms with the responsible owner.

## 5. Stop and clean up

Immediately disable `reid_enabled` and `reid_consent`, stop testing, and stop the dedicated server if there is any wrong handoff, ineffective consent withdrawal, budget-limit breach, bystander image, unclear provider terms, exposed key, blocked media/session, or device instability. After confirming there are no other user connections, clear references using the settings action or `POST /api/reid_reset`; do not retain the API’s count as identity data. Clearing the UI key resets the current process key, including one loaded from the environment, but does not remove `TYPESAFE_API_KEY` from the launch environment. Stop that process and remove the variable from any later launch environment that must remain keyless. Finally revoke Chrome’s local camera/microphone site permissions and close the acceptance tab.

Keep only anonymous numeric summaries and per-case `passed`, `failed`, `not_run`, or `pending_real` status. Anything not measured, not executed, missing tool fields, or lacking adequate evidence remains `not_run`/`pending_real`; synthetic results are not a substitute. The owner confirms the conclusion after reviewing numeric results, the profile, and the consent record.

## Worksheet template

```text
Date / timezone:
Operator code:
Repository commit:              Profile hash / version:
macOS / M4 / memory:            Chrome:
Python / PureSound SDK / models:
Server launch args / URL:       Connection count:
Scene code (A/B/C only):
Case ID:                        Status: passed / failed / not_run / pending_real
Setup and expected result:      Actual outcome summary:
Start/end time or duration:     Approved numeric values:
Recent-5s RTF p95:              Drops:             Errors:
Queue trend:                    First 10-min mean RSS: Last 10-min mean RSS:
Jev calls / estimated cost / approved cap (Jev phase only):
Stop reason / follow-up:
```

Related pages: [PoC acceptance contract](acceptance.md), [Re-ID design and retention](reid.md), [Privacy](privacy.md), and [Configuration](configuration.md).
