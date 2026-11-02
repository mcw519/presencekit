**English** · [繁體中文](zh-TW/acceptance.md)

# PoC acceptance

This plan targets **Apple M4 + Chrome + the local server, with one active connection and 1–3 participants**. It prioritizes avoiding a false ending for anyone who joined the conversation. A passing synthetic suite is evidence about the tested software paths; it does not establish camera performance or identity accuracy.

## Run the suites

The runner writes both `acceptance.json` and `acceptance.md` under `artifacts/acceptance` by default. It records an allowlist of host and package metadata, case outcomes, thresholds, and limitations; it does not dump environment variables or media.

```bash
uv run python scripts/run_acceptance.py --suite functional
uv run python scripts/run_acceptance.py --suite pipeline --pipeline-seconds 10
uv run python scripts/run_acceptance.py --suite browser --browser-seconds 10
uv run python scripts/run_acceptance.py --suite browser-reid --browser-seconds 10
.venv/bin/python scripts/run_acceptance.py --suite browser-reid --browser-seconds 10
uv run python scripts/run_acceptance.py --suite soak --soak-seconds 1800
uv run python scripts/run_acceptance.py --suite all --pipeline-seconds 10 --browser-seconds 10 --soak-seconds 1800
```

`functional` uses injected time and synthetic tracker detections plus an in-memory identity mock. `pipeline` starts an isolated local server, loads the local models, and replays synthetic PCM and video over loopback. It performs a 5-second warm-up by default; change that with the adapter's `warmup_seconds` setting. `browser` and `browser-reid` share isolated headless Chrome, a temporary profile, a loopback server and synthetic-media fixture. The fixture supplies canvas video and audio through the app media path; native `getUserMedia` is not called. Both exercise the actual capture timer, AudioWorklet, binary JPEG/PCM transport and WebSocket lifecycle. `browser` checks transport lifecycle with re-ID disabled. `browser-reid` uses the `legacy` tracker with `rules` presence/intent and `vision`, full-face ROI, `leave_distance=10m`, 2 fps and pose disabled; it does not use the functional HMM/score `PROFILE_CONFIG`. It toggles consent through production configuration, captures production crops and exercises production `CrossSessionReid` with an explicit handoff from a provisional conversation to the earlier one. Only the isolated child server injects a local mock `CloudReid`; production has no test hook or mock option. It also checks withdrawal during a pending comparison, rejection of a delayed result, continued JPEG/PCM processing, and cleanup. The consent flag here is test setup and does not represent real participants' consent. Crop bytes and page capabilities stay in memory; reports retain only allowlisted booleans and numeric measurements, not identifiers or image data. This suite tests control flow, not biometric accuracy, physical camera/microphone behavior, Jev availability or RTF. Neither browser suite contacts Jev or cloud ASR. `--browser-seconds` controls the stable-stream interval for both; lifecycle checks take additional time. `soak` uses the measured pipeline and defaults to 10 seconds of warm-up followed by 1,800 seconds of real wall time. A shorter soak smoke run cannot qualify the 30-minute gates. Pipeline and soak require local runtime files; the runner does not download missing assets.

The CLI accepts `--suite browser-reid` and shares `--browser-seconds` with the `browser` suite. In Python, `run_acceptance(..., suite="browser-reid", browser_seconds=10)` dispatches to `presencekit.acceptance_browser_reid.run_browser_reid(seeds=20, seconds=10)`. The Chrome fixture is `scripts/acceptance_browser_reid_fixture.js`.

`all` runs suites in this order: `functional`, `pipeline`, `browser`, `browser-reid`, `soak`. It can take at least 30 minutes because the soak duration is measured wall time. Choose a suite explicitly when iterating. `--seeds` controls deterministic functional scenarios; pipeline inputs use a fixed seed. `--output-dir` changes the report location. Add `--strict` to return exit code 2 if any selected or real-world acceptance remains `not_run` or `pending_real`; a failed case returns 1. Without strict mode, a report with incomplete coverage returns 0 unless a case failed. The report summary remains `incomplete` until all required cases are complete.

If Chrome or required local models are unavailable before a suite starts, the corresponding cases are `not_run`. Once browser execution begins, assertion failures and execution errors are `failed`. Browser automation uses a temporary Chrome profile and loopback only; it does not use a real camera or microphone. `real_chrome_devices` and `real_jev_quality` remain `pending_real`. TV visual-semantic classification remains `not_run` because this PoC does not implement it.

## Acceptance contract

| Area | Required outcome |
|---|---|
| Quiet participant | A joined participant can remain quiet for 120 seconds without ending the session solely for silence. |
| Occlusion | A joined participant can be occluded for 3 seconds without a false ending. |
| Group continuity | With up to 3 joined participants, the session remains active while any joined participant remains, even if others leave. A passer-by or background audio alone must not keep the conversation alive. |
| True departure | After the last joined participant leaves, the tracker reaches `ENDED` within 20 seconds. |
| New session | A later arrival starts a new session with a provisional conversation ID. A successful, unambiguous re-ID emits one `handoff` linking it to the prior conversation; the old session ID and event history remain intact. |
| Re-ID ambiguity | Only a clear member overlap that points to exactly one historical conversation can resume it. If candidates point to multiple conversations, the new session stays unlinked. Uncertain scores, timeout, missing faces, consent withdrawal, capacity limits, or service errors do not block the session and do not create a handoff. |
| Re-ID deadline | Resolve or abandon comparison within 10 seconds of the new session starting, measured in wall time. Presence states such as `AWAY` do not pause this deadline. |
| History | Keep references for up to 3 ended conversations, with up to 3 participant references per conversation. The 600-second history TTL starts at the last normal `ENDED`; reconnects and failed comparisons do not refresh it, and resuming then interrupting again preserves that original expiry. A conversation that never reached normal `ENDED` gets a fixed 600-second active-reference lease from its first interruption. Evict the least recently ended history when over capacity; never evict the active conversation to make room. |
| Consent and storage | Re-ID requires both enablement and explicit consent. Keep face crops in memory only; clear them on expiry, page release/expiry, consent withdrawal, reset, or server shutdown. Never persist references across reload or server restart. |
| Transport loss | Report sensing interruption; do not treat a disconnected camera/microphone transport as proof that participants left or as a normal `ENDED`. Reconnecting the same page may reuse its still-valid in-memory state. A page generation fences off stale workers, and app-owned cleanup checks leases every second even after the final WebSocket closes. |
| Sensor gap | Pause presence observation at the last completed frame plus 3 seconds, rather than at a delayed ticker wake, so host sleep does not count as participant absence. Resume the monotonic observation clock when a frame is processed again. |

## Performance qualification

Only the real-wall-time pipeline adapter can qualify throughput and memory stability. After warm-up, a 30-minute soak must meet all of these gates:

| Metric | Gate |
|---|---:|
| p95 of one-second samples of recent 5-second audio RTF | ≤ 0.5 |
| Audio dropped | 0 |
| Uncaught server errors during measurement | 0 |
| Audio ingress queue | No sustained growth; late-quarter mean ≤ early-quarter mean + 1 block, peak below the 25-block capacity |
| RSS growth | Mean RSS in the final 10 minutes minus mean RSS in the first 10 minutes ≤ 30 MiB |
| Measurement quality | At least 1,800 measured seconds and 95% fresh sample coverage; at least 570 samples in each RSS window |

The pipeline adapter uses a local synthetic audio signal and a blank or repeated local test-face scene. This evaluates server processing and resource trends; it does not qualify real Chrome capture, camera/microphone drivers, room acoustics, or Jev recognition quality. The original human-test audio RTF near 0.37 remains a separate unresolved diagnosis. Consult the generated soak report for the run's result; this plan is not evidence that the 30-minute gates passed.

## Current case coverage

For the operator procedure for physical M4/Chrome camera and microphone checks, performance evidence, consent, Jev acceptance, and cleanup, see the [real-device and Jev runbook](real-acceptance.md).

When Chrome and offline model prerequisites are available, the `browser-reid` suite currently returns these nine cases: `consent_required`, `provisional_conversation`, `same_page_handoff`, `nonblocking_comparison`, `consent_revocation`, `late_result_discarded`, `nonblocking_media`, `cleanup`, and `server_errors`. It verifies that the initial provisional conversation differs from the prior conversation, then checks the resumed handoff against the original session and conversation. A deliberately cancellation-resistant mock comparison must complete after cancellation (`completed_after_cancel` increases and the mock returns `.99`) before `late_result_discarded` is evaluated; the assertion checks that handoff and resume counts did not increase and the cleared UI stayed clear. The report remains the source for actual statuses; do not hard-code a pass total.

The production `state` message includes the tracker `session`. The UI clears old re-ID status and thumbnails on a new session and WebSocket reconnect, and when consent is withdrawn, re-ID is disabled, or references are wiped. Incoming re-ID status and handoff messages for an older session are ignored. These browser checks use synthetic media and a child-only mock; they do not validate presence accuracy, real camera/microphone behavior, Jev identity quality, or RTF thresholds. Pagehide cleanup is supported by the fixture.
The functional adapter currently exercises 120-second quiet, 3-second occlusion, a remaining participant, departure timeout, same person, different person, participant overlap, ambiguous overlap, uncertain result, backend error, expired history, re-ID deadline, same-page reconnect, page isolation, stale-lease rejection, history TTL preservation/expiry, reference clearing, and a long sensor-input gap. These use virtual time and mock identity where applicable. A 20-seed functional run reported **290 passed, 0 failed**; this covers only the functional suite. Background-speech flags and a low-intent passer-by are exercised, but they do not establish that the system can identify a television as the sound source. TV visual-semantic classification is explicitly `not_run` because that capability does not exist in this PoC. Missing model assets or stale pipeline measurements also produce `not_run` rather than a pass. Use the generated report for current case statuses and soak gates.

Every case uses one of `passed`, `failed`, `not_run`, or `pending_real`. Synthetic/mock cases can pass while the overall PoC remains incomplete. Real Chrome camera/microphone acceptance and a consented Jev quality/latency study remain `pending_real`; the mock proves control flow only, not biometric accuracy. See [Re-ID](reid.md), [Architecture](architecture.md), and [the report schema implementation](../presencekit/acceptance.py).
