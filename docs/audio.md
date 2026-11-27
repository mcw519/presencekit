**English** · [繁體中文](zh-TW/audio.md)

# Audio: PureSound voice isolation + vision cross-check

Audio answers "is the user speaking?". Two pipelines are available (`audio_pipeline`):

| Pipeline | Where | How it decides "speech" |
|---|---|---|
| `energy_vad` (default) | Browser | Loudness above an adaptive noise floor (`vad_k`, `vad_min`, `vad_hang`). No install needed. |
| `puresound` | Server | Streaming voice isolation, a VAD on the isolated voice, then a per-chunk cross-check against vision. |

A loudness detector cannot tell the user from a TV, music, or someone talking across the room. The `puresound`
pipeline fixes this in two steps: an acoustic model keeps mostly the **near** voice, and the camera then vetoes
whatever the acoustic model gets wrong.

```mermaid
flowchart TD
  MIC[mic chunk 20 ms] --> RAW[raw VAD]
  MIC --> ISO[PureSound isolation] --> CV[clean VAD]
  CAM[vision snapshot, held 0.75 s] --> XC{cross-check}
  RAW --> XC
  CV --> XC
  XC -- agree / silence --> KEEP[unchanged]
  XC -- leak / false_accept --> ATT[−40 dB, not speech]
  XC -- false_reject --> MIX[30% isolated + 70% raw, speech]
  XC -- acoustic_only --> KEEP
```

![Audio cross-check](img/audio_crosscheck.png)

## Install

PureSound is an optional dependency (the standalone SDK from the public
[PureSound](https://github.com/mcw519/puresound) repository, Apache-2.0, pinned to a commit). It needs
Python ≥ 3.12.

```bash
uv sync --extra puresound
uv run python scripts/fetch_puresound_model.py --clone     # or --from /path/to/a/puresound/checkout
```

The model (`dpcrn_curriculum_v2`, 4.9 MB + manifest) is copied into `models/` only if its sha256 matches the
PureSound catalogue. Note that a plain `uv sync` removes the extra again. If the SDK or model is missing,
choosing `puresound` shows the reason in the audio panel and the state machine keeps using the browser VAD.

## Step 1: isolation and dry blend

The browser captures the microphone with an `AudioWorklet`, resamples to 16 kHz mono and sends 20 ms chunks
(320 samples) over the WebSocket. PureSound uses STFT 512, a 10 ms hop, and 30 ms look-ahead. At the default
`iso_dry_blend = 0.9`, it outputs `0.9 × enhanced + 0.1 × aligned input`, matching the upstream default
sample for sample. Algorithmic latency is 52–62 ms.

In **Settings → Detailed settings → Audio**, choose **PureSound** to adjust **Isolated voice share (dry blend)** from 0.01 to 1.00.
The new value applies to the isolation output (`vi.wav`) before the vision cross-check and briefly resets the stream.

## Step 2: the clean VAD

The model has no VAD head, so the **clean VAD** is an energy detector on the *isolated* signal: 10 ms frames, an
adaptive noise floor (falls instantly, rises about 3 dB/s), speech when the level exceeds
`max(clean_vad_min_db, floor + clean_vad_margin_db)` for 60 ms, with `clean_vad_hang` (0.3 s) of hangover. The
clean VAD reads the same signal as `vi.wav`, before vision-controlled dry recovery and output gain.
The same detector on the raw input gives the **raw VAD**, used only to detect missed speech.

## Step 3: the vision cross-check

With the default adaptive settings, vision runs at 0.5–2 frames per second and audio at 50 chunks per second. Every chunk reads the
latest tracking snapshot, and the three facts "someone in the ROI", "someone within `near_range`" and "everyone is
beyond `near_range`" are each **held for `xcheck_hold`** (0.75 s). "Everyone beyond range" holds only if no near
person was seen during that time, which protects the first syllable of a near speaker.

![Decision table](img/audio_decisions.png)

| Clean VAD | Raw VAD | Vision | Decision | Output | Speech? |
|---|---|---|---|---|---|
| yes | – | nobody in the ROI | `leak` | attenuated to `xcheck_floor_db` (−40 dB) | no |
| yes | – | everyone beyond `near_range` | `false_accept` | attenuated | no |
| yes | – | someone within `near_range` | `agree` | unchanged | yes |
| no | yes | someone within `near_range` | `false_reject` | aligned raw mix if the selected recovery mode allows it | yes, except when `audio` rejects the onset |
| otherwise | | | `silence` | unchanged | no |
| any | | vision uncertain | `acoustic_only` | PureSound output as is | = clean VAD |

- **Vision uncertain** means: not started, decision frame age exceeds `xcheck_stale` (2 s), only low-confidence
  detections, or the cross-check is off.
- **Distance** is used only when distance estimation is on. Without it, "near" means "in the ROI", and
  `false_accept` cannot happen. `xcheck_require_calib` also demands a distance calibration first.
- Gain changes ramp in 10 ms (`xcheck_attack`) and out in 50 ms (`xcheck_release`) to avoid clicks.

The audio panel retains state-entry counts and also shows each decision's duration and share of all processed
PCM time, including `silence` and `acoustic_only`. These are decision statistics, not recognition accuracy.
`decision_duration_s`, `decision_ratio`, and `classified_audio_s` accumulate within an `AudioStream`; model
resets preserve them, while a new stream starts from zero. Dropped input and idle wall time are excluded.

At stable 0.5 FPS, the latest frame can be almost 2 seconds older than audio simply because of sampling.
The alignment row therefore shows the current sampling interval, time since frame completion, server
receive-to-completion time and its queue component. The `vision_timing` fields use milliseconds. The signed
capture gap uses the shared browser clock; server timings use the server monotonic clock. Cross-check
freshness uses the greater of server time since frame receipt and the nonnegative audio/video capture gap,
so finishing a slow frame no longer makes it fresh again. Without client timestamps, server residence time
is a lower bound. JPEG encoding and network latency are not measured separately. Each audio batch keeps
the timestamp and decision context it actually used, even if a new frame finishes during audio processing.

Example: the TV is on and nobody is in front of the device. PureSound passes some of the TV voice (it sounds
close enough). The clean VAD says "speech", vision says "nobody in the ROI" → `leak`: the chunk is attenuated by
40 dB and does not count as speech. The session does not start.

### Missed-onset recovery modes

With PureSound and vision cross-check enabled, open **Settings → Detailed settings → Audio → Missed-onset recovery**.

- **Original mix rule** (`legacy`, default) allows the existing dry mix on every `false_reject`.
- **Audio onset only (experimental)** (`audio`) allows extra raw audio when the acoustic onset qualifies.
  The camera still supplies the nearby-person check, so this mode can use the usual low frame rate.
- **Mouth motion and audio (experimental)** (`audiovisual`) also accepts fresh mouth motion from the near track.

SDK callers can set `Config(audio_pipeline="puresound", xcheck_recovery_mode="audio")` and submit video and audio
on the same monotonic clock. Configs saved with `xcheck_mouth_recovery=true` map to `audiovisual`.

The experimental modes adjust the extra raw-audio mix for a `false_reject` decision. Clean VAD reads `vi` before
that mix. When vision exceeds `xcheck_stale`, both modes fall back to `acoustic_only`: no new raw recovery is
allowed, the mix returns toward the configured `vi` blend, and `speech` follows clean VAD.

- **Mouth motion.** YuNet provides eye and mouth-corner landmarks. The eyes align consecutive grayscale face
  patches for translation, in-plane rotation and scale. OpenCV calculates dense optical flow. The median
  cheek motion vector estimates shared movement; large head jerks (`> 4%` of mouth width) zero out the score,
  and face boxes must stay matched and stable across consecutive frames. After subtracting cheek motion, the
  score takes the minimum of (a) the mouth's 80th-percentile flow magnitude minus the cheeks' 80th percentile
  and (b) the vertical divergence between the upper and lower lip medians, both divided by mouth width. The
  median of three consecutive scores must exceed about 0.00459 for the currently selected near track. Skipped,
  missing, ambiguous and sparse observations break the evidence sequence.
- **Acoustic onset.** Once raw VAD confirms an onset, use the first 60 ms and the previous 200 ms of raw audio.
  Accept either (a) an RMS increase of at least 8 dB with the 1.8–6 kHz band at least 12 dB below the
  150 Hz–1.8 kHz band, or (b) a strong broadband speech onset (RMS rise ≥ 18 dB, 150 Hz–1.8 kHz rise ≥ 15 dB,
  and high-to-low tilt ≤ 0 dB). If the initial burst already rose ≥ 8 dB (e.g. a voiceless consonant), keep
  scanning for up to 140 ms against the pre-onset baseline so vowel arrival still qualifies. Allow recovery for
  output blocks overlapping the first 160 ms of that onset, with the usual mix smoothing. When distance
  measurement confirms only near person(s) in view (`any_far = False`) and the current utterance qualified at
  onset or was already confirmed by `agree`, recovery continues while raw speech remains active so PureSound
  dropouts mid-sentence or at ~0.7 m do not silence a solo user. The check uses raw samples already received
  while PureSound is processing earlier audio.

Without either cue, the mix returns toward the configured `vi` blend. The optional PureSound SDK onset guard and
`iso_dry_blend` also affect `vi`. Use `iso_dry_blend=1` with the SDK onset guard off when comparing with the
offline experiment. In both `audio` and `audiovisual` modes, a `false_reject` reports `speech=True` to the
session and speech-gated ASR only while a qualifying onset (or fresh mouth motion in `audiovisual`) permits
recovery. Rejected blocks keep the `false_reject` diagnostic label and report `speech=False` without
refreshing the session's speech timer. A quiet participant can still keep the session open through the
configured presence policy. Only `legacy` reports `speech=True` unconditionally on `false_reject`. Mouth
motion contributes timing evidence and can also arise from silent activity. The acoustic thresholds were
explored on one near/far recording.

**Check capture cadence and latency for the mouth cue.** Mouth flow requires four successive timestamped frames to produce
three scores, with gaps of at most 85 ms. The WebSocket also requires comparable audio/video timestamps and
an effective frame age of at most 85 ms when making the audio decision. The default 0.5–2 fps cadence is too
sparse, so recovery at those rates relies on the acoustic cue. To run the video experiment, disable adaptive
fps and set fixed camera fps to up to 24. This mode runs YuNet on timestamped frames even in stable scenes
to obtain new landmarks. It adds vision work and sends more video. Audio keeps using the available evidence
without waiting for another frame or adding another PureSound pass.

The audio panel shows the cue, mouth score, age and frame interval. Check the processed frame rate,
`mouth` timing in the performance panel, and audio processing and queue P95 before choosing a capture rate.
The recording JSON includes sampled `recovery` diagnostics. To pace a local recording through the same
WebSocket runtime and compare frame rates without browser or network overhead, run
`python experiments/puresound_mix/realtime.py BUNDLE OUTPUT.json` in the project's environment.

In one nine-second replay, 0.5 and 2 fps produced no usable mouth scores. At 15 and 24 fps, scores were
available for 55 of 60 and 59 of 62 sampled audio reports, respectively. At 24 fps, enabling recovery
increased vision P95 from 2.45 to 13.88 ms. Audio processing P95 was 6.33 ms with the feature off and
6.31 ms with it on. Measure on the deployment device since inference load, browser and transport vary.

## Step 4: what the state machine sees

With the PureSound pipeline active, the **corrected** speech flag replaces the browser VAD:

- `onset_entry_only` (on): a session opens **only** on a corrected speech onset plus a visible, confident person
  within `leave_distance` seen in the last 1.5 s. Passers-by and voices alone never open a session.
- `onset_needs_vision` (on): returning from AWAY by speech also needs a visible person.
- `speech_join = auto` attributes clean speech to the only candidate, in every priority mode.
- If the audio stream stops for more than 1 s the browser VAD takes over again.

When the group changes (`NEW_GROUP`, a new session, or a disjoint replacement of the participant set) the
isolation state is reset (`iso_reset_on_change`), costing about 32 ms of output. Adding or removing a member while
an existing participant stays in view keeps the isolator running.

## Listening and recording

- **Monitor**: the audio panel can play the clean or raw stream back to you. Use headphones. Monitoring adds a
  small jitter buffer (≈ 80 ms) and your network round trip; decisions are made on the server and are not delayed
  by it.
- **Video recording**: start the camera, then click **Record video**. The video includes the camera, face and
  body boxes, skeletons, and visible ROIs. It uses the microphone with browser VAD or the final cross-checked
  audio with PureSound. Stop to play it in the page or download it as MP4 or WebM, depending on the browser.
  The browser records up to 10 minutes per clip and keeps the preview in page memory.
- **Video and audio bundle**: for a PureSound recording, choose **Download video and audio tracks (ZIP)**.
  It contains the same video, three sample-aligned 16 kHz WAV tracks (`raw` for the microphone, `vi` for the
  isolated voice, and `out` for the final output), and a JSON decision timeline. The WAVs share a sample clock
  and the video follows the browser clock.

## Performance

On an 8-core x86 test server, single-threaded ONNX Runtime:

| Item | Value |
|---|---|
| Isolation real-time factor | 0.22–0.27 (≈ 5 ms per 20 ms chunk) |
| Algorithmic latency | 52–62 ms |
| WebSocket + queue + compute, p50 / p95 | 6.0 / 7.8 ms |
| End-to-end estimate | ≈ 85–100 ms |
| Server CPU, 2 fps video + one face | video only 3.8% → video + audio 31.4% of one core |

Audio is much more expensive than vision. Measure on your target device before enabling it there.

## Known limits

- `near_range`, `xcheck_hold` and the clean-VAD levels are not yet tuned in many rooms; the −50 dBFS floor depends on
  microphone gain.
- The browser's own echo cancellation and noise suppression run before PureSound.
- A TV inside the ROI and closer than `near_range` cannot be told apart from a person.
- With several people near the device, the cross-check knows *someone near* is speaking, not *who*.
- Distance errors (for example a low camera angle) can cause wrong `false_accept`; calibrate distance first.

## License

PureSound code, SDK and weights are Apache-2.0. Its training data are not covered by that license; check before
commercial use. See [Licenses](licenses.md).
