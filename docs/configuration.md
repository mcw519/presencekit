**English** · [繁體中文](zh-TW/configuration.md)

# Configuration

Server-side parameters live in `presencekit/config.py` (`Config`), with their allowed ranges in `RANGES`. Values
sent from the page are validated and clamped, applied **to that connection only**, and any warnings are sent back.
`GET /api/config` returns defaults and ranges.

The **Settings** drawer groups parameters by pipeline module, in data-flow order:

```
input → detection → pose (opt.) → distance (opt.) → tracking → intent → presence → state / timeouts
      → audio (PureSound + cross-check) → speech recognition (cloud) → re-ID (cloud)      [performance, display: cross-cutting]
```

Each module starts with its switch or mode; the parameters below it appear only when they apply, and greyed-out
ones say why. *Advanced* parameters are hidden until **Advanced** is on (changed advanced values are always
shown). Each module has its own **↺ Reset**.

Settings are saved in the browser (`localStorage["presence-cfg-v1"]`) and re-sent on every connection; they can be
exported and imported as JSON. Saved settings override newer defaults, so press **Defaults** after upgrading if
you want the new ones.

## Grouped view (v0.13)

`Config` keeps its flat keys: they remain the storage, the wire format (`/api/config`, WebSocket `config`
messages, presets, exported JSON) and the reference below, and they work unchanged in v0.13. In addition every
key belongs to exactly one group in `presencekit.config.GROUPS` (`core`, `tracker`, `presence`, `intent`,
`vision`, `pose`, `perf`, `audio`, `asr`, `reid`). `cfg.group("reid").reid_threshold` is the same value as
`cfg.reid_threshold`; `cfg.group(name).update({...})` applies the normal validation to that group only, and
`cfg.as_groups()` returns a nested copy. Server-process settings stay environment variables.

> **Deprecation planned:** direct flat-key access from Python code is planned to be deprecated in the **next**
> version in favor of the grouped view. v0.13 emits no warnings, and the flat wire format is not affected.

## Dependencies between modules

| Module | Switch / mode | Notes |
|---|---|---|
| Input | — | `fps` is ignored while `adaptive_fps` is on; `vad_*` only with `audio_pipeline = energy_vad` |
| Detection | — | `npu_precision` only with `embed_backend = npu`; `cos_match_int8` only with int8 / npu |
| Pose | `pose_enabled` | `roi_rule = body` needs pose; `turned_away_seconds` needs `turned_away_early` |
| Distance | `distance_enabled` | `shoulder_width_m` needs pose; feeds intent `w_distance` and the audio `false_accept` case |
| Tracking | `tracker_mode` | `coast_*`, `gate_mahalanobis`, `appearance_*` only with `kalman` |
| Intent | `intent_model` | weights and thresholds only with `score`; `intent_weighted_presence` needs `presence_model = hmm` |
| Presence | `presence_model` | probability parameters only with `hmm`; `present_frames` / `absent_frames` only with `rules` |
| State / timeouts | — | `present_max_silence` only in `vision` mode; `present_timeout` used outside `vision` |
| Audio | `audio_pipeline` | everything else only with `puresound`; `xcheck_*` need `xcheck_enabled`; `false_accept` and `xcheck_require_calib` need distance |
| Speech recognition | `asr_provider` | needs `audio_pipeline = puresound`; some options are model-specific |
| Re-ID | `reid_enabled` | also needs `reid_consent` and an API key |
| Performance | — | sub-parameters need their saver switched on |
| Display | — | this browser only |

## Presets

*Recommended* is the field-tested configuration; *Low camera*, *Single person*, *Group* and *Noisy room* build on
it and list only the differences. *Low power* changes only performance settings, on top of whatever is active.

| Preset | Settings |
|---|---|
| Recommended | tracker `kalman`, presence `hmm`, intent `score`, pose on, distance on, `audio_pipeline` `puresound`, cross-check on, `near_range` 1.6 m, `min_conf` 0.4, `leave_halflife` 1.5 s, `force_detect_seconds` 2 s, `join_seconds` 1.5 s, `grace_seconds` 10 s, face ROI (0.25, 0.1, 0.8, 0.89), body ROI (0.18, 0.04, 0.85, 0.92) |
| Low camera | Recommended + `min_conf` 0.35, `leave_halflife` 2 s, `force_detect_seconds` 1.5 s, face ROI (0.2, 0.05, 0.85, 0.95), body ROI (0.1, 0, 0.9, 1), `body_min_conf` 0.35 |
| Single person | Recommended + face ROI (0.25, 0.05, 0.75, 0.95), `present_timeout` 45, `empty_timeout` 5, `new_group_seconds` 3, `absent_frames` 3 |
| Group | Recommended + face ROI (0.08, 0.02, 0.92, 0.98), `join_seconds` 3, `grace_seconds` 20, `present_frames` 2, `absent_frames` 4, `present_timeout` 90, `empty_timeout` 8, `new_group_seconds` 5, `cos_match` 0.42, `fps` 2, `near_range` 2 m |
| Noisy room | Recommended + `priority_mode` `vision`, `vad_k` 3.5, `vad_min` 0.02, `vad_hang` 300 ms, `min_conf` 0.5, `near_range` 1.2 m, `base_timeout` 15, `present_timeout` 45, `empty_timeout` 5 |
| Low power | motion gate, embedding throttle (5 s), adaptive fps 0.5–1, ROI crop, `max_width` 480, pose and distance off |
| Defaults | back to the defaults in the tables below |

## Environment variables

| Variable | Meaning |
|---|---|
| `PRESENCE_TOKEN` | When set, the page, API and WebSocket require `?token=` |
| `PRESENCE_MODEL_DIR` | Where models are stored (default: `models/` of a source checkout, else `~/.cache/presencekit/models`) |
| `PRESENCE_ORT_PROVIDERS` | ONNX Runtime execution providers, e.g. `coreml,cpu` or `auto` (default `cpu`); see [SDK and ONNX Runtime](sdk.md) |
| `PRESENCE_ORT_THREADS`, `PRESENCE_ORT_GRAPH_OPT` | Threads per model (0 = ORT decides) and graph optimisation level (`disable`/`basic`/`extended`/`all`) |
| `PRESENCE_FACE_EMBEDDER=hist` | Do not load SFace; use the HSV histogram fallback |
| `PRESENCE_NPU_DELEGATE` | External TFLite delegate library for `embed_backend = npu` (server-side only) |
| `ASSEMBLYAI_API_KEY` | Key for the optional speech recognition |
| `TYPESAFE_API_KEY` | Key for the optional re-ID service; `REID_JEV_URL` overrides its endpoint |
| `PRESENCE_LOG_*`, `PRESENCE_EVENT_BUFFER`, `PRESENCE_VISION_LOCK_TIMEOUT` | Logging, see [Logging](logging.md) |

`npu_delegate_path` has no UI and cannot be changed over the WebSocket; it is set only from
`PRESENCE_NPU_DELEGATE`.

## Parameter reference

Generated from the code and the UI string tables by `scripts/gen_config_docs.py`; a test fails if it is out of
date. Labels and help texts are the same ones shown in Settings.

<!-- BEGIN GENERATED: scripts/gen_config_docs.py -->

### Input (camera / microphone)

| Key | Default | Range | Description |
|---|---|---|---|
| `fps` | 1 | — | **Frames per second** *(browser only)* — How many frames per second are sent to the server. Higher is more responsive but costs bandwidth and CPU. |
| `vad_k` | 2.5 | — | **VAD sensitivity (threshold factor)** *(browser only)* — Speech is when the level exceeds noise floor × this factor. Smaller is more sensitive. |
| `vad_min` | 0.005 | — | **Minimum speech level** *(advanced, browser only)* — Absolute level threshold (RMS); raise it in loud rooms. |
| `vad_hang` | 400 | — | **Speech hangover** *(advanced, browser only)* — How long "speaking" holds after the level drops, so sentences are not cut in the middle. |

### Vision detection

| Key | Default | Range | Description |
|---|---|---|---|
| `face_roi` | (0.2, 0.05, 0.8, 0.95) | box (x0, y0, x1, y1) | **Face ROI (x0,y0,x1,y1)** — A face counts only if its center is inside this (cyan) box. Drag it or its corners directly on the video. |
| `min_conf` | 0.4 | 0 – 1 | **Minimum detection confidence** — Detections below this value count as neither "someone" nor "no one"; the decision falls back to audio. |
| `embed_backend` | `fp32` | `fp32` / `int8` / `npu` | **Embedding backend** — How face features (SFace) are computed. fp32: the original model, most accurate (default). int8: the official opencv_zoo INT8 model, 4× smaller; slower on an x86 server but usually faster on ARM phones. npu: converted to TFLite so Android can run it on the NPU (NNAPI / NeuroPilot); a server has no NPU and runs it on the CPU, and "Resources" shows the accelerator actually used. |
| `npu_precision` | `int8` | `int8` / `fp16` | **NPU model precision** *(advanced)* — TFLite model used by the npu backend. int8: fully integer-quantized, best for NPUs. fp16: half precision, good for GPUs and the fastest CPU option without an NPU (int8 falls back to fp16 when there is no NPU). |
| `cos_match_int8` | 0.4 | 0 – 1 | **Same-person threshold (quantized)** *(advanced)* — Same-person similarity threshold for the int8 and npu int8 backends. Quantized models agreed with fp32 on about 99.9% of decisions in tests, so the default equals fp32. |

### Pose (optional)

| Key | Default | Range | Description |
|---|---|---|---|
| `pose_enabled` | off | on / off | **Enable pose detection** *(Mode / switch)* — MoveNet MultiPose Lightning (Apache-2.0, up to 6 people) detects bodies and keypoints (shoulders, hips, nose, eyes), so someone turned sideways or away still counts as present. Costs more CPU; off by default. |
| `pose_conf` | 0.4 | 0.05 – 1 | **Body detection confidence** — Body detections below this are ignored. |
| `pose_interval` | 2 | 0 – 2 | **Pose inference interval** *(advanced)* — Stable scenes reuse body results for up to this many seconds (default 2). Significant image motion, geometry/count changes, missing faces or unstable states refresh immediately. 0 disables throttling. |
| `body_min_conf` | 0.4 | 0.05 – 1 | **Body-only confidence threshold** *(advanced)* — With only a body and no face, this confidence is needed to count as "someone there" (with a face, the detection confidence applies). |
| `pose_roi` | (0, 0, 1, 1) | box (x0, y0, x1, y1) | **Body ROI (x0,y0,x1,y1)** — A body counts only if its center is inside this (purple) box; usually larger than the face ROI. |
| `roi_rule` | `either` | `either` / `face` / `body` | **Participant rule** — Face or body: face in the face ROI or body in the body ROI. Face only: face ROI only. Body only: body ROI only (needs pose detection). |
| `turned_away_early` | on | on / off | **Early warning when turning away** — When every participant has turned (side or back) and is getting smaller (walking away), enter MAYBE_AWAY early. |
| `turned_away_seconds` | 2 | 0 – 30 | **Turned-away duration** *(advanced)* — How long "turned away and receding" must last to trigger. |
| `recede_ratio` | 0.1 | 0.01 – 0.9 | **Receding ratio** *(advanced)* — Shrinking by more than this fraction (body or face width) within about 2 s counts as walking away. |

### Distance (optional)

| Key | Default | Range | Description |
|---|---|---|---|
| `distance_enabled` | off | on / off | **Enable distance estimation** *(Mode / switch)* — Estimates distance from face width (about 0.15 m) and shoulder width (about 0.38 m, frontal only) plus the camera field of view. |
| `camera_hfov` | 60 | 20 – 170 | **Camera horizontal FOV** — Horizontal field of view of the camera. If unknown, keep the default and calibrate below. |
| `distance_calib` | 1 | 0.2 – 5 | **Calibration factor** — Estimated distances are multiplied by this. "I am standing at this distance" below computes it. |
| `zone_min` | 0.4 | 0 – 10 | **Conversation zone near edge** — Closer than this is marked "too close". |
| `zone_max` | 2 | 0.1 – 20 | **Conversation zone far edge** — Outer edge of the conversation zone. Beyond it but before the leave distance is marked "far". |
| `leave_distance` | 2.5 | 0.2 – 30 | **Leave distance** — Beyond this distance a person counts as gone even while still in view. |
| `face_width_m` | 0.15 | 0.08 – 0.3 | **Average face width** *(advanced)* — Average face width used for distance estimation. |
| `shoulder_width_m` | 0.38 | 0.2 – 0.6 | **Average shoulder width** *(advanced)* — Average shoulder width used for distance estimation. |
| `distance_ema` | 0.3 | 0.01 – 1 | **Smoothing factor** *(advanced)* — Weight of each new estimate (EMA). Smaller is smoother but slower. |

### Tracking

| Key | Default | Range | Description |
|---|---|---|---|
| `tracker_mode` | `legacy` | `legacy` / `kalman` | **Tracking mode** *(Mode / switch)* — Classic: IoU + face similarity, greedy one-by-one matching. Kalman: a Kalman filter predicts each person's position and velocity; IoU, motion distance, face similarity and clothing color (torso HSV histogram, no model) are combined and matched at once with the Hungarian algorithm. During short occlusions the track coasts along its prediction (dashed box). |
| `join_seconds` | 2 | 0 – 30 | **Time to become a participant** — How long someone must stay in the ROI to become a participant (speaking joins immediately). |
| `grace_seconds` | 15 | 0 – 120 | **Grace period when lost** — A participant is removed only after being invisible this long, so a brief occlusion does not drop them. |
| `cos_match` | 0.4 | 0 – 1 | **Same-person similarity threshold** — Face embeddings with at least this cosine similarity are the same person (SFace: about 0.36). |
| `cos_weak` | 0.2 | 0 – 1 | **Weak similarity (overlapping boxes)** *(advanced)* — When the boxes roughly overlap, this lower similarity is enough to count as the same person. |
| `iou_match` | 0.3 | 0 – 1 | **Overlap threshold (IoU)** *(advanced)* — How much two boxes in consecutive frames must overlap to count as "the same position". |
| `embed_ema` | 0.15 | 0 – 1 | **Template update rate** *(advanced)* — Fraction of the face template updated each frame. Higher adapts faster to lighting changes but is easier to drift to someone else. |
| `coast_seconds` | 2 | 0 – 10 | **Coast time** — Kalman: how long a track keeps following its prediction without detections. After that it counts as invisible (the grace period still applies). |
| `coast_present_conf` | 0.4 | 0 – 1 | **Coasting counts as present above** *(advanced)* — Kalman: a predicted box still counts as "someone there" while its visibility confidence (falling from 1 to 0 over time) is above this. Raise it to decide "left" sooner. |
| `gate_mahalanobis` | 9.21 | 1 – 100 | **Motion gate** *(advanced)* — Kalman: a detection farther than this statistical distance (Mahalanobis²) from the prediction is not matched unless the face similarity is high. 9.21 ≈ 99%. |
| `appearance_min_sim` | 0.6 | 0 – 1 | **Clothing similarity threshold** *(advanced)* — Kalman: without a face, clothing-color similarity below this means a different person. |
| `appearance_ema` | 0.2 | 0 – 1 | **Clothing template update rate** *(advanced)* — Kalman: fraction of the clothing-color template replaced after each match. |

### Engagement intent

| Key | Default | Range | Description |
|---|---|---|---|
| `intent_model` | `rules` | `rules` / `score` | **Join rule** *(Mode / switch)* — Rules: anyone who stays in the ROI for the join time becomes a participant. Intent score: each person gets a 0–1 engagement intent (facing the robot, distance, dwell time, approaching, not just passing by, centered, speaking); they join when it stays above the threshold for the join time. Passers-by and bystanders do not join and do not keep a conversation alive. |
| `join_threshold` | 0.6 | 0.05 – 0.99 | **Join threshold** — Intent-score mode: intent must stay above this for the join time to become a participant (and to start a new conversation). |
| `drop_threshold` | 0.35 | 0 – 0.95 | **Drop-to-bystander threshold** — Intent-score mode: a participant whose intent stays below this for the drop time becomes a bystander (the last participant is never dropped; presence decides whether they left). |
| `drop_seconds` | 8 | 0 – 120 | **Drop time** *(advanced)* — How long intent must stay below the threshold before a participant becomes a bystander. |
| `intent_tau` | 1 | 0.1 – 20 | **Smoothing time** *(advanced)* — Time constant smoothing the intent score: larger is steadier but slower. |
| `intent_weighted_presence` | off | on / off | **Weight presence by intent** — When on (probabilistic mode), a participant's presence probability is multiplied by their intent, so low-intent participants keep the conversation alive less. |
| `w_facing` | 0.45 | 0 – 1 | **Weight: facing** *(advanced)* — Is the face frontal (YuNet five landmarks) plus shoulder direction (pose). Facing also scales the whole score: someone turned away rarely scores high. |
| `w_distance` | 0.1 | 0 – 1 | **Weight: distance** *(advanced)* — 1 inside conversation distance, falling to 0 at the leave distance. Ignored while distance estimation is off. |
| `w_dwell` | 0.1 | 0 – 1 | **Weight: dwell** *(advanced)* — Grows the longer someone stays in the ROI (full after 4 s). |
| `w_approach` | 0.05 | 0 – 1 | **Weight: approaching** *(advanced)* — Growing (approaching) adds, shrinking (walking away) subtracts. |
| `w_still` | 0.15 | 0 – 1 | **Weight: not passing by** *(advanced)* — The faster the sideways motion the lower the score; filters out people walking past. |
| `w_center` | 0.15 | 0 – 1 | **Weight: position** *(advanced)* — Higher the closer to the center of the ROI. |

### Presence

| Key | Default | Range | Description |
|---|---|---|---|
| `presence_model` | `rules` | `rules` / `hmm` | **Presence model** *(Mode / switch)* — Rules: consecutive-frame counters (someone / no one) plus timeouts. Probabilistic: each person has a presence probability that rises when seen and falls when not; the group probability and thresholds decide the state, which is robust to detection flicker and short occlusions. Works best with the Kalman tracker. |
| `priority_mode` | `vision` | `vision` / `balanced` / `audio` | **Decision priority** — Vision first: while someone is in view, silence never ends the conversation; when the ROI is confirmed empty, the empty-ROI timeout applies regardless of sound. Audio is only used when vision is unsure, and audio alone never brings the state back from AWAY. Good for noisy rooms. Balanced: audio and vision decide together. Audio first: mostly silence-based; vision only adjusts the timeouts. |
| `speech_join` | `auto` | `auto` / `off` / `single_candidate` / `all` | **Join immediately on speech** *(advanced)* — When someone speaks, whether a person in the ROI becomes a participant without waiting for the join time (audio has no position). Auto: off in vision-first mode, otherwise "single candidate". Single candidate: joins immediately only if exactly one candidate is visible, the face is inside the face ROI and facing the camera (and inside the conversation zone when distance is on). All: every visible candidate joins immediately. |
| `present_frames` | 2 | 1 – 20 | **Frames to confirm presence** — Consecutive frames with someone needed to count as "someone there". |
| `absent_frames` | 3 | 1 – 20 | **Frames to confirm absence** — Consecutive frames without anyone needed to count as "ROI empty"; prevents flicker. |
| `leave_halflife` | 1.5 | 0.1 – 30 | **Leave half-life** — Probabilistic mode: when nobody is seen at all, the odds of "present" halve every this many seconds. Smaller decides "left" faster. The default 1.5 s tolerates short detection gaps. |
| `detect_gain` | 4 | 0.1 – 20 | **Detection evidence strength** *(advanced)* — Probabilistic mode: how fast the probability rises per second while someone is detected (times the detection confidence). Larger returns to ACTIVE faster. |
| `lowconf_factor` | 0.15 | 0 – 1 | **Low-confidence decay ratio** *(advanced)* — Probabilistic mode: with only low-confidence detections, the probability falls this many times as fast as when nothing is seen. Smaller avoids deciding "left" in poor lighting. |
| `p_active` | 0.8 | 0.05 – 0.99 | **ACTIVE threshold** *(advanced)* — Probabilistic mode: the group probability must exceed this to return from MAYBE_AWAY / AWAY to ACTIVE. |
| `p_maybe` | 0.6 | 0.02 – 0.98 | **MAYBE_AWAY threshold** — Probabilistic mode: below this group probability the state becomes MAYBE_AWAY. |
| `p_away` | 0.3 | 0.01 – 0.95 | **AWAY threshold** — Probabilistic mode: below this group probability for the AWAY hold time the state becomes AWAY. |
| `away_hold_seconds` | 1 | 0 – 30 | **AWAY hold time** *(advanced)* — Probabilistic mode: how long the probability must stay below the AWAY threshold. |

### State machine / timeouts

| Key | Default | Range | Description |
|---|---|---|---|
| `idle_after` | 3 | 0 – 60 | **Silence until IDLE** — Silence after which ACTIVE becomes IDLE. |
| `present_timeout` | 15 | 1 – 600 | **Timeout while present** — Balanced mode: silence after which the conversation is considered over although a participant is in view. Default 15 s, shorter than the 20 s used when vision is unsure. |
| `empty_timeout` | 5 | 1 – 120 | **Timeout when ROI is empty** — How quickly the conversation ends when nobody is there and it is silent (shortened). |
| `base_timeout` | 20 | 1 – 300 | **Timeout when vision is unsure** *(advanced)* — Audio-only timeout used when vision confidence is too low. |
| `present_max_silence` | 0 | 0 – 3600 | **Max silence while present (vision first)** — In vision-first mode: how long a person may stay in view without speaking before the conversation is considered over. 0 = no limit. |
| `maybe_away_frac` | 0.5 | 0.05 – 0.95 | **MAYBE_AWAY fraction** *(advanced)* — Enter MAYBE_AWAY when the running timeout reaches this fraction. |
| `end_after_away` | 5 | 0 – 120 | **AWAY until ENDED** — How long AWAY lasts before the conversation ends and embeddings are cleared. |
| `new_group_seconds` | 3 | 0 – 60 | **New-group confirmation** — How long a new group must be present, after all previous participants are gone, before NEW_GROUP fires. |

### Audio (PureSound + vision cross-check)

| Key | Default | Range | Description |
|---|---|---|---|
| `audio_pipeline` | `energy_vad` | `energy_vad` / `puresound` | **Audio pipeline** *(Mode / switch)* — Browser VAD: a level-based VAD in the browser decides speech. PureSound: microphone audio (16 kHz PCM) streams to the server, PureSound streaming voice isolation (dpcrn_curriculum_v2) removes background sound, and speech is detected on the clean track. Needs the puresound package and model on the server; otherwise the panel shows why and the browser VAD keeps working. |
| `onset_entry_only` | on | on / off | **Start conversations only on "entry"** — When on (PureSound mode), a conversation starts only when a speech onset on the clean track coincides with a nearby person in view; someone just walking by, or sound alone (a TV, a distant person), never starts one. |
| `onset_needs_vision` | on | on / off | **Speech needs a person in view** — PureSound mode: for speech to start a conversation or return from AWAY, a nearby person must have been in view within the last 1.5 s (when vision is unsure, audio alone decides). |
| `iso_onset_guard` | off | on / off | **Onset guard** — Built into PureSound: keeps the original audio for the first second of each utterance and switches to the isolated voice once speech continues, so the first word is not swallowed. |
| `iso_reset_on_change` | on | on / off | **Reset on speaker change** *(advanced)* — Reset the voice-isolation stream state (and restart the onset guard) on NEW_GROUP, a new conversation or a participant change. |
| `clean_vad_margin_db` | 10 | 3 – 30 | **Clean VAD margin** — How many dB above the noise floor the isolated voice must be to count as speech. Smaller is more sensitive. |
| `clean_vad_min_db` | -50 | -80 – -10 | **Clean VAD minimum level** *(advanced)* — Minimum level (dBFS) of the isolated voice to count as speech; distant or very quiet speech is ignored. |
| `clean_vad_hang` | 0.3 | 0 – 2 | **Clean VAD hangover** *(advanced)* — How long "speaking" holds after the isolated voice drops. |
| `xcheck_enabled` | on | on / off | **Vision cross-check** — PureSound filters continuously (pickup range about 1 m) and every 20 ms the result is checked against the video: clean speech but nobody in view = leak (attenuated, not speech); clean speech but everyone outside the near range = acoustic out of range (attenuated, not speech); silent clean track while someone is in range and the raw mic hears speech = acoustic miss (more raw audio mixed in, counts as speech). When vision is unsure, audio alone decides. |
| `near_range` | 1 | 0.2 – 10 | **Near range** — PureSound pickup range. With distance estimation on, everyone farther than this is "out of range"; without it only "is someone in the ROI" is checked. |
| `xcheck_hold` | 0.75 | 0 – 5 | **Vision hold time** — Video arrives at 0.5–2 frames per second, audio every 20 ms; a person stays "seen" this long to align both and keep the start of sentences. |
| `xcheck_stale` | 2 | 0.2 – 30 | **Vision stale time** *(advanced)* — With no new vision result for this long, audio alone decides. |
| `xcheck_require_calib` | off | on / off | **Require distance calibration** — When on, "out of range" is only used after distance calibration. |
| `xcheck_floor_db` | -40 | -80 – 0 | **Attenuation** — How much the output is lowered on leak or out of range (≤ -60 dB = mute). |
| `xcheck_recover_mix` | 0.3 | 0 – 0.9 | **Enhancement share on a miss** *(advanced)* — On an acoustic miss, the share of isolated voice in the output; the rest is time-aligned raw audio. Smaller keeps more of the original. |
| `xcheck_attack` | 0.01 | 0.001 – 0.5 | **Gain attack** *(advanced)* — Fade-in time constant when attenuation ends. |
| `xcheck_release` | 0.05 | 0.005 – 2 | **Gain release** *(advanced)* — Fade-out time constant when attenuation starts. |

### Speech recognition (cloud, optional)

| Key | Default | Range | Description |
|---|---|---|---|
| `asr_provider` | `none` | `none` / `assemblyai` | **Speech recognition service** *(Mode / switch)* — None: no speech recognition (default). AssemblyAI: audio streams live to the AssemblyAI cloud (Universal-Streaming v3) for transcription. ⚠ An external cloud service: audio leaves this device; make sure users agree and local rules allow it. Needs the PureSound pipeline and an API key. |
| `asr_source` | `clean` | `clean` / `raw` | **Audio sent** — Clean: the clean track after PureSound and the vision check (default). Raw: the raw microphone. |
| `asr_feed` | `speech` | `speech` / `session` | **When to send** — Speech only: audio is sent only while it counts as speech, silence otherwise (keeps timing; the content stays on the device). Whole conversation: all audio during the conversation. The stream always closes at ENDED. |
| `asr_region` | `global` | `global` / `us` / `eu` | **Data region** — AssemblyAI endpoint: global (nearest region), US or EU (data stays in the region). |
| `asr_speech_model` | `universal-3-6-pro` | `universal-3-6-pro` / `universal-3-5-pro` / `universal-streaming-english` / `universal-streaming-multilingual` | **Model** — Universal-3.6 Pro: 32 languages (including Chinese zh and Cantonese yue), mixed languages allowed (default). Universal-3.5 Pro: 19 languages (zh, no yue). Universal-Streaming English: English only. Universal-Streaming Multilingual: English, Spanish, German, French, Portuguese, Italian; no Chinese. |
| `asr_language_codes` | `zh,en` | text ≤ 200 | **Languages** — Comma-separated language codes (e.g. zh,en) to favor; mixing still works. Empty = no restriction. Universal-3.x Pro only. |
| `asr_keyterms` | — | text ≤ 2000 | **Key terms** *(advanced)* — Comma-separated, up to 100, to improve recognition of these words (names, products). |
| `asr_format_turns` | on | on / off | **Format results** *(advanced)* — Add punctuation and casing. Universal-Streaming models only (3.x Pro always formats). |
| `asr_eot_confidence` | 0.4 | 0 – 1 | **End-of-turn confidence** *(advanced)* — Confidence needed to end a turn. Universal-Streaming models only. |
| `asr_min_turn_silence` | 400 | 50 – 10000 | **Min end-of-turn silence** *(advanced)* — Silence after which the end of a turn is checked. |
| `asr_max_turn_silence` | 1280 | 100 – 10000 | **Max end-of-turn silence** *(advanced)* — Silence after which a turn always ends. |

### Cross-session re-ID (cloud, optional)

| Key | Default | Range | Description |
|---|---|---|---|
| `reid_enabled` | off | on / off | **Cross-session re-ID** *(Mode / switch)* — A new session starts immediately. Joined participants are compared with recent conversation members; only unique, unambiguous overlap resumes a conversation through handoff. Uncertain results stay unlinked. External cloud service: face crops leave this device. |
| `reid_consent` | off | on / off | **Users are informed and agreed** — Required: up to three participant face crops per conversation are retained only in server memory and sent to a third-party cloud for comparison. Never written to disk or logs; optional thumbnails reach this browser. |
| `reid_show_crops` | on | on / off | **Show comparison photos** — Show this session's face and the matched reference face (small thumbnails, about 112 px) in the "User comparison" card. Thumbnails go only to this browser over this connection and are never written to disk, logs, debug events or recordings; "Clear reference faces" or reset removes them. |
| `reid_provider` | `jev` | `jev` | **Cloud service** *(advanced)* — Cloud backend used for comparison (pluggable). Currently Jev (Typesafe). |
| `reid_model` | `jev-latest` | `jev-latest` / `jev-1.13.0` | **Model** — jev-latest follows service updates; pin jev-1.13.0 for reproducible results. |
| `reid_threshold` | 0.9 | 0.5 – 0.99 | **Same-person threshold** — A score at least this high means the same person and resumes the conversation. Mistaking someone is costly, so the default is 0.9 (LFW evaluation: FAR 0%, FRR 14%). |
| `reid_uncertain_low` | 0.4 | 0 – 0.95 | **Uncertain lower bound** *(advanced)* — Between this and the threshold, one more fresh frame is compared; still below the threshold means a new conversation. |
| `reid_ttl` | 600 | 30 – 600 | **Reference retention** — Retention of ended conversation members in memory, measured from the last normal session END. Reconnects do not renew it; expiry deletes the references. |
| `reid_max_refs` | 3 | 1 – 5 | **Recent conversations** — Maximum ended conversations retained, with up to three member references each. The active conversation is separate. All candidates are checked for ambiguity. |
| `reid_min_face` | 0.06 | 0.02 – 0.5 | **Minimum face width** *(advanced)* — A face must be at least this fraction of the frame width to be compared. |
| `reid_face_wait` | 3 | 0.5 – 15 | **Wait for a clear face** *(advanced)* — After the first participant appears, wait at most this long for a clear face; otherwise start a new conversation. |
| `reid_decision_timeout` | 10 | 0.5 – 30 | **Maximum wait** *(advanced)* — Wall-clock deadline from session start, including crop collection and remote calls. Unresolved decisions stay unlinked; AWAY does not pause it. |
| `reid_timeout` | 2 | 0.3 – 10 | **Call timeout** *(advanced)* — Give up on a cloud call after this long and start a new conversation. |
| `reid_cooldown` | 5 | 0 – 120 | **Cooldown** *(advanced)* — Minimum time between decisions; a session that reopens quickly starts a new conversation without a call. |
| `reid_cap_per_hour` | 30 | 1 – 600 | **Hourly cap** — Maximum calls per hour; after that, always a new conversation. |
| `reid_cap_total` | 200 | 1 – 10000 | **Per-connection cap** *(advanced)* — Maximum calls on this connection. |
| `reid_cost_per_call` | 0 | 0 – 10 | **Price per call** — Your own price per call, only for the cost estimate (0 = show tokens only). |

### Performance

| Key | Default | Range | Description |
|---|---|---|---|
| `motion_gate` | on | on / off | **Motion gate** — When the image barely changes and the state is stable, skip face detection and reuse the last result. |
| `motion_threshold` | 0.015 | 0 – 0.2 | **Change threshold** *(advanced)* — Mean difference of a small grayscale image (0–1) below which the frame counts as unchanged. Raise for noisy cameras. |
| `force_detect_seconds` | 2 | 0.5 – 30 | **Forced detection interval** *(advanced)* — Run a full detection at least this often even when nothing changes. |
| `embed_throttle` | on | on / off | **Embedding throttling** — Reuse the last embedding while a face does not move; recompute for new faces, movement or scale beyond the threshold, low similarity, or after the interval. |
| `embed_interval` | 3 | 0 – 30 | **Embedding refresh interval** *(advanced)* — Recompute each face's embedding at least this often. |
| `embed_move_thresh` | 0.15 | 0 – 1 | **Movement threshold** *(advanced)* — Recompute when the center moves or the size changes by more than this fraction of the face box. |
| `adaptive_fps` | on | on / off | **Adaptive fps** — The server suggests the frame rate from the state: low when stable, high on changes or MAYBE_AWAY. Replaces "Frames per second" below when on. |
| `fps_low` | 0.5 | 0.2 – 5 | **Stable fps** — Suggested fps while stable (ACTIVE/IDLE/ENDED, no new faces). |
| `fps_high` | 2 | 0.2 – 5 | **Changing fps** — Suggested fps on new faces, lost participants or MAYBE_AWAY. |
| `roi_crop` | on | on / off | **ROI crop + downscale** — Detect only on the ROI (plus a margin), cropped and downscaled; boxes are mapped back to full-frame coordinates. |
| `crop_margin` | 0.1 | 0 – 0.5 | **Crop margin** *(advanced)* — Extra margin kept around the ROI (fraction of the frame). |
| `max_width` | 640 | 96 – 1920 | **Max detection width** — Maximum width of the image fed to the detector. Smaller is faster, but small far-away faces may be missed. |

### Display (this browser only)

| Key | Default | Range | Description |
|---|---|---|---|
| `mirror_view` | on | — | **Mirror view** *(browser only)* — Flip the preview and boxes left-right like a mirror (selfie view), which feels more natural. Display only: the frames sent to the server are not flipped and match what the robot camera sees; ROI values are always stored in raw camera coordinates (when mirrored, the left side of the screen is the high-x side). |
| `show_face_roi` | on | — | **Show face ROI** *(browser only)* — Draw the cyan face ROI box on the video (display only). |
| `show_pose_roi` | on | — | **Show body ROI** *(browser only)* — Draw the purple body ROI box on the video (display only). |

<!-- END GENERATED -->
