**English** · [繁體中文](zh-TW/tuning.md)

# Tuning

Defaults are a starting point. Tune on the real device, camera position, lighting and crowd. A good loop: change
one thing, watch the UI for a few minutes, **export the settings JSON** with a descriptive name
(`lobby-noisy.json`), repeat.

## 0. Start from a mode

Choose *Standard*, *Near-field priority* or *Low power* in Settings. Open **Detailed settings** to tune
individual parameters and export your configuration. See [the three modes](configuration.md#modes).

## 1. Fix the ROI first

Have someone stand where users normally stand. Drag the face ROI (cyan) to cover every spot from which people talk
to the device, and keep corridors and background out. Keeping passers-by out of the ROI is the single most
effective way to reduce errors. With pose enabled, adjust the body ROI (purple) too.

## 2. Detection confidence `min_conf`

Watch the `conf` of each track in the people list. A user facing the camera should stay clearly above
`min_conf`. If backlight or side views often fall below it, the system keeps falling back to audio: lower it a
little, or improve lighting and camera angle.

## 3. Identity `cos_match` / `cos_weak`

1. Let person A stand for 30 s, turning and moving; note the similarity values in the people list. That is the
   *same person* range.
2. Let B and C step in; note their values. That is the *different people* range.
3. Put `cos_match` between them, closer to the low end of *same person* (better an extra track than two people
   merged).
4. `cos_weak` handles the "same place, different angle" case; about half of `cos_match` usually works.

## 4. Flicker and departures

- **Kalman + HMM** (recommended): `leave_halflife` is the main knob. Lower = faster "left", higher = more tolerant
  of occlusions. Watch the presence probability chart.
- **Rules**: missed frames → raise `absent_frames`; slow departures → lower `absent_frames` or `empty_timeout`, or
  raise the fps.
- People hiding each other → raise `grace_seconds` (and `coast_seconds` for Kalman).

## 5. Who joins

With `intent_model = score`, hover the intent bars to see every cue. Bystanders joining → raise `w_facing` or
`join_threshold`. Users joining too slowly → lower `join_threshold` or `intent_tau`. See [Intent](intent.md).

## 6. Priority and timeouts

- Noisy (TV, background voices): `vision`. Audio is only used when vision is uncertain.
- Quiet, camera so-so: `balanced`.
- No camera or often blocked: `audio`.
- In `vision` mode, if "standing there but ignoring the device" should end the session, set `present_max_silence`
  (for example 120 s).

## 7. Audio

- **Browser VAD**: the grey line on the level meter is the noise floor, the yellow line the speech threshold.
  Silence should sit below yellow, speech clearly above. Noisy room → raise `vad_k` or `vad_min`; words cut in the
  middle → raise `vad_hang`.
- **PureSound**: set `near_range` to the distance at which users really talk (enable and calibrate distance
  first). Watch the decision counters in the audio panel: many `leak` with nobody there is expected (the
  cross-check is doing its job); many `false_reject` means the isolation is too aggressive for your microphone.

## 8. Performance

Look at the per-stage p95 latencies in the resource panel. If the total p95 approaches `1000 / fps` ms, or the
dropped counter keeps rising, lower the fps or enable the power savers. See [Performance](performance.md).
