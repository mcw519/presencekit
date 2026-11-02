**English** · [繁體中文](zh-TW/privacy.md)

# Privacy

PresenceKit watches people with a camera and a microphone, so it is designed to keep as little as possible for
as short as possible. This page collects every rule in one place. The rules are enforced by tests where possible.

## Defaults: nothing leaves, nothing is written

| Data | What happens | Kept |
|---|---|---|
| Camera frames | Decoded in memory, analysed, dropped. Never written to disk, never sent back. | Only while being processed |
| Face embeddings (128 numbers) | Kept in the tracker's memory to recognise people within a session. | Until the session ENDS (then erased) |
| Microphone audio | Browser VAD by default (audio stays in the browser). With PureSound, chunks are processed in server memory. | Seconds, in memory |
| Transcripts (optional ASR) | Forwarded to the browser, last 100 lines in page memory. | Not on the server |
| Logs | Structured events with counts, scores and timings. Never pixels, audio, text or keys. | Your log file |

`tests/test_privacy.py`, `tests/test_audio.py` and `tests/test_recording.py` run full sessions and check that no
file is written.

## Things you can turn on

Each of these is **off by default** and must be enabled by the operator.

| Feature | What changes | Read more |
|---|---|---|
| Cloud speech recognition | Audio is streamed to AssemblyAI during sessions (only speech chunks by default). | [ASR](asr.md) |
| Cross-session re-ID | Needs `reid_consent` too. One face crop per user kept **in server memory** for at most 10 minutes; two crops per comparison are sent to the re-ID service. Optional 112 px thumbnails go to *this browser only*. | [Re-ID](reid.md) |
| Recording | Started by the user. Audio tracks are collected in the **browser** and downloaded to the user's own computer. The server writes nothing; video is never recorded. | [Audio](audio.md#listening-and-recording) |

If you deploy any of them, tell the people in front of the device, get consent where required (faces are biometric
data in many jurisdictions, for example GDPR art. 9), and check the providers' retention policies.

## Secrets

- The access token comes only from the `PRESENCE_TOKEN` environment variable.
- API keys (`ASSEMBLYAI_API_KEY`, `TYPESAFE_API_KEY`) come from the environment or the Settings password fields;
  they are kept in process memory, never written, logged or returned to the browser.
- Logs mask fields named `key`, `token`, `secret`, `password` or `authorization`, and `token=` / `key=` in URLs.
- `PRESENCE_NPU_DELEGATE` (a native library) can only be set from the environment, never over the network.

## Network exposure

Without `PRESENCE_TOKEN` anyone who can reach the port can use the demo. Set a token and use HTTPS when it is
reachable by others. Report security problems as described in [SECURITY.md](../SECURITY.md).
