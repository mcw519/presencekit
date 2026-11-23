**English** · [繁體中文](zh-TW/reid.md)

# Cross-session re-identification

Re-ID can associate a newly started session with one recent conversation. It is optional and requires both `reid_enabled` and explicit `reid_consent`. The current PoC profile targets one local M4 + Chrome connection and up to three participants. The Jev service compares face crops; these biometric images leave the device when a real backend call occurs.

## Session and handoff behavior

Every new tracker session immediately receives a provisional, random `conversation_id` and continues normal presence/audio processing. Re-ID runs in the background. A successful, unambiguous match emits a `handoff` with the previous provisional conversation ID and the selected historical conversation ID. Session IDs and their event history are preserved. A failed, uncertain, ambiguous, expired, or unavailable comparison leaves the provisional conversation unlinked and never blocks the session.

The driver stores the best available face crop for up to three joined participant tracks in a session. On a normal `ENDED`, those participant references are committed to the ended conversation. On a later session, each usable current participant face is checked against every eligible historical reference. A conversation is eligible for handoff only when matches identify exactly one historical conversation. If qualifying matches point to multiple conversations, the result is ambiguous and no handoff is made. A clear match to one participant can continue a group conversation; no cross-session face match is a claim that every current participant is known.

The comparison decision has a **10-second wall-clock deadline** from the new session start. Presence states such as `AWAY` do not pause it. The first clear participant face may be awaited for up to `reid_face_wait`; uncertain scores can request one fresh-frame retry, subject to the same overall deadline. Timeouts, missing faces, uncertain scores, request caps, missing credentials, circuit-breaker state, and service errors leave the new conversation unlinked. Jev quality and latency still require separate tests with consented subjects; mock results validate only application control flow.

Set `reid_model=jev-1.14.0` to pin that version and enable `reid_batch` to try one HTTP request per current face with all eligible reference faces. The default is one request per pair. Both modes require a complete set of valid scores before deciding. Each request consumes one call from the budget, and input tokens are recorded once per response. A failed batch leaves the provisional conversation unlinked.

The server sends `handoff` with `session`, `conversation_id`, `previous_conversation_id`, `decision`, `resumed_from_session`, `score`, `reason`, and `calls`. The `conversation_id` remains an opaque random identifier; this application does not store dialogue content.

## Page continuity and expiry

The browser creates a cryptographically random 128-bit page capability in JavaScript memory. It offers it as the WebSocket subprotocol `page-<32 lowercase hex>` alongside `presence-v1`, and reuses it for reconnects in the same loaded page. The capability is not placed in the URL or browser persistent storage. Diagnostic redaction masks the capability if it appears in a WebSocket `Sec-WebSocket-Protocol` debug header. On `pagehide`, the browser sends the capability in the JSON body of a best-effort `sendBeacon` clear request. A page reload creates a new capability and cannot recover the previous page's references.

The server keeps page state and face crops in process memory only. A normal `ENDED` starts the historical conversation's **600-second TTL**. Reconnects and unsuccessful comparisons do not refresh that history TTL; if a resumed conversation is interrupted again, its original normal-`ENDED` expiry remains in force. A conversation that has never reached normal `ENDED` receives one 600-second active-reference lease at its first interruption; repeated reconnects and interruptions do not renew it. `pagehide` sends a best-effort clear request. An app-owned cleanup task prunes page and reference leases every second, including after the last WebSocket has disconnected, and shutdown wipes all page stores. Consent withdrawal and explicit reset also clear references.

Each acquired WebSocket gets a page generation. Replacing a transport invalidates older work before handing over the shared page store; old-generation workers and callbacks cannot mutate the new connection's state. Re-ID tasks also carry store-generation and session guards, and a candidate that expires before the final handoff is rejected. A transport interruption is not a normal `ENDED`: it preserves eligible active references for same-page reconnect under the bounded lease.

When camera input stops, presence observation time pauses at **three seconds after the last completed frame**, rather than at the time a delayed ticker happens to wake. This prevents a host sleep/resume gap from counting as elapsed-away time. The clock resumes when a frame is processed and remains monotonic.

At most three inactive historical conversations are retained in the PoC profile; each can contain up to three participant crops. When history exceeds capacity, the least recently ended inactive conversation is evicted. The active conversation is not evicted to satisfy the history cap. The server's page registry has a separate bounded page capacity; expired detached pages are eligible for cleanup.

## Privacy and service calls

| Data | Handling |
|---|---|
| Reference/probe | Face-only 320 × 320 JPEG crops in server memory; whole frames are not retained for comparison. |
| Retention | Up to 600 seconds from normal `ENDED`; page detach also has a 600-second hard lease. No disk persistence. |
| Provider | With configuration and consent, Typesafe Jev receives the current face crop and one reference crop per request, or all eligible reference crops in batch mode. Synthetic acceptance makes no Jev calls. |
| Consent | Both the feature switch and consent flag must be true. Turning it off wipes stored references. |
| API key | `TYPESAFE_API_KEY` or the Settings key field; held in memory and not logged or echoed. |
| Browser thumbnails | Optional 112 px comparison previews are sent only to this page when `reid_show_crops` is enabled; they are not diagnostic event data. |

Before enabling the cloud backend, inform participants, obtain the required consent, and review the provider's image handling and retention terms. Do not use real face data in synthetic acceptance runs.

## Acceptance status

When Chrome and offline model prerequisites are available, the `browser-reid` suite covers nine named cases: `consent_required`, `provisional_conversation`, `same_page_handoff`, `nonblocking_comparison`, `consent_revocation`, `late_result_discarded`, `nonblocking_media`, `cleanup`, and `server_errors`. The provisional ID is checked against the original conversation; a successful handoff is checked against the exact original session and conversation. After consent withdrawal, the cancellation-resistant mock must increment `completed_after_cancel` and return its `.99` result after cancellation, and the suite checks that handoff/resume counts do not increase and the UI remains cleared. The production `state` message carries `session`; the UI drops old handoff status and thumbnails on a new session or WebSocket reconnect, consent withdrawal, feature disablement, and reference wipe, and ignores re-ID messages from older sessions.

The `functional` adapter uses opaque mock identity values and virtual time to exercise re-ID decision rules and continuity contracts. The browser adapter adds synthetic Chrome media, production configuration and crop callbacks, `CrossSessionReid`, and explicit handoff; its mock `CloudReid` is injected only into the isolated child server. Its consent setting is test scaffolding and does not represent consent by real participants. Crops and page capabilities remain in memory; reports do not retain identifiers or image data. No Jev request is made, so these checks cannot establish biometric accuracy or provider behavior. Consult the generated report for case statuses. See the full [PoC acceptance plan](acceptance.md) and [architecture](architecture.md).
