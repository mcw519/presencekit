# Security policy

## Supported versions

Only the latest release on the default branch receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use the repository's private vulnerability
reporting (GitHub: *Security → Report a vulnerability*). Include what you found, how to reproduce it and the
impact you expect. You should get a first reply within a week. Please give us a reasonable time to fix the
problem before you disclose it.

請不要用公開 issue 回報安全問題，請使用 repository 的私密漏洞回報功能（GitHub：*Security → Report a vulnerability*）。

## Scope and deployment notes

PresenceKit is a multimodal presence SDK with a web demo. When you run it:

- Set `PRESENCE_TOKEN` whenever anyone else can reach the server, and serve it over HTTPS. Without a token every
  endpoint and the WebSocket are open.
- API keys (`ASSEMBLYAI_API_KEY`, `TYPESAFE_API_KEY`) are kept in process memory and shared by all connections.
- `PRESENCE_NPU_DELEGATE` loads a native library; it can only be set from the server environment.
- The optional cloud features (speech recognition, re-ID) send audio or face crops to third parties; see
  [docs/privacy.md](docs/privacy.md).

## Dependency and TLS trust baseline

Use uv **0.13.4 or newer** and synchronize with `uv sync --locked --extra puresound`. The project enforces
the uv minimum and constrains certifi to **2026.10.19 or newer** because of the compromised emSign CA
incident ([uv advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62),
[certifi advisory](https://github.com/advisories/GHSA-6q48-6c7w-9gw2)). Updating certifi alone does not
update Python/OpenSSL's separate default CA bundle; maintain that bundle through its package manager
and restart long-running clients after trust-store updates. Keep TLS verification enabled.

LiteRT 2.3.0 declares `backports.strenum` unconditionally. On Python 3.11 and newer, the project constrains
this transitive dependency to the official **1.2.8** release, whose `Requires-Python` range is
`>=3.8.6,<4.0.0`; 1.3.1 excludes Python 3.11 and newer. This keeps the original wheel metadata intact
and satisfies LiteRT without ignoring compatibility warnings. Revisit this constraint when upstream
corrects its dependency marker. Verify both `uv pip check` and installed `Requires-Python` ranges;
an audit without known vulnerabilities does not establish runtime compatibility.

Issues we consider security bugs include: a way to read camera frames, audio, face crops, keys or the token
without authorisation; anything that writes images or audio to the server's disk; secrets appearing in logs or
responses; and ways to run code through the WebSocket or HTTP API.
