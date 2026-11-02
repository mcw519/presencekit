# Vendored PureSound web design system

Pinned copy of the PureSound playground's CSS design system, so presence-demo and the
PureSound playground share one set of tokens instead of divergent copies.

- Source: https://github.com/mcw519/puresound, `puresound/web/static/`
- Revision: `121bf623b491cb7f18805bec65099543ecc5323f`. These web assets are pinned
  independently of the `puresound-streaming` SDK in `pyproject.toml`.
- License: Apache-2.0 (`LICENSE`, `NOTICE`); IBM Plex fonts: SIL OFL 1.1 (`fonts/LICENSE.txt`)

| File | Upstream | Changes |
| --- | --- | --- |
| `tokens.css` | `css/tokens.css` | none |
| `components.css` | `css/components.css` | none |
| `base.css` | `css/base.css` | font URLs made relative (`/fonts/` → `fonts/`) |
| `fonts/*` | `fonts/*` | none |
| `shell.css` | `css/shell.css` | none |
| `screens-shared.css` | subset of `css/screens.css` (panel title + metric) | extracted |

The PureSound Python package (`puresound-streaming`) does not ship these web assets, so they
cannot be loaded from the installed dependency; `tests/test_design_tokens.py` checks the copy
against the checksums below. To update: copy the files from the new revision, re-apply the
`base.css` change, update the revision here and the checksums in the test.
