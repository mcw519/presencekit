---
version: alpha
name: presencekit-console
description: "Presence-demo's console is a PureSound playground screen: same shell, same components, same tokens. This file records how presence maps its content onto those components; it does not invent a second design system."
source:
  repository: https://github.com/mcw519/puresound
  path: puresound/web/static
  revision: 121bf623b491cb7f18805bec65099543ecc5323f
  vendored: presencekit/static/vendor/puresound
  license: Apache-2.0 (CSS), SIL OFL 1.1 (IBM Plex)
---

**English** · [繁體中文](zh-TW/ui.md)


# Web UI

The demo page is a single screen built from the PureSound design system. This page covers the layout, the
language switch, and the design rules for changing the UI. The YAML header above follows the
[DESIGN.md](https://github.com/google-labs-code/design.md) format.

## Layout

- **Sidebar**: connection status, language switch (EN / 中文), theme (auto / light / dark), navigation.
- **Main column**: the camera view with overlays (face and body boxes, skeletons, ROIs, captions), the audio
  lanes (waveform / spectrum), the ROI editor, and monitoring / recording.
- **Right column**: the state machine, speech recognition, the presence-probability chart, countdown bars, the
  people list with intent bars, the re-ID comparison card, resources and the event log.
- **Settings drawer**: presets, then one block per pipeline module (input → detection → pose → distance →
  tracking → intent → presence → state → audio → ASR → re-ID → performance → display). Advanced parameters are
  hidden until *Advanced* is on; parameters that do not apply are greyed out with the reason.

## Languages (i18n)

All visible text comes from string tables, one file per language:

- `static/i18n/en.js` and `static/i18n/zh-TW.js` define `I18N["en"]` / `I18N["zh-TW"]` with the same keys.
- Markup uses `data-i18n="key"` (text) and `data-i18n-title`, `data-i18n-aria-label`, `data-i18n-placeholder`
  (attributes). Scripts call `t("key", {name: value})`; `{name}` placeholders are substituted.
- Settings use `s.<key>` (label), `s.<key>.h` (help) and `s.<key>.o.<value>` (option names); modules use
  `mod.<id>`. The parameter reference in [Configuration](configuration.md) is generated from the same tables.
- Language choice: `?lang=` in the URL, else the saved choice (`localStorage["presence-lang"]`), else the browser
  language (`zh*` → `zh-TW`), else English. Switching reloads the page.
- `tests/test_i18n.py` checks that both tables have identical keys and placeholders, that every key used in the
  code exists, that every setting and module is translated, and that no Chinese text is hard-coded in the scripts.

To add a language, copy `en.js` to `<lang>.js`, translate the values, add the code to `LANGS` in `js/i18n.js`, add
a button to `#langSw` in `index.html`, and extend the test.


## Principle

**Use PureSound's design, not just its fonts.** Since v0.10.3 the HTML prefers PureSound component
classes from the vendored stylesheets. `app.css` only adds presence-specific layout (camera view,
state-machine strip, countdown bars, scopes, people list, settings module accordion) and maps a few
presence roles onto upstream tokens. It must not redefine `.button`, `.panel`, `.field`, `.control`,
`.segmented`, `.chip`, `.toast`, `.inspector`, or any `--page` / `--ink` / `--accent` token.

## Loaded stylesheets

| Href | Upstream | Role |
| --- | --- | --- |
| `/static/vendor/puresound/tokens.css` | `css/tokens.css` | colours, type, density |
| `/static/vendor/puresound/base.css` | `css/base.css` | IBM Plex, reset, focus, reduced motion |
| `/static/vendor/puresound/components.css` | `css/components.css` | button, field, control, segmented, chip, state, toggle, range, table, stat, panel, empty-state, toast, dialog |
| `/static/vendor/puresound/shell.css` | `css/shell.css` | app-shell, sidebar, page-header, workbench, inspector |
| `/static/vendor/puresound/screens-shared.css` | subset of `css/screens.css` | panel-title h2, metric-row / metric |
| `/static/app.css` | presence only | camera, state strip, scopes, people, module accordion |

Checksums of the unmodified upstream files are pinned in `tests/test_design_tokens.py`.

## Component mapping (presence → PureSound)

| Presence UI | PureSound class | Notes |
| --- | --- | --- |
| Shell | `.app-shell` `.sidebar` `.main-content` `.screen` | Same as playground |
| Nav items | `.nav-link` `.nav-icon` `.nav-group-title` | Buttons that look like playground links |
| Connection | `.status-dot.is-ok` / `.is-err` `.runtime` | Replaces custom `.dot` |
| Theme | `.nav-switch` | Auto / Light / Dark |
| Page header | `.page-header` `.page-heading` `.page-title` `.page-desc` `.page-actions` | Sticky, blur |
| Primary action (Start) | `.button.button-primary` | Ink fill, not accent |
| Secondary (Settings) | `.button.button-secondary` | |
| Stop | `.button.button-danger` | Outlined danger |
| Ghost / icon close | `.button.button-ghost.button-icon.button-small` | |
| Camera / status / audio cards | `.panel.panel-pad` + `.panel-title` > `h2` + `.panel-help` | Not custom `.card` + uppercase `h3` |
| Speaking / silence / ROI / session | `.metric-row` > `.metric` > `span` + `strong` | Playground result metrics |
| State machine | `.fsm-axis` > `.fsm-node` + `.fsm-edge`; `.fsm-hist` > `.fsm-tr` (app.css) | Nodes on an axis; current node filled with its state colour, last transition's source dashed and path in accent |
| Audio signals | `.sig-group.raw` / `.sig-group.clean` (waveform + spectrogram) | One bundle per signal; mint / periwinkle edge |
| ROI editor | `.segmented#roiPick` (face / body / lock) + drag on `#view canvas` | Face cyan dashed, body purple dotted; only the selected ROI is draggable |
| Live caption | `#view > .caption` (`.is-in` fade, `.is-final` white; `--nav` 78 % backing, `--on-nav` text) | Single bottom line over the camera, never mirrored |
| Cross-session re-ID line | `.reid-line#reidSt` in the state card: decision (`↩ resume #n` in `--ok`, `＋ new` + reason), short `conversation_id` (mono), calls · tokens · cost · refs (mono, right) | Hidden when the feature is off; never shows face images |
| Monitor stage | `.stage-head` (title + `.segmented#trkView` wave / spectrum / both, outside the frame) + `section.panel.monitor-stage` (12 px gutter on `--nav-2`) = `#view` rounded inset + `.sig-stack` inset panel of `.sig-lane`s (`.lane-label` gutter: chip, name, dB; `.lane-plots` rounded canvases) | Modelled on PureSound CompareDeck; shared gutter keeps video edges and time axes aligned; `--stage-chrome` per view keeps it inside the viewport |
| Layout | Left column: monitor stage, then ROI card, then monitor/record/timeline card; right column: state, ASR, P(group), countdowns | |
| Monitor off/raw/clean | `.segmented` `[role=radiogroup]` `.is-active` | |
| Live camera chip | `.chip.is-live` | Accent border + pulse |
| Recording indicator | `#recInd` (pill on `--bad-soft`) | |
| Resource tiles | `.stat` > `span` + `strong` | |
| Latency table | `.table` + `.num` | |
| Empty tracks | `.empty-state` | Dashed panel |
| Toast | `.toast.is-visible` | Always-dark nav chrome |
| Error banner | `#errBanner` (toast chrome + `--nav-bad` rule) | |
| Settings drawer | `.inspector` `.inspector-head` `.inspector-body` `.inspector-section` `.eyebrow` | Opens with `.is-open` + `.inspector-backdrop` |
| Presets | `.segmented` | |
| Advanced toggle | `.toggle-row` `.toggle-ui` | |
| Bool settings | `.toggle-row` | Built by `rowHTML` |
| Selects / text / numbers | `.control` | |
| Sliders | `.range` + `.field-label` + `output` | Accent thumb |
| Hints | `.toggle-hint` / `.tip` | |
| Module strip | `.chip` in `.flow-row` | Accent when active |
| Module blocks | `.mod` `<details>` | Presence-only accordion on inspector density |
| Scopes (wave / spectrum / timeline) | always-dark `--nav` ground | Same as playground audio players |
| Raw / clean series | `--mint` / `--periwinkle` | Same signal colours as playground |

## Role aliases in `:root` (app.css)

`--ok` → `--good`, `--idle` → `--mint-ink`, `--end` → `--ink-2`, `--viz-raw` → `--mint`,
`--viz-clean` → `--periwinkle`, `--violet` → `--magenta`. Camera overlays use the dark-theme hex
values of those tokens because the camera ground is always dark.

## Density and layout

- 4 px grid, `--panel-pad` 16, `--gap` 16, control height 32 / 26, radius 4 / 6 / pill — all from tokens.
- Workbench: main column (camera panel) + right stack at `--inspector-w` (400 px ≥ 1400 px).
- Inspector drawer 520 px; sections separated by hairlines; eyebrow labels for Presets / Data flow / Import.
- One primary button per page (Start). Accent is only for focus, selection, live, and P(group).

## Accessibility

WCAG 2.1 AA on dashboard + settings, light and dark (headless Chrome). `--ink-3` / `--on-nav-3` are
forced to `--ink-2` / `--on-nav-2` for text in `app.css` because upstream tertiary ink is below 4.5:1
in places. Focus ring and reduced-motion come from `base.css`.

## Do's and Don'ts

- Do add UI with PureSound classes first; only invent a class when no upstream component fits.
- Do keep `app.css` free of upstream token and component redefinitions (enforced by tests).
- Don't hard-code hex in components except camera overlays (documented above).
- Don't show real camera frames in docs screenshots.
- Don't hard-code UI text in HTML or JS; add it to both string tables.
