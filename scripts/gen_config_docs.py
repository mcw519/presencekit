"""Generate the parameter reference in docs/configuration.md (and the zh-TW mirror).

The tables come straight from the code, so they cannot drift:

* defaults and ranges: ``presencekit.config.Config`` / ``RANGES`` (server keys)
  and ``DEF`` in ``static/js/settings.js`` (browser-only keys);
* module grouping, gates and "advanced" flags: ``MODULES`` in ``settings.js``;
* labels and help text: the UI string tables ``static/i18n/<lang>.js``.

Usage::

    uv run python scripts/gen_config_docs.py          # rewrite the docs
    uv run python scripts/gen_config_docs.py --check  # exit 1 if they are stale (used by tests)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from presencekit.config import RANGES, Config  # noqa: E402

STATIC = ROOT / "presencekit" / "static"
BEGIN = "<!-- BEGIN GENERATED: scripts/gen_config_docs.py -->"
END = "<!-- END GENERATED -->"
TARGETS = {"en": ROOT / "docs" / "configuration.md", "zh-TW": ROOT / "docs" / "zh-TW" / "configuration.md"}
HEAD = {
    "en": ("Key", "Default", "Range", "Description", "advanced", "browser only", "Mode / switch"),
    "zh-TW": ("鍵", "預設", "範圍", "說明", "進階", "只在瀏覽器", "模式／開關"),
}


def load_strings(lang: str) -> dict[str, str]:
    """Parse ``I18N["<lang>"] = {...};`` (a JSON object literal) from the string table."""
    src = (STATIC / "i18n" / f"{lang}.js").read_text(encoding="utf-8")
    body = src[src.index("{", src.index("I18N[")) : src.rindex("}") + 1]
    return json.loads(body)


def load_modules() -> list[dict]:
    """Extract id, gate, keys and adv of every settings module from settings.js."""
    src = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")
    block = src[src.index("const MODULES=[") : src.index("const MOD_PRESETS")]
    mods = []
    for chunk in re.split(r"\n \{id:", block)[1:]:
        mid = re.match(r"'(\w+)'", chunk).group(1)
        gate = re.search(r"gate:'(\w+)'", chunk)
        keys = re.findall(r"'(\w+)'", re.search(r"keys:\[(.*?)\]", chunk).group(1))
        adv = re.findall(r"'(\w+)'", re.search(r"adv:\[(.*?)\]", chunk, re.S).group(1))
        mods.append({"id": mid, "gate": gate.group(1) if gate else None, "keys": keys, "adv": set(adv)})
    return mods


def client_defaults() -> dict[str, str]:
    """Browser-only defaults (keys that are not part of the server Config) from ``DEF``."""
    src = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")
    body = src[src.index("const DEF={") + len("const DEF={") : src.index("};", src.index("const DEF={"))]
    return dict(re.findall(r"(\w+):(\[[^\]]*\]|'[^']*'|[^,]+)", body))


def fmt(v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    if isinstance(v, str):
        return f"`{v}`" if v else "—"
    return f"{v:g}" if isinstance(v, float) else str(v)


def fmt_range(key: str) -> str:
    if key not in RANGES:
        return "—"
    lo, hi, typ = RANGES[key]
    if typ is str:
        return " / ".join(f"`{o}`" for o in lo)
    if typ is bool:
        return "on / off"
    if typ == "text":
        return f"text ≤ {hi}"
    if typ in ("box", tuple) or isinstance(lo, (tuple, list)):
        return "box (x0, y0, x1, y1)"
    return f"{fmt(lo)} – {fmt(hi)}"


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render(lang: str) -> str:
    s = load_strings(lang)
    cfg = Config()
    server = {f.name for f in fields(Config)}
    client = client_defaults()
    h = HEAD[lang]
    out = []
    for m in load_modules():
        out.append(f"### {s.get('mod.' + m['id'], m['id'])}\n")
        out.append(f"| {h[0]} | {h[1]} | {h[2]} | {h[3]} |\n|---|---|---|---|")
        keys = ([m["gate"]] if m["gate"] else []) + m["keys"]
        for k in keys:
            if k in server:
                default = fmt(getattr(cfg, k))
            else:
                raw = client.get(k)
                default = "—" if raw is None else fmt(json.loads(raw.replace("'", '"')))
            label = s.get(f"s.{k}", k)
            helptext = s.get(f"s.{k}.h", "")
            flags = []
            if k == m["gate"]:
                flags.append(h[6])
            if k in m["adv"]:
                flags.append(h[4])
            if k not in server:
                flags.append(h[5])
            tag = f" *({', '.join(flags)})*" if flags else ""
            desc = f"**{cell(label)}**{tag}" + (f" — {cell(helptext)}" if helptext else "")
            out.append(f"| `{k}` | {cell(default)} | {fmt_range(k)} | {desc} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def splice(path: Path, body: str) -> str:
    text = path.read_text(encoding="utf-8")
    a, b = text.index(BEGIN), text.index(END)
    return text[: a + len(BEGIN)] + "\n\n" + body + "\n" + text[b:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only check that the docs are up to date")
    args = ap.parse_args()
    stale = []
    for lang, path in TARGETS.items():
        new = splice(path, render(lang))
        if new != path.read_text(encoding="utf-8"):
            stale.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(new, encoding="utf-8")
    if args.check and stale:
        print("stale:", ", ".join(stale), "- run scripts/gen_config_docs.py")
        return 1
    print("updated:" if stale else "up to date", ", ".join(stale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
