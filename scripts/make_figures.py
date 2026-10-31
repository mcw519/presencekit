"""Render the plot figures in docs/img from the real tracker on synthetic scenes.

Usage (matplotlib is a dev-only dependency, not installed with the package):

    uv run --with matplotlib python scripts/make_figures.py

Writes ``docs/img/<name>.png`` (English) and ``docs/img/<name>.zh-TW.png``.
Mermaid diagrams (``docs/img/*.mmd``) are rendered separately with mermaid-cli,
see docs/README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import eval_tracking as ev  # noqa: E402

from presencekit.tracker import Config, PresenceTracker  # noqa: E402

OUT = ROOT / "docs" / "img"
STATE_COLOR = {"ACTIVE": "#2e7d32", "IDLE": "#7cb342", "MAYBE_AWAY": "#f9a825", "AWAY": "#e65100", "ENDED": "#9e9e9e"}

TXT = {
    "en": {
        "pres_title": "Presence: 3 s occlusion at t=8 s, person leaves at t=18 s (1 fps)",
        "p": "P(group)",
        "t": "time (s)",
        "occl": "occluded",
        "left": "left",
        "rules": "kalman + rules",
        "hmm": "kalman + hmm",
        "intent_title": "Intent score: visitor walks up and faces the device; passer-by crosses at 3.5 m",
        "score": "intent score",
        "visitor": "visitor",
        "passer": "passer-by",
        "join": "join 0.6",
        "drop": "drop 0.35",
        "lo_title": "Log-odds update per second of evidence (leave_halflife 1.5 s)",
        "lo_y": "P(present)",
        "seen": "detected (conf 0.9)",
        "unseen": "not seen",
        "low": "only low-confidence detections",
        "far": "beyond leave_distance",
    },
    "zh-TW": {
        "pres_title": "在場判斷：t=8 s 遮擋 3 秒，t=18 s 離開（1 fps）",
        "p": "P（群組）",
        "t": "時間（秒）",
        "occl": "被遮擋",
        "left": "已離開",
        "rules": "kalman + rules",
        "hmm": "kalman + hmm",
        "intent_title": "互動意圖分數：訪客走近並面向裝置；路人在 3.5 m 外走過",
        "score": "意圖分數",
        "visitor": "訪客",
        "passer": "路人",
        "join": "加入 0.6",
        "drop": "退出 0.35",
        "lo_title": "每秒證據對在場機率的影響（leave_halflife 1.5 s）",
        "lo_y": "P（在場）",
        "seen": "偵測到（信心 0.9）",
        "unseen": "沒看到",
        "low": "只有低信心偵測",
        "far": "超出 leave_distance",
    },
}


def setup_font(lang: str) -> None:
    if lang == "zh-TW":
        plt.rcParams["font.family"] = ["Noto Sans CJK TC", "Noto Serif CJK TC", "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def simulate(scn: ev.Scenario, variant: dict, seed: int = 1, track_pid: dict | None = None):
    """Run one scenario and return (times, snapshots)."""
    rng = np.random.default_rng(seed)
    for p in scn.persons:
        ev.make_identity(rng, p)
    clk = [0.0]
    tr = PresenceTracker(Config(pose_enabled=True, **variant, **scn.cfg), clock=lambda: clk[0])
    ts, snaps, t = [], [], 0.0
    while t < scn.duration:
        dets, _ = ev.detections(scn, t, rng)
        clk[0] = t
        snaps.append(tr.update(dets, speaking=scn.speech(t)))
        ts.append(t)
        t += scn.dt(t)
    return ts, snaps


def state_band(ax, ts, snaps, y0, h):
    for i, s in enumerate(snaps):
        t1 = ts[i + 1] if i + 1 < len(ts) else ts[i] + 1
        ax.broken_barh([(ts[i], t1 - ts[i])], (y0, h), color=STATE_COLOR.get(s["state"], "#ccc"))


def legend_states(ax):
    from matplotlib.patches import Patch

    ax.figure.legend(
        handles=[Patch(color=c, label=n) for n, c in STATE_COLOR.items()],
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
    )


def presence_figure(lang, name, scn_fn, title_key, spans):
    T = TXT[lang]
    fig, (ax, axs) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ts, hmm = simulate(scn_fn(), {"tracker_mode": "kalman", "presence_model": "hmm"})
    _, rules = simulate(scn_fn(), {"tracker_mode": "kalman", "presence_model": "rules"})
    ax.plot(
        ts, [s["p_group"] if s["p_group"] is not None else np.nan for s in hmm], color="#1565c0", lw=2, label=T["p"]
    )
    for y, lab in ((0.8, "p_active"), (0.6, "p_maybe"), (0.3, "p_away")):
        ax.axhline(y, ls="--", lw=0.8, color="#777")
        ax.text(ts[-1], y + 0.01, lab, ha="right", va="bottom", fontsize=8, color="#555")
    for (a, b), key in spans:
        ax.axvspan(a, b, color="#ef9a9a" if key == "left" else "#ffe082", alpha=0.35, lw=0)
        ax.text((a + b) / 2, 0.05, T[key], ha="center", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(T["p"])
    ax.set_title(T[title_key], fontsize=10)
    state_band(axs, ts, hmm, 1.1, 0.8)
    state_band(axs, ts, rules, 0.1, 0.8)
    axs.set_yticks([0.5, 1.5], [T["rules"], T["hmm"]], fontsize=8)
    axs.set_ylim(0, 2)
    axs.set_xlabel(T["t"])
    legend_states(axs)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / f"{name}{suffix(lang)}.png", dpi=130)
    plt.close(fig)


def scn_occlusion_leave():
    return ev.Scenario(
        "occl", [ev.Person(1, ev.gone_at(18.0), occluded=ev.window((8, 11)))], 30, cfg={"leave_distance": 3.0}
    )


def intent_figure(lang):
    T = TXT[lang]
    walk = ev.lerp((0.85, 0.45), (0.5, 0.4), 1, 6)
    small = lambda t: 0.45 + 0.55 * min(1.0, max(0.0, (t - 1) / 5))  # noqa: E731
    passer = ev.lerp((0.05, 0.42), (0.95, 0.42), 4, 14)
    scn = ev.Scenario(
        "intent",
        [
            ev.Person(1, walk, scale=small),
            ev.Person(2, lambda t: passer(t) if 4 <= t <= 14 else None, scale=lambda t: 0.35, no_face=lambda t: True),
        ],
        20,
        dt=ev.HALF,
    )
    ts, snaps = simulate(scn, {"tracker_mode": "kalman", "presence_model": "hmm", "intent_model": "score"})
    series: dict[int, list] = {}
    for i, s in enumerate(snaps):
        for k in s["tracks"]:
            series.setdefault(k["id"], [np.nan] * len(ts))[i] = k["intent"] if k["intent"] is not None else np.nan
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ids = sorted(series)
    for n, tid in enumerate(ids[:2]):
        ax.plot(ts, series[tid], lw=2, color=["#1565c0", "#8e24aa"][n], label=[T["visitor"], T["passer"]][n])
    ax.axhline(0.6, ls="--", color="#2e7d32", lw=1)
    ax.text(ts[-1], 0.61, T["join"], ha="right", fontsize=8, color="#2e7d32")
    ax.axhline(0.35, ls="--", color="#c62828", lw=1)
    ax.text(ts[-1], 0.36, T["drop"], ha="right", fontsize=8, color="#c62828")
    ax.set_ylim(0, 1)
    ax.set_xlabel(T["t"])
    ax.set_ylabel(T["score"])
    ax.set_title(T["intent_title"], fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"intent_timeline{suffix(lang)}.png", dpi=130)
    plt.close(fig)


def logodds_figure(lang):
    """Closed-form curves of the per-track update, starting from P=0.5."""
    T = TXT[lang]
    c = Config()
    lam = np.log(2) / c.leave_halflife
    t = np.linspace(0, 4, 200)
    lim = 3.9
    curves = [
        (T["seen"], c.detect_gain * 0.9, "#2e7d32"),
        (T["low"], -lam * c.lowconf_factor, "#f9a825"),
        (T["unseen"], -lam, "#e65100"),
        (T["far"], -3 * lam, "#b71c1c"),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for lab, rate, col in curves:
        L = np.clip(rate * t, -lim, lim)
        ax.plot(t, 1 / (1 + np.exp(-L)), lw=2, color=col, label=f"{lab}: {rate:+.2f}/s")
    for y in (0.8, 0.6, 0.3):
        ax.axhline(y, ls="--", lw=0.7, color="#888")
    ax.set_xlabel(T["t"])
    ax.set_ylabel(T["lo_y"])
    ax.set_ylim(0, 1)
    ax.set_title(T["lo_title"], fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / f"hmm_logodds{suffix(lang)}.png", dpi=130)
    plt.close(fig)


TABLES = {
    "rules_timeouts": {
        "en": (
            "Rules presence: silence timeout by priority_mode and what vision sees",
            ["priority_mode", "people visible", "ROI empty", "vision unsure"],
            [
                [
                    "vision",
                    "never times out on silence\n(present_max_silence cap, 0 = off)",
                    "empty_timeout 5 s\n(audio ignored)",
                    "base_timeout 20 s",
                ],
                [
                    "balanced",
                    "present_timeout 15 s",
                    "empty_timeout 5 s\nclock = min(silence, empty)",
                    "base_timeout 20 s",
                ],
                ["audio", "1.5 × base_timeout = 30 s", "0.5 × base_timeout = 10 s", "base_timeout 20 s"],
            ],
        ),
        "zh-TW": (
            "Rules 在場判斷：依 priority_mode 與影像狀態決定的靜默逾時",
            ["priority_mode", "看得到人", "ROI 空", "影像不確定"],
            [
                [
                    "vision",
                    "不因靜默離開\n（present_max_silence 上限，0＝關）",
                    "empty_timeout 5 s\n（不看聲音）",
                    "base_timeout 20 s",
                ],
                [
                    "balanced",
                    "present_timeout 15 s",
                    "empty_timeout 5 s\n計時＝min(靜默, 空的秒數)",
                    "base_timeout 20 s",
                ],
                ["audio", "1.5 × base_timeout = 30 s", "0.5 × base_timeout = 10 s", "base_timeout 20 s"],
            ],
        ),
    },
    "audio_decisions": {
        "en": (
            "Audio cross-check: per 20 ms chunk",
            ["case", "vision says", "audio says", "output", "counts as speech"],
            [
                ["agree", "person in range", "clean VAD: speech", "unchanged", "yes"],
                ["silence", "any", "no speech", "unchanged", "no"],
                ["leak", "nobody in ROI", "speech (TV, next room)", "−40 dB", "no"],
                ["false_accept", "person out of range", "speech", "−40 dB", "no"],
                ["false_reject", "person in range", "raw VAD only", "30% isolated + 70% raw", "yes"],
                ["acoustic_only", "no vision / disabled", "clean VAD", "PureSound output", "clean VAD"],
            ],
        ),
        "zh-TW": (
            "聲音交叉確認：每 20 ms 一塊",
            ["情況", "影像", "聲音", "輸出", "算說話"],
            [
                ["agree 一致", "範圍內有人", "乾淨 VAD：說話", "不變", "是"],
                ["silence 安靜", "任何", "沒說話", "不變", "否"],
                ["leak 漏音", "ROI 沒人", "有聲音（電視、隔壁）", "−40 dB", "否"],
                ["false_accept 誤收", "人在範圍外", "有聲音", "−40 dB", "否"],
                ["false_reject 漏收", "範圍內有人", "只有原始 VAD", "30% 分離 + 70% 原聲", "是"],
                ["acoustic_only 只看聲音", "沒有影像／關閉", "乾淨 VAD", "PureSound 輸出", "依乾淨 VAD"],
            ],
        ),
    },
}


def table_figure(lang: str, name: str) -> None:
    title, head, rows = TABLES[name][lang]
    fig, ax = plt.subplots(figsize=(10, 0.55 * (len(rows) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=head, loc="center", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 2.4)
    for (r, _c), cell in tb.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_facecolor("#e3eaf5")
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}{suffix(lang)}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def suffix(lang: str) -> str:
    return "" if lang == "en" else "." + lang


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "zh-TW"):
        setup_font(lang)
        presence_figure(
            lang, "presence_timeline", scn_occlusion_leave, "pres_title", [((8, 11), "occl"), ((18, 30), "left")]
        )
        intent_figure(lang)
        logodds_figure(lang)
        for name in TABLES:
            table_figure(lang, name)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
