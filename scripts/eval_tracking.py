"""Compare tracker / presence-model variants on synthetic scenarios (no images at all).

Variants: legacy+rules (v0.4 behaviour), kalman+rules (v0.5), kalman+hmm (v0.6), kalman+hmm+score (v0.7:
intent-based join).
Scenarios are generated in memory as detector outputs (boxes, noisy face embeddings, torso colour
histograms, body keypoints) plus a VAD speech flag, with ground-truth person ids.

    uv run python scripts/eval_tracking.py            # markdown tables (alias: scripts/eval_presence.py)
    uv run python scripts/eval_tracking.py --seeds 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presencekit.body import Body  # noqa: E402
from presencekit.tracker import Config, Detection, PresenceTracker  # noqa: E402

FACE_W, FACE_H = 0.08, 0.11
VARIANTS = {
    "legacy+rules": {"tracker_mode": "legacy", "presence_model": "rules"},
    "kalman+rules": {"tracker_mode": "kalman", "presence_model": "rules"},
    "kalman+hmm": {"tracker_mode": "kalman", "presence_model": "hmm"},
    "kalman+hmm+score": {"tracker_mode": "kalman", "presence_model": "hmm", "intent_model": "score"},
}


def never(t):
    return False


def one(t):
    return 1.0


@dataclass
class Person:
    pid: int
    path: callable  # t -> (cx, cy) of the face, or None when out of the scene
    occluded: callable = never  # in the scene but not detected
    no_face: callable = never  # side/back view: body only
    scale: callable = one  # apparent size (shrinks when walking away)
    conf: callable = lambda t, rng: 0.9  # detector confidence  # noqa: E731
    frontal: float = 0.9  # frontal-face score (YuNet landmarks)
    view: str | None = None  # body orientation: facing | side | away (None: facing if face else side)
    bystander: bool = False  # should never become a participant
    joiner: bool = False  # should become a participant (time-to-join measured from first appearance)
    emb: np.ndarray = None
    app: np.ndarray = None


@dataclass
class Scenario:
    name: str
    persons: list
    duration: float
    dt: callable = one  # frame interval at time t
    speech: callable = never  # VAD speech flag at time t
    expect_new_people: bool = False  # NEW_GROUP / new session is legitimate
    leave_at: float | None = None  # everybody leaves (starts leaving) at this time
    brief_absence: tuple | None = None  # (t0, t1): out of scene briefly; AWAY in here is false
    cfg: dict = field(default_factory=dict)


def unit(v):
    return v / np.linalg.norm(v)


def make_identity(rng, p: Person):
    p.emb = unit(rng.normal(size=128))
    p.app = rng.dirichlet(np.full(48, 0.3))


def body_for(cx, cy, face, s=1.0, view=None):
    fh = FACE_H * s
    w = FACE_W * 3.2 * s
    top = cy - fh * 0.7
    kp = {
        "l_shoulder": (cx + w * 0.35, cy + fh * 1.1, 0.9),
        "r_shoulder": (cx - w * 0.35, cy + fh * 1.1, 0.9),
        "l_hip": (cx + w * 0.2, cy + fh * 4.0, 0.8),
        "r_hip": (cx - w * 0.2, cy + fh * 4.0, 0.8),
    }
    view = view or ("facing" if face else "side")
    if face:
        kp.update(nose=(cx, cy, 0.9), l_eye=(cx + 0.015 * s, cy - 0.01, 0.9), r_eye=(cx - 0.015 * s, cy - 0.01, 0.9))
    if view == "side":  # shoulders collapse
        kp["l_shoulder"], kp["r_shoulder"] = (cx - 0.01, cy + fh * 1.1, 0.9), (cx + 0.01, cy + fh * 1.1, 0.9)
    elif view == "away":  # back to the camera: left shoulder on image left
        kp["l_shoulder"], kp["r_shoulder"] = (cx - w * 0.35, cy + fh * 1.1, 0.9), (cx + w * 0.35, cy + fh * 1.1, 0.9)
    return Body((cx - w / 2, top, cx + w / 2, min(1.0, top + 0.75 * s)), 0.8, kp)


def detections(scn: Scenario, t: float, rng) -> tuple[list[Detection], list[int]]:
    dets, gt = [], []
    for p in scn.persons:
        pos = p.path(t)
        if pos is None or p.occluded(t):
            continue
        s = p.scale(t)
        cx, cy = pos[0] + rng.normal(0, 0.004), pos[1] + rng.normal(0, 0.004)
        face = not p.no_face(t)
        body = body_for(cx, cy, face, s, p.view)
        app = 0.85 * p.app + 0.15 * rng.dirichlet(np.full(48, 0.3))
        fw, fh = FACE_W * s, FACE_H * s
        fb = (cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2)
        conf = p.conf(t, rng)
        if face:
            emb = unit(p.emb + rng.normal(size=128) * 0.06)
            dets.append(Detection(fb, conf, emb, face_bbox=fb, body=body, appearance=app, frontal=p.frontal))
        else:
            dets.append(Detection(fb, min(conf, 0.75), None, face_bbox=None, body=body, appearance=app))
        gt.append(p.pid)
    return dets, gt


def lerp(a, b, t0, t1):
    def f(t):
        if t < t0:
            return a
        if t > t1:
            return b
        k = (t - t0) / (t1 - t0)
        return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)

    return f


def window(*spans):
    return lambda t: any(a <= t < b for a, b in spans)


def still(xy):
    return lambda t: xy


def gone_at(t_leave, xy=(0.5, 0.4)):
    return lambda t: xy if t < t_leave else None


def flicker(t, rng):
    return 0.9 if rng.random() < 0.6 else 0.4  # 40% of frames below min_conf


OCC = window((6, 7), (13, 15), (21, 24))  # 1 s, 2 s, 3 s occlusions
HALF = lambda t: 0.5  # noqa: E731


def scenarios() -> list[Scenario]:
    return [
        # --- tracking (v0.5) ---
        Scenario("遮擋 1–3 s（1 fps）", [Person(1, still((0.5, 0.4)), occluded=OCC)], 30),
        Scenario("遮擋 1–3 s（2 fps）", [Person(1, still((0.5, 0.4)), occluded=OCC)], 30, dt=HALF),
        Scenario(
            "兩人交錯走過",
            [
                Person(1, lerp((0.3, 0.4), (0.7, 0.4), 6, 14), occluded=window((9.5, 10.5))),
                Person(2, lerp((0.7, 0.42), (0.3, 0.42), 6, 14)),
            ],
            24,
        ),
        Scenario(
            "側身看不到臉",
            [Person(1, lerp((0.4, 0.4), (0.46, 0.4), 6, 14), no_face=window((6, 14))), Person(2, still((0.65, 0.4)))],
            22,
        ),
        Scenario(
            "fps 變動 0.5–2",
            [Person(1, lambda t: (0.4 + 0.15 * np.sin(t / 4), 0.4))],
            30,
            dt=lambda t: 2.0 if int(t / 6) % 2 else 0.5,
        ),
        Scenario(
            "兩人側身互換位置",
            [
                Person(1, lerp((0.4, 0.4), (0.62, 0.4), 8, 11), no_face=window((7, 13))),
                Person(2, lerp((0.62, 0.4), (0.4, 0.4), 8, 11), no_face=window((7, 13))),
            ],
            20,
            dt=HALF,
        ),
        Scenario(
            "離開後新的人站到同一位置",
            [Person(1, gone_at(10)), Person(2, lambda t: (0.5, 0.4) if t >= 12 else None)],
            25,
            expect_new_people=True,
        ),
        Scenario("離開（1 fps）", [Person(1, gone_at(10))], 25, leave_at=10.0),
        Scenario("離開（2 fps）", [Person(1, gone_at(10))], 25, dt=HALF, leave_at=10.0),
        # --- presence model (v0.6) ---
        Scenario("低信心閃爍（40% 幀 conf 0.4）", [Person(1, still((0.5, 0.4)), conf=flicker)], 60),
        Scenario("ROI 空 + 電視聲", [Person(1, gone_at(10))], 30, speech=lambda t: t >= 8, leave_at=10.0),
        Scenario("長時間安靜在場（120 s）", [Person(1, still((0.5, 0.4)))], 120),
        Scenario(
            "慢慢離開（轉身、變小 6 s）",
            [
                Person(
                    1,
                    lambda t: (0.5 + 0.02 * max(0, t - 10), 0.4) if t < 16 else None,
                    no_face=lambda t: t >= 10,
                    scale=lambda t: 1.0 if t < 10 else max(0.45, 1.0 - 0.09 * (t - 10)),
                )
            ],
            30,
            leave_at=10.0,
        ),
        Scenario(
            "短暫走開又回來（3.5 s）",
            [Person(1, lambda t: None if 10 <= t < 13.5 else (0.5, 0.4))],
            30,
            dt=HALF,
            brief_absence=(10, 13.5),
        ),
        Scenario(
            "遠處路人（3.5 m）+ 參與者離開",
            [Person(1, gone_at(15)), Person(2, still((0.7, 0.35)), scale=lambda t: 0.4)],
            35,
            leave_at=15.0,
            cfg={"distance_enabled": True},
        ),
        # --- interaction intent (v0.7) ---
        Scenario(
            "路人橫越走過",
            [
                Person(1, still((0.5, 0.4)), joiner=True),
                Person(2, lambda t: (0.1 + 0.12 * (t - 8), 0.42) if 8 <= t < 15 else None, no_face=one, bystander=True),
            ],
            25,
        ),
        Scenario(
            "旁觀者站在旁邊看（參與者 25 s 離開）",
            [
                Person(1, gone_at(25), joiner=True),
                Person(2, lambda t: (0.25, 0.42) if t >= 3 else None, frontal=0.35, view="side", bystander=True),
            ],
            45,
            leave_at=25.0,
        ),
        Scenario(
            "旁邊背對滑手機（參與者 25 s 離開）",
            [
                Person(1, gone_at(25), joiner=True),
                Person(2, lambda t: (0.66, 0.42) if t >= 3 else None, no_face=one, view="away", bystander=True),
            ],
            45,
            leave_at=25.0,
        ),
        Scenario(
            "新來的人走近並說話",
            [
                Person(1, still((0.4, 0.4)), joiner=True),
                Person(2, lambda t: lerp((0.85, 0.42), (0.6, 0.42), 10, 13)(t) if t >= 10 else None, joiner=True),
            ],
            30,
            speech=window((13.5, 18)),
        ),
        Scenario(
            "三人一組，一人離開",
            [
                Person(1, still((0.35, 0.4)), joiner=True),
                Person(2, gone_at(20, (0.5, 0.4)), joiner=True),
                Person(3, still((0.65, 0.4)), joiner=True),
            ],
            40,
        ),
        Scenario(
            "新的人站到離開者的位置（換組）",
            [Person(1, gone_at(10), joiner=True), Person(2, lambda t: (0.5, 0.4) if t >= 12 else None, joiner=True)],
            30,
            expect_new_people=True,
        ),
    ]


@dataclass
class Result:
    id_switch: int = 0
    frag: int = 0
    false_maybe: int = 0
    false_away: int = 0
    false_new_group: int = 0
    flips: int = 0
    leave_s: list = field(default_factory=list)
    false_join: int = 0
    join_s: list = field(default_factory=list)
    missed_join: int = 0


def run(scn: Scenario, variant: dict, seed: int) -> Result:
    rng = np.random.default_rng(seed)
    for p in scn.persons:
        make_identity(rng, p)
    clk = [0.0]
    cfg = Config(pose_enabled=True, **variant, **scn.cfg)
    tr = PresenceTracker(cfg, clock=lambda: clk[0])
    res = Result()
    owner: dict[int, int] = {}
    last: dict[int, int] = {}
    joined: dict[int, float] = {}
    t = 0.0
    while t < scn.duration:
        dets, gt = detections(scn, t, rng)
        clk[0] = t
        n_log = len(tr.log)
        tr.update(dets, speaking=scn.speech(t))
        for d, g in zip(dets, gt, strict=True):
            tid = next(
                (k.id for k in tr.tracks.values() if k.visible_kind == "detected" and tuple(k.bbox) == tuple(d.bbox)),
                None,
            )
            if tid is None:
                continue
            prev = last.get(g)
            if prev is not None and prev != tid:
                if owner.get(tid, g) != g:
                    res.id_switch += 1
                else:
                    res.frag += 1
            elif prev is None and owner.get(tid, g) != g:
                res.id_switch += 1
            owner.setdefault(tid, g)
            last[g] = tid
        for k in tr.tracks.values():
            g = owner.get(k.id)
            if g is not None and k.participant:
                joined.setdefault(g, t)
        step = scn.dt(t)
        k = 0.5
        while k < step - 1e-9:  # server ticker every 0.5 s
            clk[0] = t + k
            tr.speaking = scn.speech(t + k)
            tr.tick()
            k += 0.5
        for e in tr.log[n_log:]:
            et = e["t"]
            if e["event"] == "transition":
                res.flips += 1
                before_leave = scn.leave_at is None or et < scn.leave_at
                if in_scene_at(scn, et) and before_leave:
                    res.false_maybe += e["to"] == "MAYBE_AWAY"
                    res.false_away += e["to"] == "AWAY"
                if scn.brief_absence and scn.brief_absence[0] <= et < scn.brief_absence[1] + 1.0:
                    res.false_away += e["to"] == "AWAY"
                if e["to"] == "AWAY" and scn.leave_at is not None and et >= scn.leave_at and not res.leave_s:
                    res.leave_s.append(et - scn.leave_at)
            if e["event"] == "NEW_GROUP" and not scn.expect_new_people:
                res.false_new_group += 1
        t += step
    for p in scn.persons:
        if p.bystander and p.pid in joined:
            res.false_join += 1
        if p.joiner:
            t0 = next((x / 10 for x in range(int(scn.duration * 10)) if p.path(x / 10) is not None), None)
            if p.pid in joined:
                res.join_s.append(joined[p.pid] - t0)
            else:
                res.missed_join += 1
    return res


def in_scene_at(scn: Scenario, t: float) -> bool:
    return any(p.path(t) is not None for p in scn.persons)


KEYS = ("id_switch", "frag", "false_maybe", "false_away", "false_new_group", "false_join")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    print(f"每個情境跑 {args.seeds} 個亂數種子；計數是總和，離開判定時間取平均，狀態切換是每分鐘平均。\n")
    print(
        "| 情境 | 版本 | ID 互換 | 斷裂 | 誤判 MAYBE_AWAY | 誤判 AWAY | 誤判 NEW_GROUP | 誤加入 | 加入時間 (s) "
        "| 離開判定 (s) | 狀態切換/分 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = {v: dict.fromkeys(KEYS, 0) for v in VARIANTS}
    for scn in scenarios():
        for name, var in VARIANTS.items():
            agg = dict.fromkeys(KEYS, 0)
            leave, flips, missed, joins, mj = [], 0, 0, [], 0
            for s in range(args.seeds):
                r = run(scn, var, s)
                for k in KEYS:
                    agg[k] += getattr(r, k)
                leave += r.leave_s
                joins += r.join_s
                mj += r.missed_join
                flips += r.flips
                missed += scn.leave_at is not None and not r.leave_s
            for k in KEYS:
                tot[name][k] += agg[k]
            lv = f"{np.mean(leave):.1f}" if leave else "—"
            if missed:
                lv += f"（{missed} 次沒判定）"
            js = f"{np.mean(joins):.1f}" if joins else "—"
            if mj:
                js += f"（{mj} 次沒加入）"
            fpm = flips / args.seeds / (scn.duration / 60)
            print(f"| {scn.name} | {name} | " + " | ".join(str(agg[k]) for k in KEYS) + f" | {js} | {lv} | {fpm:.1f} |")
    for name in VARIANTS:
        print(f"| **合計** | {name} | " + " | ".join(f"**{tot[name][k]}**" for k in KEYS) + " | | | |")


if __name__ == "__main__":
    main()
