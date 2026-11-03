"""Alert Dog, live: three households with the same patient, side by side.

The same virtual patient (adult#001) lives the same seeded routine in three households: one
with the kennel-trained Cadence dog, one with the monitor's trend alarm (the dog's trainer),
one with no alert at all. Interventions from the page (an extra bolus, exercise, a changed
body) reach all three at the same minute. The households diverge where an alert changes a
patient's day.

The dog is loaded from the kennel (``brains/``; trained on first start, about half a
minute) and then lives through ``Brain.live``: nothing is reset while the server runs.

    python server.py            # http://localhost:8672

Console lines: a quiet status every ~2s, plus a line whenever the dog alerts,
learns, eats, or the page intervenes.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from aiohttp import web

from alertdog import companions as C
from alertdog import kennel
from alertdog.life import Life
from alertdog.world import DAY, STEP, World

HERE = Path(__file__).resolve().parent
WINDOW = DAY // STEP  # one day of moments on the chart
SPEEDS = {"slow": 2, "normal": 8, "fast": 40, "faster": 160}  # moments per second
HOME = (("home", 0, 10**6, 1.0),)


class House:
    def __init__(self, key: str, label: str, companion, seed: int) -> None:
        self.key, self.label = key, label
        self.life = Life(World("adult#001", seed=seed), companion, stages=HOME)
        self.glucose: deque = deque(maxlen=WINDOW)
        self.marks: deque = deque(maxlen=400)
        self.level: deque = deque(maxlen=WINDOW)
        self.days: dict[int, dict[str, float]] = {}

    def moment(self):
        m, d = self.life.moment()
        self.glucose.append([m.minute, round(m.glucose, 1)])
        self.level.append([m.minute, round(float(d.get("level", 0.0)), 3), d.get("mode")])
        for kind, on in (("alert", m.alert), ("treat", m.treated > 0), ("meal", m.meal > 0),
                         ("bolus", m.bolus > 0), ("exercise", "exercise" in m.events),
                         ("felt", "felt low" in m.events), ("nuisance", m.nuisance)):
            if on:
                self.marks.append([m.minute, kind])
        day = self.days.setdefault(m.minute // DAY, {
            "moments": 0, "below": 0, "severe": 0, "alerts": 0, "nuisance": 0, "aroused": 0})
        day["moments"] += 1
        day["below"] += m.below
        day["severe"] += m.severe
        day["alerts"] += m.alert
        day["nuisance"] += m.nuisance
        day["aroused"] += d.get("mode") == "aroused"
        return m, d

    def view(self) -> dict:
        t = next(iter(self.life.tallies.values()), None)
        rep = t.report() if t else {}
        out = {
            "key": self.key, "label": self.label,
            "glucose": list(self.glucose), "marks": list(self.marks),
            "now": self.glucose[-1][1] if self.glucose else None,
            "total": rep,
            "days": [{"day": k, **v} for k, v in sorted(self.days.items())[-14:]],
        }
        if self.key == "dog":
            dog = self.life.companion
            last = self.life.last
            life = dog.describe()
            totals = {"total_sweeps": life.pop("sweeps"), "total_learning_sweeps": life.pop("learning_sweeps")}
            out["dog"] = {
                **life, **totals,
                "mode": last.get("mode"), "alert": bool(last.get("alert")),
                "level": round(float(last.get("level", 0.0)), 3),
                "surprise": round(float(last.get("surprise", 0.0)), 3),
                "sweeps": last.get("sweeps", 0), "learning_sweeps": last.get("learning_sweeps", 0),
                "ms": round(float(last.get("ms", 0.0)), 2),
                "level_history": list(self.level),
                "motor": motor(dog),
                "activation": activation(dog),
            }
        return out


def motor(dog: C.Dog) -> list[float] | None:
    state = dog.brain.basal_ganglia.state
    if state is None:
        return None
    act = np.asarray(state.activation)[0]
    return [round(float(v), 3) for v in act[dog.brain.motor_index]]


def activation(dog: C.Dog) -> list[float] | None:
    state = dog.brain.basal_ganglia.state
    if state is None:
        return None
    return [round(float(v), 3) for v in np.asarray(state.activation)[0]]


def wiring(dog: C.Dog) -> dict:
    """The dog's fixed graph: which neuron belongs to which population, and every synapse."""
    c = dog.brain.connectome
    return {
        "n": int(c.n),
        "populations": {k: list(v) for k, v in c.populations.items() if "/" not in k},
        "pre": np.asarray(c.pre).astype(int).tolist(),
        "post": np.asarray(c.post).astype(int).tolist(),
        "sign": np.asarray(c.sign).astype(int).tolist(),
    }


def stamp(world: World) -> str:
    """A short wall-clock label for console lines: day and HH:MM."""
    c = world.clock
    return f"第{world.day + 1}天 {c // 60:02d}:{c % 60:02d}"


class Household:
    """The three households and the clock that moves them."""

    def __init__(self, seed: int, dog_seed: int) -> None:
        began = time.time()
        dog = kennel.trained(dog_seed)
        self.kennel_seconds = round(time.time() - began, 1)
        self.houses = [
            House("dog", "Cadence alert dog", dog, seed),
            House("rule", "Monitor trend alarm (the dog's trainer)", C.Rule(), seed),
            House("none", "No alert", C.Never(), seed),
        ]
        self.speed = SPEEDS["normal"]
        self.paused = False
        self.body = 1.0
        self.lock = threading.Lock()
        self.log: deque = deque(maxlen=30)
        self._last_status = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _say(self, world: World, text: str) -> None:
        print(f"[{stamp(world)}] {text}", flush=True)

    def _status(self, force: bool = False) -> None:
        """A quiet heartbeat: three glucoses and the dog's mode, at most every 2s."""
        now = time.time()
        if not force and now - self._last_status < 2.0:
            return
        self._last_status = now
        dog, rule, none = self.houses
        w = dog.life.world
        d = dog.life.last
        mode = {"aroused": "警觉·学习", "routine": "平静"}.get(d.get("mode"), d.get("mode") or "—")
        bits = [
            f"血糖 狗{dog.glucose[-1][1]:.0f} 规则{rule.glucose[-1][1]:.0f} 无提醒{none.glucose[-1][1]:.0f}"
            if dog.glucose else "血糖 —",
            mode,
        ]
        if d.get("alert"):
            bits.append("提醒")
        if w.asleep:
            bits.append("睡着")
        if self.body != 1.0:
            bits.append(f"身体×{self.body}")
        self._say(w, " · ".join(bits))

    def _events(self, moments: list) -> None:
        """Print only the interesting moments: alerts, treatments, meals, real surprises.

        The first day at home is continuously aroused; that alone is not logged as an
        event (the quiet status line already shows 警觉·学习). A learning line is
        printed when surprise > 0, or when learning coincides with another event.
        """
        (m, d), (mr, _), (mn, _) = moments
        parts: list[str] = []
        surprise = float(d.get("surprise", 0) or 0)
        aroused = d.get("mode") == "aroused"
        if m.alert:
            parts.append("提醒（误报）" if m.nuisance else "提醒")
        if mr.alert:
            parts.append("规则也提醒" if m.alert else "规则提醒")
        if m.treated > 0:
            parts.append(f"吃了 {m.treated:.0f}g 糖")
        if m.meal > 0:
            parts.append(f"吃饭 {m.meal:.0f}g")
        if m.bolus > 0:
            parts.append(f"打针 {m.bolus:.1f}U")
        if "exercise" in m.events:
            parts.append("运动")
        if "felt low" in m.events:
            parts.append("自己觉得低了")
        if aroused and (parts or surprise > 0):
            parts.insert(
                0,
                f"警觉·学习 surprise={surprise:.2f} "
                f"学习扫={d.get('learning_sweeps', 0)} 扫={d.get('sweeps', 0)}",
            )
        if not parts:
            return
        self._say(
            self.houses[0].life.world,
            f"{' · '.join(parts)} · 血糖 狗{m.glucose:.0f} 规则{mr.glucose:.0f} 无提醒{mn.glucose:.0f}",
        )

    def _run(self) -> None:
        while True:
            began = time.time()
            if not self.paused:
                with self.lock:
                    moments = []
                    for h in self.houses:
                        h.life.override = self.body
                        moments.append(h.moment())
                    self._events(moments)
                    self._status()
            time.sleep(max(0.0, 1.0 / self.speed - (time.time() - began)))

    def act(self, kind: str) -> str:
        with self.lock:
            world = self.houses[0].life.world
            cr = world.params.carb_ratio
            if kind == "bolus":
                for h in self.houses:
                    h.life.world.intervene("bolus", round(50 / cr, 2))
                note = f"多打了 {50 / cr:.1f} U 胰岛素（相当于 50 克饭的量，但没有吃饭）"
            elif kind == "exercise":
                for h in self.houses:
                    h.life.world.intervene("exercise", 50.0)
                note = "开始运动 50 分钟"
            elif kind == "meal":
                for h in self.houses:
                    h.life.world.intervene("meal", 60.0)
                    h.life.world.intervene("bolus", round(60 / cr, 2))
                note = "加了一顿 60 克的饭，并按量打针"
            elif kind == "sensitive":
                self.body = 1.3
                note = "身体对胰岛素的敏感性提高了 30%"
            elif kind == "restore":
                self.body = 1.0
                note = "身体恢复原样"
            elif kind == "pause":
                self.paused = not self.paused
                note = "暂停" if self.paused else "继续"
            elif kind in SPEEDS:
                self.speed = SPEEDS[kind]
                note = "速度：" + {"slow": "慢", "normal": "正常", "fast": "快", "faster": "很快"}[kind]
            else:
                raise web.HTTPBadRequest(text=f"unknown action {kind}")
            self.log.appendleft([world.minute, note])
            print(f"[{stamp(world)}] 干预 · {note}", flush=True)
            return note

    def view(self) -> dict:
        with self.lock:
            w = self.houses[0].life.world
            return {
                "minute": w.minute, "day": w.day, "clock": w.clock, "asleep": w.asleep,
                "speed": self.speed, "paused": self.paused, "body": self.body,
                "patient": w.params.name, "kennel_seconds": self.kennel_seconds,
                "log": list(self.log),
                "houses": [h.view() for h in self.houses],
            }


def evidence() -> dict | None:
    files = sorted((HERE / "evidence").glob("adult001-seeds*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    means = {stage: {kind: {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
                     for kind, row in rows.items()} for stage, rows in data["means"].items()}
    return {"file": files[-1].name, "means": means, "configuration": {
        "seeds": data["configuration"]["seeds"], "stages": data["configuration"]["stages"]},
        "environment": data["environment"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8672)
    ap.add_argument("--seed", type=int, default=7, help="the households' life seed")
    ap.add_argument("--dog", type=int, default=10, help="the kennel seed of the dog")
    args = ap.parse_args()
    print("training or loading the kennel dog ...", flush=True)
    home = Household(args.seed, args.dog)
    print(f"ready: http://localhost:{args.port}", flush=True)

    async def index(_):
        return web.FileResponse(HERE / "static" / "index.html")

    async def state(_):
        return web.json_response(home.view())

    async def act(request):
        return web.json_response({"note": home.act(request.match_info["kind"])})

    async def proof(_):
        return web.json_response(evidence())

    graph = wiring(home.houses[0].life.companion)

    async def brain(_):
        return web.json_response(graph)

    async def weights(_):
        dog = home.houses[0].life.companion
        with home.lock:
            efficacy = np.asarray(dog.brain.brain.efficacy, dtype="<f4").copy()
            age = int(dog.brain.arousal.age)
        return web.json_response({"age": age, "efficacy": base64.b64encode(efficacy.tobytes()).decode()})

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_post("/api/act/{kind}", act)
    app.router.add_get("/api/evidence", proof)
    app.router.add_get("/api/brain", brain)
    app.router.add_get("/api/weights", weights)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
