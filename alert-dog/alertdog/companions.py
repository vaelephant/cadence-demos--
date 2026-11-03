"""The companions: the Cadence alert dog and the controls it is measured against.

Every companion answers ``decide(world, reward)``: ``reward`` is what the patient lived
through after the companion's previous decision (None at the first moment), and the answer
is whether to alert now. The dog is one continuing ``Brain.compose`` life with arousal: it
senses the moment, its two motor neurons settle with the rest of its graph into "quiet" or
"alert", and ``Brain.live`` decides from the outcome whether this moment is routine (one
settle, no learning) or repair (sampling, learning, memory writes).

Before it lives with the patient on its own, the dog is an apprentice (``watch``): the
trainer, the monitor's trend rule, is at the controls, and the dog is taught the trainer's
decision on the trainer's lessons, as alert dogs are trained by a trainer before placement.

Controls, each declared as such:

``rule``    the trend alarm built into continuous glucose monitors and the dog's trainer:
            alert when the reading is below 75 or the 15-minute trend projects below 70
            within 20 minutes.
``wild``    a dog with no apprenticeship, learning from outcomes from its first moment.
``none``    never alerts: the patient's own symptoms only.
``random``  alerts at a fixed rate with no regard to the patient.
``frozen``  the apprenticed dog's brain answering greedily, never learning again: the
            control for "kept learning".
``always``  the apprenticed dog's brain learning from every outcome through ``Brain.step``:
            the control for arousal-gated learning.
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from cadence import Brain, LearningPhaseError

from .world import World

QUIET, ALERT = 0, 1

GLUCOSE = (55, 70, 85, 100, 120, 150, 190, 250)
TREND = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0)  # mg/dL per minute over the last 15 minutes
INSULIN = (0.0, 2.0, 4.0, 8.0)  # units on board
CARBS = (0.0, 20.0, 40.0, 80.0)  # grams on board
INPUTS = len(GLUCOSE) + len(TREND) + 8 + len(INSULIN) + len(CARBS) + 4 + 2

FOUNDER: dict[str, Any] = {
    # Hand-set from the robot arena's nursery point (robot-arena/arena/brain.py), not tuned
    # here; every value is a gene and this set is the control for any later selection.
    "modules": (48,),
    "sensory_scale": 4.0,
    "temperature": 0.3,
    "trace_amplitude": 0.3,
    "trace_decay": 0.1,
    "episodic": True,
    "consolidation": 0.05,
    "eta": 0.03,
    "eta_bias": 0.003,
    "lam": 0.8,  # credit reaches about half an hour back: an alert acts through a meal
    "gamma": 0.95,
    "eta_critic": 5.0,
    "arousal": {
        "threshold": 0.2,
        "decay": 0.9,
        "tolerance": 2.0,
        "floor": 0.1,
        "fast": 0.05,
        "slow": 0.005,
        "heat": 1.0,
        "youth": 288,  # its first day at the controls is aroused: settling in
        "value_surprise": 1.0,
        "record_surprise": 0.0,
        "need": 0.0,  # every reward here is a cost; a need would be a want forever
    },
}


ADULT: dict[str, Any] = {
    # Leaving the kennel, as the robot arena's ring stage does: no heat (want no longer
    # widens exploration), a sharper sampling temperature, so an aroused adult samples near
    # its own best answer, and a tenth of the actor step, so repair refines the kennel's
    # training instead of overwriting it. Chosen on development seeds 0-2 between the
    # founder step 0.03 and 0.003 (evidence/dev-kennel.json, evidence/dev-adult-eta.json).
    "heat": 0.0,
    "temperature": 0.15,
    "eta": 0.003,
}


def grow_up(brain: Brain, adult: dict[str, Any] | None = None) -> None:
    a = {**ADULT, **(adult or {})}
    genes: dict[str, Any] = {"temperature": float(a["temperature"])}
    if "eta" in a:
        genes.update(actor_eta=float(a["eta"]), actor_eta_bias=float(a["eta"]) / 10.0)
    brain.retune(arousal={"heat": float(a["heat"])}, **genes)


def _bumps(value: float, centres: tuple[float, ...]) -> list[float]:
    """Population code: each centre fires by its distance to the value, in centre spacings."""
    out = []
    for i, c in enumerate(centres):
        width = (centres[min(i + 1, len(centres) - 1)] - centres[max(i - 1, 0)]) / 2 or 1.0
        out.append(math.exp(-(((value - c) / width) ** 2)))
    return out


def senses(world: World) -> np.ndarray:
    """What the dog perceives at this moment: the reading and its trend, the time of day, the
    insulin and carbohydrate on board, how long since exercise, and whether the patient sleeps
    or has just eaten for a low."""
    h = world.history
    g = h[-1]
    trend = (g - h[-4]) / 15.0 if len(h) >= 4 else 0.0
    hour = world.clock // 180
    since = world.since_exercise()
    exercise = [0.0] * 4
    if since is not None:
        exercise[0 if since == 0 else 1 if since < 120 else 2 if since < 300 else 3] = 1.0
    x = (
        _bumps(g, GLUCOSE)
        + _bumps(max(-3.5, min(2.5, trend)), TREND)
        + [1.0 if i == hour else 0.0 for i in range(8)]
        + _bumps(min(10.0, world.insulin_on_board()), INSULIN)
        + _bumps(min(100.0, world.carbs_on_board()), CARBS)
        + exercise
        + [1.0 if world.asleep else 0.0, 1.0 if world.minute - world.last_treat < 30 else 0.0]
    )
    return np.asarray([x], dtype=float)


def compose(seed: int = 0, genes: dict[str, Any] | None = None) -> Brain:
    """A newborn dog: every sense an input, two motor neurons, quiet and alert."""
    g = copy.deepcopy(FOUNDER) if genes is None else genes
    return Brain.compose(
        INPUTS,
        2,
        modules=tuple(g["modules"]),
        seed=seed,
        episodic=bool(g["episodic"]),
        consolidation=float(g["consolidation"]),
        working_memory_amplitude=float(g["trace_amplitude"]),
        working_memory_decay=float(g["trace_decay"]),
        arousal=g["arousal"],
        sensory_scale=float(g["sensory_scale"]),
        temperature=float(g["temperature"]),
        actor_eta=float(g["eta"]),
        actor_eta_bias=float(g["eta_bias"]),
        actor_lam=float(g["lam"]),
        actor_gamma=float(g["gamma"]),
        actor_eta_critic=float(g["eta_critic"]),
    )


def _quiet(mode: str, ms: float = 0.0, **extra: Any) -> dict[str, Any]:
    return {"alert": False, "mode": mode, "sweeps": 0, "learning_sweeps": 0, "level": 0.0, "ms": ms, **extra}


class Dog:
    """The alert dog: one ``Brain.live`` life with this patient."""

    kind = "dog"

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.owed = 0.0  # an outcome not yet delivered because an answer refused
        self.refusals = 0

    @classmethod
    def newborn(cls, seed: int = 0) -> "Dog":
        return cls(compose(seed))

    @classmethod
    def load(cls, path: str | Path) -> "Dog":
        return cls(Brain.load(Path(path)))

    def save(self, path: str | Path) -> Path:
        return self.brain.save(Path(path))

    def grow_up(self, adult: dict[str, Any] | None = None) -> None:
        grow_up(self.brain, adult)

    def watch(self, world: World, trainer_alert: bool, lesson: bool) -> dict[str, Any]:
        """A moment of the apprenticeship: the trainer is at the controls. First the dog's own
        answer, read greedily from its settled state and not executed; then, if this moment
        is one of the trainer's lessons, the trainer's decision as its label."""
        brain, x = self.brain, senses(world)
        began = time.perf_counter()
        drive = brain.stimulus(x)
        try:
            own = int(np.atleast_1d(brain.act(x, greedy=True))[0]) == ALERT
        except (RuntimeError, ValueError):
            own = None
            self.refusals += 1
        settled = brain.last_settlement or {}
        report: dict[str, Any] = {}
        if lesson:
            try:
                _, report = brain.learner.step(drive, np.array([ALERT if trainer_alert else QUIET]))
            except LearningPhaseError as error:
                report = dict(error.report)
                self.refusals += 1
        return {
            "own": own,
            "agree": own == trainer_alert,
            "lesson": lesson,
            "unsettled": bool(report.get("free_budget_exhausted", 0.0)),
            "sweeps": int(settled.get("steps", 0)),
            "learning_sweeps": int(report.get("free_steps", 0)) if lesson else 0,
            "ms": (time.perf_counter() - began) * 1000.0,
        }

    def decide(self, world: World, reward: float | None) -> dict[str, Any]:
        brain, x = self.brain, senses(world)
        began = time.perf_counter()
        feedback: dict[str, Any] = {}
        if reward is not None and brain.pending_feedback:
            feedback = {"reward": [reward + self.owed], "done": [False]}
        elif reward is not None:
            self.owed += reward
        action = None
        try:
            action = brain.live(x, **feedback)
            self.owed = 0.0
        except (RuntimeError, ValueError):
            self.refusals += 1
            if feedback and not brain.pending_feedback:
                try:  # the outcome was taken; only the answer refused
                    action = brain.live(x)
                    self.owed = 0.0
                except (RuntimeError, ValueError):
                    pass
            elif feedback:
                self.owed = float(feedback["reward"][0])
        ms = (time.perf_counter() - began) * 1000.0
        if action is None:
            return _quiet("refused", ms)
        reading = brain.last_arousal or {}
        arousal = brain.arousal
        mode = reading.get("mode", arousal.mode)
        return {
            "alert": int(np.atleast_1d(action)[0]) == ALERT,
            "mode": mode,
            "sweeps": int(reading.get("sweeps", 0)),
            "learning_sweeps": int(reading.get("learning_sweeps", 0)),
            "level": float(arousal.level),
            "surprise": float(reading.get("surprise", 0.0)),
            "ms": ms,
        }

    def describe(self) -> dict[str, Any]:
        a = self.brain.arousal
        return {
            "neurons": int(self.brain.connectome.n),
            "parameters": int(self.brain.parameters()),
            "age": int(a.age),
            "moments": {k: int(v) for k, v in a.moments.items()},
            "sweeps": {k: int(v) for k, v in a.sweeps.items()},
            "learning_sweeps": int(a.learning_sweeps),
            "refusals": self.refusals,
        }


class Frozen:
    """A dog's brain that answers greedily and never learns."""

    kind = "frozen"

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.refusals = 0

    def decide(self, world: World, reward: float | None) -> dict[str, Any]:
        began = time.perf_counter()
        try:
            action = self.brain.act(senses(world), greedy=True)
        except (RuntimeError, ValueError):
            self.refusals += 1
            return _quiet("refused", (time.perf_counter() - began) * 1000.0)
        settled = self.brain.last_settlement or {}
        return {
            "alert": int(np.atleast_1d(action)[0]) == ALERT,
            "mode": "frozen",
            "sweeps": int(settled.get("steps", 0)),
            "learning_sweeps": 0,
            "level": 0.0,
            "ms": (time.perf_counter() - began) * 1000.0,
        }


class Always:
    """A dog's brain that learns from every outcome (``Brain.step``), aroused or not."""

    kind = "always"

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.refusals = 0

    def grow_up(self) -> None:
        grow_up(self.brain)

    def decide(self, world: World, reward: float | None) -> dict[str, Any]:
        began = time.perf_counter()
        feedback = {"reward": [reward], "done": [False]} if reward is not None and self.brain.pending_feedback else {}
        try:
            action = self.brain.step(senses(world), **feedback)
        except (RuntimeError, ValueError):
            self.refusals += 1
            return _quiet("refused", (time.perf_counter() - began) * 1000.0)
        settled = self.brain.last_settlement or {}
        learning = self.brain.last_learning or {}
        return {
            "alert": int(np.atleast_1d(action)[0]) == ALERT,
            "mode": "always",
            "sweeps": int(settled.get("steps", 0)),
            "learning_sweeps": int(learning.get("free_steps", 0)),
            "level": 0.0,
            "ms": (time.perf_counter() - began) * 1000.0,
        }


class Rule:
    """The trend alarm of a continuous glucose monitor."""

    kind = "rule"

    def decide(self, world: World, reward: float | None) -> dict[str, Any]:
        h = world.history
        g = h[-1]
        trend = (g - h[-4]) / 15.0 if len(h) >= 4 else 0.0
        return {**_quiet("rule"), "alert": g < 75 or g + 20 * trend < 70}


class Never:
    kind = "none"

    def decide(self, world: World, reward: float | None) -> dict[str, Any]:
        return _quiet("none")


class Random:
    kind = "random"

    def __init__(self, rate: float = 0.02, seed: int = 0) -> None:
        self.rate, self.rng = rate, random.Random(seed)

    def decide(self, world: World, reward: float | None) -> dict[str, Any]:
        return {**_quiet("random"), "alert": self.rng.random() < self.rate}
