"""A companion's life with one patient, in declared stages, and what it is measured by.

The stages of a life at home (days):

``arrival``   0-7    a new home: the dog's first day at the controls is its youth
``routine``   7-21   the same body; arousal decides when to learn
``changed``   21-35  the body becomes 30% more sensitive to insulin
``recovered`` 35-49  the body is as it was

With a ``trainer`` (the kennel), the trainer's decisions are executed and a companion that
can ``watch`` is shown every trainer alert and a random share (``QUIET_LESSONS``) of the
trainer's quiet moments as lessons. That curriculum is supplied by hand, as a dog trainer
chooses samples.

The measures are the clinic's: time below 70 and below 54 mg/dL, low episodes (15 minutes
or more below 70) per day and how many of them came at night, time in range 70-180, alerts
per day and the share of alerts after which the patient found nothing to treat. The work is
the brain's own count of settling sweeps and learning sweeps, and its wall time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .world import DAY, STEP, Moment, World

QUIET_LESSONS = 0.05

STAGES: tuple[tuple[str, int, int, float], ...] = (
    ("arrival", 0, 7, 1.0),
    ("routine", 7, 21, 1.0),
    ("changed", 21, 35, 1.3),
    ("recovered", 35, 49, 1.0),
)


def stage_of(day: int, stages=STAGES) -> tuple[str, float]:
    for name, start, end, sensitivity in stages:
        if start <= day < end:
            return name, sensitivity
    return stages[-1][0], stages[-1][3]


@dataclass
class Tally:
    moments: int = 0
    below: int = 0
    severe: int = 0
    in_range: int = 0
    alerts: int = 0
    nuisance: int = 0
    episodes: int = 0
    night_episodes: int = 0
    sweeps: int = 0
    learning_sweeps: int = 0
    aroused: int = 0
    ms: float = 0.0
    reward: float = 0.0
    watched: int = 0
    agreed: int = 0
    lessons: int = 0
    unsettled: int = 0  # lessons whose free phase ran out of its budget before settling

    def report(self) -> dict[str, float | None]:
        days = self.moments * STEP / DAY or 1.0
        minutes = self.moments * STEP or 1
        return {
            "days": round(days, 2),
            "below_70_pct": round(100 * self.below / minutes, 2),
            "below_54_pct": round(100 * self.severe / minutes, 2),
            "in_range_pct": round(100 * self.in_range / (self.moments or 1), 1),
            "lows_per_day": round(self.episodes / days, 2),
            "night_lows_per_day": round(self.night_episodes / days, 2),
            "alerts_per_day": round(self.alerts / days, 2),
            "nuisance_share": round(self.nuisance / self.alerts, 2) if self.alerts else None,
            "aroused_share": round(self.aroused / (self.moments or 1), 3),
            "sweeps_per_moment": round(self.sweeps / (self.moments or 1), 1),
            "learning_sweeps_per_moment": round(self.learning_sweeps / (self.moments or 1), 2),
            "ms_per_moment": round(self.ms / (self.moments or 1), 2),
            "reward_per_day": round(self.reward / days, 2),
            "lessons": self.lessons,
            "unsettled_lessons": self.unsettled,
            "agreement_with_trainer": round(self.agreed / self.watched, 3) if self.watched else None,
        }


@dataclass
class Life:
    """One companion with one patient, moment by moment."""

    world: World
    companion: Any
    trainer: Any = None
    stages: tuple = STAGES
    tallies: dict[str, Tally] = field(default_factory=dict)
    reward: float | None = None
    last: dict[str, Any] = field(default_factory=dict)
    override: float | None = None  # a body set from outside (the page), over the stages
    _run: int = 0  # consecutive minutes below 70, for episodes

    def __post_init__(self) -> None:
        self._lessons = random.Random(self.world.seed * 31 + 7)

    @property
    def stage(self) -> str:
        return stage_of(self.world.day, self.stages)[0]

    @property
    def apprentice(self) -> bool:
        return self.trainer is not None and hasattr(self.companion, "watch")

    def moment(self) -> tuple[Moment, dict[str, Any]]:
        stage, sensitivity = stage_of(self.world.day, self.stages)
        self.world.body.sensitivity = sensitivity if self.override is None else self.override
        t = self.tallies.setdefault(stage, Tally())
        if self.apprentice:
            decision = self.trainer.decide(self.world, self.reward)
            lesson = decision["alert"] or self._lessons.random() < QUIET_LESSONS
            seen = self.companion.watch(self.world, bool(decision["alert"]), lesson)
            decision = {**decision, **{k: seen[k] for k in ("sweeps", "learning_sweeps", "ms")},
                        "mode": "apprentice", "own": seen["own"], "lesson": lesson}
            t.watched += 1
            t.agreed += bool(seen["agree"])
            t.lessons += lesson
            t.unsettled += seen["unsettled"]
        else:
            decision = self.companion.decide(self.world, self.reward)
        m = self.world.advance(bool(decision["alert"]))
        self.reward = None if self.apprentice else m.reward
        self.last = decision
        t.moments += 1
        t.below += m.below
        t.severe += m.severe
        t.in_range += 70 <= m.glucose <= 180
        t.alerts += m.alert
        t.nuisance += m.nuisance
        t.sweeps += decision.get("sweeps", 0)
        t.learning_sweeps += decision.get("learning_sweeps", 0)
        t.aroused += decision.get("mode") == "aroused"
        t.ms += decision.get("ms", 0.0)
        t.reward += m.reward
        if m.below:
            if self._run < 15 <= self._run + m.below:
                t.episodes += 1
                t.night_episodes += m.asleep
            self._run += m.below
        else:
            self._run = 0
        return m, decision

    def run(self, days: float, every: Callable[["Life", Moment, dict], None] | None = None) -> None:
        for _ in range(int(days * DAY / STEP)):
            m, d = self.moment()
            if every:
                every(self, m, d)

    def report(self) -> dict[str, dict[str, float | None]]:
        return {name: self.tallies[name].report() for name, *_ in self.stages if name in self.tallies}
