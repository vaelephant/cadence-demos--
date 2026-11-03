"""The patient's life in five-minute moments, and what an alert does in it.

Every five minutes the sensor reads the patient's glucose and the companion (a brain, a
rule or a control) decides: stay quiet or alert "going low". Then five minutes of life pass.

The patient lives a routine with this patient's own habits: three meals with a bolus each,
an afternoon snack, exercise on some evenings, sleep from 23:00 to 07:00. Bolus doses carry
everyday estimation error, and this patient over-doses the lunch bolus on some days.

An alert has a consequence. An awake patient checks the sensor and eats 15 g of fast
carbohydrate when the glucose is low or falling toward low; otherwise the alert is only an
interruption. At night an alert wakes the patient, who then does the same check. Without an
alert, an awake patient notices a low only below 60 mg/dL, and asleep only below 45.

What the companion is paid each moment is what the patient lived through: minutes below
70 mg/dL and, worse, below 54, and the interruption of every alert (0.2 by day, 0.5 at
night, against 1 for every five minutes below 70). Nobody tells it when
to alert; it is never shown the future.

The life is seeded: the same seed, companion actions and interventions give the same life.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .patient import Params, Patient

STEP = 5  # minutes per moment
DAY = 24 * 60
TREAT = 15.0  # g of fast carbohydrate
LOW, SEVERE = 70.0, 54.0


@dataclass(frozen=True)
class Habits:
    """What makes this patient's lows this patient's own."""

    exercise_days: tuple[int, ...] = (1, 3, 5)  # Tue, Thu, Sat
    exercise_at: int = 17 * 60 + 30
    exercise_minutes: int = 50
    afterglow: float = 0.7  # extra insulin sensitivity right after exercise
    lunch_overdose: float = 0.25  # share of days the lunch bolus is overestimated
    overdose: float = 1.6
    dose_error: float = 0.12  # lognormal spread of every bolus


@dataclass
class Moment:
    """What one five-minute moment held: the sensor, the events, the outcome."""

    minute: int
    glucose: float  # sensor reading at the start of the moment, mg/dL
    alert: bool = False
    treated: float = 0.0  # g of fast carbohydrate eaten in this moment
    meal: float = 0.0
    bolus: float = 0.0
    exercising: bool = False
    asleep: bool = False
    nuisance: bool = False  # alerted, checked, nothing to treat
    below: int = 0  # minutes below LOW during the moment
    severe: int = 0  # minutes below SEVERE
    reward: float = 0.0
    events: list[str] = field(default_factory=list)


class World:
    """One patient's continuing life."""

    def __init__(self, patient: str = "adult#001", seed: int = 0, habits: Habits | None = None):
        self.params = Params.named(patient)
        self.body = Patient(self.params)
        self.habits = habits or Habits()
        self.seed = seed
        self.minute = 0
        self.history: list[float] = []  # sensor readings, one per moment
        self.boluses: list[tuple[int, float]] = []
        self.meals: list[tuple[int, float]] = []
        self.exercise_end: int | None = None
        self.last_treat = -10**9
        self.pending: list[tuple[int, str, float]] = []  # (minute, kind, amount)
        self._plan_day = -1
        self._noise = random.Random(seed * 7919 + 1)
        self.history.append(self.reading())

    # -- the clock -------------------------------------------------------------------------

    @property
    def day(self) -> int:
        return self.minute // DAY

    @property
    def clock(self) -> int:
        return self.minute % DAY

    def asleep_at(self, minute: int) -> bool:
        c = minute % DAY
        return c >= 23 * 60 or c < 7 * 60

    @property
    def asleep(self) -> bool:
        return self.asleep_at(self.minute)

    # -- the body as the companion may read it ---------------------------------------------

    def reading(self) -> float:
        """The sensor: subcutaneous glucose with a little noise, clipped to a sensor's range."""
        return min(400.0, max(39.0, self.body.glucose + self._noise.gauss(0.0, 2.0)))

    def insulin_on_board(self) -> float:
        """Bolus units still acting, with a linear four-hour decay."""
        return sum(u * max(0.0, 1 - (self.minute - t) / 240) for t, u in self.boluses)

    def carbs_on_board(self) -> float:
        """Grams still being absorbed, with a linear three-hour decay."""
        return sum(g * max(0.0, 1 - (self.minute - t) / 180) for t, g in self.meals)

    def since_exercise(self) -> float | None:
        """Minutes since exercise ended; 0 while exercising; None if none in the last day."""
        if self.body.exercising:
            return 0.0
        if self.exercise_end is None or self.minute - self.exercise_end > DAY:
            return None
        return float(self.minute - self.exercise_end)

    # -- the routine -----------------------------------------------------------------------

    def _plan(self) -> None:
        """Schedule today's meals, boluses and exercise from the day's own seed."""
        day = self.day
        if day == self._plan_day:
            return
        self._plan_day = day
        rng = random.Random(self.seed * 100003 + day)
        base = day * DAY
        h = self.habits
        meals = [
            (7 * 60 + rng.randint(0, 60), rng.gauss(45, 8), False),
            (12 * 60 + rng.randint(0, 60), rng.gauss(65, 12), rng.random() < h.lunch_overdose),
            (18 * 60 + 30 + rng.randint(0, 60), rng.gauss(75, 12), False),
        ]
        for at, grams, over in meals:
            grams = max(15.0, grams)
            dose = grams / self.params.carb_ratio * math.exp(rng.gauss(0, h.dose_error))
            if over:
                dose *= h.overdose
            self.pending.append((base + at, "meal", round(grams, 1)))
            self.pending.append((base + at, "bolus", round(dose, 2)))
        if rng.random() < 0.4:
            self.pending.append((base + 15 * 60 + rng.randint(0, 60), "snack", round(rng.uniform(15, 25), 1)))
        if day % 7 in h.exercise_days:
            self.pending.append((base + h.exercise_at + rng.randint(-20, 20), "exercise", float(h.exercise_minutes)))

    def intervene(self, kind: str, amount: float = 0.0) -> None:
        """An event from outside the routine, this minute: a meal, a bolus, exercise."""
        self.pending.append((self.minute, kind, amount))

    # -- one moment ------------------------------------------------------------------------

    def advance(self, alert: bool) -> Moment:
        """Five minutes of life after the companion's decision; returns what they held."""
        self._plan()
        m = Moment(minute=self.minute, glucose=self.history[-1], alert=alert, asleep=self.asleep)
        if alert:
            self._check(m, woken=self.asleep)
        for _ in range(STEP):
            insulin = self.params.basal
            due = [e for e in self.pending if e[0] <= self.minute]
            self.pending = [e for e in self.pending if e[0] > self.minute]
            for _, kind, amount in due:
                if kind in ("meal", "snack"):
                    self.body.eat(amount)
                    self.meals.append((self.minute, amount))
                    m.meal += amount
                    m.events.append(kind)
                elif kind == "bolus":
                    insulin += amount  # the whole bolus within this minute
                    self.boluses.append((self.minute, amount))
                    m.bolus += amount
                    m.events.append("bolus")
                elif kind == "exercise":
                    self.body.exercising = True
                    self.pending.append((self.minute + int(amount), "rest", 0.0))
                    m.events.append("exercise")
                elif kind == "rest":
                    self.body.exercising = False
                    self.body.afterglow += self.habits.afterglow
                    self.exercise_end = self.minute
            self.body.step(insulin)
            self.minute += 1
            g = self.body.glucose
            m.below += g < LOW
            m.severe += g < SEVERE
            self._notice(m, g)
        m.exercising = self.body.exercising
        self.boluses = [(t, u) for t, u in self.boluses if self.minute - t < 240]
        self.meals = [(t, g) for t, g in self.meals if self.minute - t < 180]
        m.reward = -(m.below + 2.0 * m.severe) / STEP - (0.5 if m.asleep else 0.2) * alert
        self.history.append(self.reading())
        return m

    def _check(self, m: Moment, woken: bool) -> None:
        """The patient answers an alert: a look at the sensor, carbohydrate if it is low."""
        g, before = self.history[-1], self.history[-4] if len(self.history) >= 4 else self.history[-1]
        falling = (g - before) / 15.0  # mg/dL per minute over the last quarter hour
        if self.minute - self.last_treat >= 15 and (g < 90 or (g < 130 and falling < -1.0)):
            self._treat(m)
            m.events.append("treated on alert")
        else:
            m.nuisance = True
        m.events.append("woken" if woken else "checked")

    def _notice(self, m: Moment, g: float) -> None:
        """The patient's own symptoms: noticed below 60 awake, below 45 asleep."""
        limit = 45.0 if self.asleep else 60.0
        if g < limit and self.minute - self.last_treat >= 15 and self._noise.random() < 0.2:
            self._treat(m)
            m.events.append("felt low")

    def _treat(self, m: Moment) -> None:
        self.body.eat(TREAT)
        self.meals.append((self.minute, TREAT))
        self.last_treat = self.minute
        m.treated += TREAT
