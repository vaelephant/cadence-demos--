"""A virtual type 1 diabetic patient: the UVA/Padova 2008 model as implemented in simglucose.

The equations and the thirty virtual patients' parameters are those of simglucose 0.2.11
(Jinyu Xie, MIT licence, ``data/LICENSE-simglucose``). simglucose integrates them with scipy's
dopri5 once per simulated minute; here a fixed one-minute fourth-order Runge-Kutta step in
plain Python does the same work about thirty times faster (``tests`` compare the two).

Two modifiers are this demo's own and are not part of the published model:

``sensitivity`` multiplies the insulin-dependent glucose utilisation (``Vmx``) and the insulin
suppression of hepatic production (``kp3``): 1.0 is the published patient, 1.4 a body that
has become 40% more sensitive to insulin.

``exercise`` adds insulin-independent utilisation (``Fsnc``) while it lasts, and
``afterglow`` raises ``sensitivity`` for hours afterwards, the delayed lows that follow
exercise. Both are illustrative, not fitted physiology.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / "data"
STATES = 13
EAT_RATE = 5.0  # g/min, as in simglucose


def _rows(name: str) -> dict[str, dict[str, str]]:
    with open(DATA / name, newline="") as handle:
        return {row["Name"]: row for row in csv.DictReader(handle)}


def names() -> list[str]:
    return list(_rows("vpatient_params.csv"))


@dataclass
class Params:
    name: str
    values: dict[str, float]
    x0: list[float]
    carb_ratio: float  # g of carbohydrate covered by one unit of insulin
    correction: float  # mg/dL lowered by one unit
    total_daily: float  # units per day

    @property
    def basal(self) -> float:
        """The steady basal infusion, U/min."""
        return self.values["u2ss"] * self.values["BW"] / 6000.0

    @classmethod
    def named(cls, name: str) -> "Params":
        row = _rows("vpatient_params.csv")[name]
        quest = _rows("Quest.csv")[name]
        x0 = [float(row[f"x0_{i:2d}"]) for i in range(1, STATES + 1)]
        skip = {"Name", "i", "patient_history"} | {f"x0_{i:2d}" for i in range(1, STATES + 1)}
        values = {k: float(v) for k, v in row.items() if k not in skip}
        return cls(name, values, x0, float(quest["CR"]), float(quest["CF"]), float(quest["TDI"]))


@dataclass
class Patient:
    """One patient's body, advanced one minute at a time."""

    params: Params
    sensitivity: float = 1.0
    exercising: bool = False
    afterglow: float = 0.0  # extra sensitivity after exercise, decays with a 4 h time constant
    x: list[float] = field(default_factory=list)
    minute: int = 0
    planned: float = 0.0  # g still to be eaten
    eating: bool = False
    last_cho: float = 0.0
    last_qsto: float = 0.0
    foodtaken: float = 0.0

    def __post_init__(self) -> None:
        if not self.x:
            self.x = list(self.params.x0)
        self.last_qsto = self.x[0] + self.x[1]

    @property
    def glucose(self) -> float:
        """Subcutaneous glucose, mg/dL: what a sensor reads, before sensor noise."""
        return self.x[12] / self.params.values["Vg"]

    def eat(self, grams: float) -> None:
        self.planned += max(0.0, grams)

    def step(self, insulin: float) -> None:
        """One minute at ``insulin`` U/min (basal plus any bolus given this minute)."""
        to_eat = min(EAT_RATE, self.planned) if self.planned > 0 else 0.0
        self.planned = max(0.0, self.planned - to_eat)
        if to_eat > 0 and self.last_cho <= 0:
            self.last_qsto = self.x[0] + self.x[1]
            self.foodtaken = 0.0
            self.eating = True
        if self.eating:
            self.foodtaken += to_eat
        if to_eat <= 0 and self.last_cho > 0:
            self.eating = False
        self.last_cho = to_eat
        p = self.params.values
        s = self.sensitivity + self.afterglow
        extra = 1.5 if self.exercising else 0.0
        dbar = self.last_qsto + self.foodtaken * 1000.0
        f = lambda x: _model(x, to_eat, insulin, p, dbar, s, extra)
        x = self.x
        k1 = f(x)
        k2 = f([a + 0.5 * b for a, b in zip(x, k1)])
        k3 = f([a + 0.5 * b for a, b in zip(x, k2)])
        k4 = f([a + b for a, b in zip(x, k3)])
        self.x = [a + (b + 2 * c + 2 * d + e) / 6.0 for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
        self.afterglow *= math.exp(-1.0 / 240.0)
        self.minute += 1


def _model(x, cho, insulin, p, dbar, sensitivity, exercise):
    """The UVA/Padova 2008 right-hand side of simglucose's ``T1DPatient.model``."""
    d = cho * 1000.0
    u = insulin * 6000.0 / p["BW"]
    qsto = x[0] + x[1]
    if dbar > 0:
        aa = 5.0 / (2.0 * dbar * (1.0 - p["b"]))
        cc = 5.0 / (2.0 * dbar * p["d"])
        kgut = p["kmin"] + (p["kmax"] - p["kmin"]) / 2.0 * (
            math.tanh(aa * (qsto - p["b"] * dbar)) - math.tanh(cc * (qsto - p["d"] * dbar)) + 2.0
        )
    else:
        kgut = p["kmax"]
    dx0 = -p["kmax"] * x[0] + d
    dx1 = p["kmax"] * x[0] - x[1] * kgut
    dx2 = kgut * x[1] - p["kabs"] * x[2]
    rat = p["f"] * p["kabs"] * x[2] / p["BW"]
    egp = p["kp1"] - p["kp2"] * x[3] - sensitivity * p["kp3"] * x[8]
    uii = p["Fsnc"] + exercise
    et = p["ke1"] * (x[3] - p["ke2"]) if x[3] > p["ke2"] else 0.0
    dx3 = max(egp, 0.0) + rat - uii - et - p["k1"] * x[3] + p["k2"] * x[4]
    dx3 = dx3 if x[3] >= 0 else 0.0
    vmt = p["Vm0"] + sensitivity * p["Vmx"] * x[6]
    uid = vmt * x[4] / (p["Km0"] + x[4])
    dx4 = -uid + p["k1"] * x[3] - p["k2"] * x[4]
    dx4 = dx4 if x[4] >= 0 else 0.0
    dx5 = -(p["m2"] + p["m4"]) * x[5] + p["m1"] * x[9] + p["ka1"] * x[10] + p["ka2"] * x[11]
    it = x[5] / p["Vi"]
    dx5 = dx5 if x[5] >= 0 else 0.0
    dx6 = -p["p2u"] * x[6] + p["p2u"] * (it - p["Ib"])
    dx7 = -p["ki"] * (x[7] - it)
    dx8 = -p["ki"] * (x[8] - x[7])
    dx9 = -(p["m1"] + p["m30"]) * x[9] + p["m2"] * x[5]
    dx9 = dx9 if x[9] >= 0 else 0.0
    dx10 = u - (p["ka1"] + p["kd"]) * x[10]
    dx10 = dx10 if x[10] >= 0 else 0.0
    dx11 = p["kd"] * x[10] - p["ka2"] * x[11]
    dx11 = dx11 if x[11] >= 0 else 0.0
    dx12 = -p["ksc"] * x[12] + p["ksc"] * x[3]
    dx12 = dx12 if x[12] >= 0 else 0.0
    return [dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12]
