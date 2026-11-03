"""The kennel: where a newborn dog is trained before it is placed with its patient.

A newborn brain lives six days with each of ten virtual patients while the trainer (the
monitor's trend rule) is at the controls. Every trainer alert and one in twenty of its
quiet moments are the dog's lessons (``life.QUIET_LESSONS``). The patient a dog is later
placed with is never one of the kennel's, so what the dog brings home is general knowledge
of lows, not knowledge of that patient.

On leaving, the dog grows up (``companions.ADULT``) and its whole brain is saved. Its first
day at home is its youth (``FOUNDER["arousal"]["youth"]``): no moment of the kennel used
``Brain.live``, so the arousal's age starts at placement.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from . import companions as C
from .life import Life
from .world import World

PATIENTS = (
    "adolescent#002", "adolescent#004", "adolescent#006",
    "adult#002", "adult#003", "adult#005", "adult#006", "adult#008",
    "child#002", "child#004",
)
DAYS_EACH = 6
BRAINS = Path(__file__).resolve().parent.parent / "brains"


def train(seed: int = 0, genes: dict[str, Any] | None = None) -> tuple[C.Dog, dict[str, Any]]:
    dog = C.Dog(C.compose(seed, genes))
    report: dict[str, Any] = {}
    with warnings.catch_warnings():
        # a lesson whose free phase ran out of budget is counted (``unsettled_lessons``)
        warnings.filterwarnings("ignore", message="finite teaching free phase")
        for i, patient in enumerate(PATIENTS):
            life = Life(World(patient, seed=1000 * (seed + 1) + i), dog, trainer=C.Rule(),
                        stages=(("kennel", 0, DAYS_EACH, 1.0),))
            life.run(DAYS_EACH)
            report[patient] = life.report()["kennel"]
    dog.grow_up((genes or {}).get("adult"))
    return dog, report


def path_for(seed: int, label: str = "founder") -> Path:
    return BRAINS / f"kennel-{label}-s{seed}.npz"


def trained(seed: int = 0, genes: dict[str, Any] | None = None, label: str = "founder") -> C.Dog:
    """The kennel's dog for this seed, trained once and then loaded from ``brains/``."""
    path = path_for(seed, label)
    if not path.exists():
        dog, report = train(seed, genes)
        path.parent.mkdir(exist_ok=True)
        dog.save(path)
        path.with_suffix(".json").write_text(json.dumps(report, indent=1))
    return C.Dog.load(path)
