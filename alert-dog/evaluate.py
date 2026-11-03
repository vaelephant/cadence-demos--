"""Measure the alert dog against its controls over whole lives at home.

Every companion lives the same staged life (``alertdog.life.STAGES``) with the same patient
and the same seeds; only the companion differs. Because an alert changes the patient's day,
the lives diverge after the first alert, as two households would. The kennel's dogs are
trained first (``alertdog.kennel``), one per seed, and saved under ``brains/``.

    python evaluate.py                      # adult#001, seeds 10-19, every companion
    python evaluate.py --patients adult#001 adolescent#001 child#001 --seeds 10 11 12

Writes ``evidence/<label>.json`` with the configuration, the per-stage measures of every
life and their means, and prints a table per stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cadence
import numpy as np

from alertdog import companions as C
from alertdog import kennel
from alertdog.life import QUIET_LESSONS, STAGES, Life
from alertdog.world import World

COMPANIONS = ("dog", "rule", "none", "random", "wild", "frozen", "always")


def genes_with(patch: dict | None) -> dict:
    genes = copy.deepcopy(C.FOUNDER)
    patch = dict(patch or {})
    genes["arousal"].update(patch.pop("arousal", {}))
    genes.update(patch)
    return genes


def raise_dog(job: tuple[int, dict | None, str]) -> str:
    seed, patch, label = job
    kennel.trained(seed, genes_with(patch), label)
    return str(kennel.path_for(seed, label))


def one(job: tuple[str, str, int, dict | None, str]) -> dict:
    kind, patient, seed, patch, label = job
    world = World(patient, seed=seed)
    began = time.process_time()
    days = STAGES[-1][2]
    if kind == "dog":
        companion = kennel.trained(seed, genes_with(patch), label)
    elif kind == "frozen":
        companion = C.Frozen(kennel.trained(seed, genes_with(patch), label).brain)
    elif kind == "always":
        companion = C.Always(kennel.trained(seed, genes_with(patch), label).brain)
    elif kind == "wild":
        companion = C.Dog(C.compose(seed, genes_with(patch)))
    else:
        companion = {"rule": C.Rule(), "none": C.Never(), "random": C.Random(0.02, seed)}[kind]
    life = Life(world, companion)
    life.run(days)
    extra = companion.describe() if hasattr(companion, "describe") else {}
    extra["refusals"] = getattr(companion, "refusals", 0)
    return {"companion": kind, "patient": patient, "seed": seed, "stages": life.report(),
            "cpu_seconds": round(time.process_time() - began, 1), **extra}


def mean(lives: list[dict], stage: str, key: str) -> float:
    values = [life["stages"][stage][key] for life in lives
              if stage in life["stages"] and life["stages"][stage][key] is not None]
    return round(float(np.mean(values)), 3) if values else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", default=["adult#001"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)),
                    help="life seeds; 0-2 were used to choose the design, so reports use others")
    ap.add_argument("--companions", nargs="+", default=list(COMPANIONS))
    ap.add_argument("--genes", type=json.loads, default=None, help="JSON patch over the founder genes")
    ap.add_argument("--label", default="lives")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    brains = "founder" if args.genes is None else args.label
    began = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        if {"dog", "frozen", "always"} & set(args.companions):
            list(pool.map(raise_dog, [(s, args.genes, brains) for s in args.seeds]))
        kennel_seconds = round(time.time() - began, 1)
        jobs = [(k, p, s, args.genes, brains) for p in args.patients for s in args.seeds for k in args.companions]
        lives = list(pool.map(one, jobs))
    keys = ("below_70_pct", "below_54_pct", "lows_per_day", "night_lows_per_day", "in_range_pct",
            "alerts_per_day", "nuisance_share", "aroused_share", "learning_sweeps_per_moment")
    summary: dict[str, dict] = {}
    for name, *_ in STAGES:
        print(f"\n{name}")
        print("  " + "companion".ljust(10) + "".join(k[:14].rjust(15) for k in keys))
        for kind in args.companions:
            mine = [life for life in lives if life["companion"] == kind]
            row = {k: mean(mine, name, k) for k in keys}
            summary.setdefault(name, {})[kind] = row
            print("  " + kind.ljust(10) + "".join(f"{row[k]:15.2f}" for k in keys))
    out = Path(__file__).parent / "evidence" / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "question": "Does one continuing Cadence brain, trained in a kennel and then living with one "
                    "virtual patient it has never met, keep that patient out of lows, adapt when the "
                    "body changes and hold up when it changes back?",
        "configuration": {"patients": args.patients, "seeds": args.seeds, "stages": STAGES,
                          "genes": genes_with(args.genes), "adult": C.ADULT, "inputs": C.INPUTS,
                          "kennel": {"patients": kennel.PATIENTS, "days_each": kennel.DAYS_EACH,
                                     "quiet_lessons": QUIET_LESSONS, "brains": brains}},
        "environment": {"cadence": cadence.__version__, "python": platform.python_version(),
                        "numpy": np.__version__, "machine": platform.machine()},
        "kennel_wall_seconds": kennel_seconds,
        "wall_seconds": round(time.time() - began, 1),
        "means": summary,
        "lives": lives,
    }, indent=1, default=list))
    print(f"\nwrote {out} in {time.time() - began:.0f}s")


if __name__ == "__main__":
    main()
