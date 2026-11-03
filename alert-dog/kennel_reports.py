"""Write the kennel report beside each saved brain that lacks one, retraining to recover it.

The kennel is deterministic for a seed, so the retrained brain must equal the saved one;
a report is written only when every parameter matches.

    python kennel_reports.py 10 11 12
"""

from __future__ import annotations

import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from alertdog import kennel


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype == object or b.dtype == object:
        return repr(a.tolist()) == repr(b.tolist())
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def recover(seed: int) -> str:
    path = kennel.path_for(seed)
    dog, report = kennel.train(seed)
    with tempfile.TemporaryDirectory() as tmp:
        dog.save(Path(tmp) / "fresh")
        with np.load(path, allow_pickle=True) as a, np.load(Path(tmp) / "fresh.npz", allow_pickle=True) as b:
            same = set(a.files) == set(b.files) and all(_equal(a[k], b[k]) for k in a.files)
    if not same:
        return f"s{seed}: retrained brain differs from {path.name}; no report written"
    path.with_suffix(".json").write_text(json.dumps(report, indent=1))
    lessons = sum(r["lessons"] for r in report.values())
    unsettled = sum(r["unsettled_lessons"] for r in report.values())
    return f"s{seed}: {lessons} lessons, {unsettled} unsettled"


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or list(range(10, 20))
    with ProcessPoolExecutor(4) as pool:
        for line in pool.map(recover, seeds):
            print(line)
