import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alertdog import companions as C  # noqa: E402
from alertdog.life import Life  # noqa: E402
from alertdog.patient import Params, Patient, names  # noqa: E402
from alertdog.world import World  # noqa: E402


def hours(p: Patient, h: float, insulin: float | None = None) -> None:
    for _ in range(int(h * 60)):
        p.step(p.params.basal if insulin is None else insulin)


def test_thirty_patients_rest_near_their_basal_glucose():
    assert len(names()) == 30
    for name in ("adolescent#001", "adult#001", "child#001"):
        p = Patient(Params.named(name))
        start = p.glucose
        hours(p, 6)
        assert abs(p.glucose - start) < 25


def test_meal_raises_and_bolus_lowers():
    fed, dosed = Patient(Params.named("adult#001")), Patient(Params.named("adult#001"))
    fed.eat(60)
    hours(fed, 1.5)
    dosed.step(dosed.params.basal + 6.0)
    hours(dosed, 2)
    rest = Patient(Params.named("adult#001"))
    hours(rest, 2)
    assert fed.glucose > rest.glucose + 30
    assert dosed.glucose < rest.glucose - 15


def test_more_sensitive_body_goes_lower_on_the_same_dose():
    normal, sensitive = Patient(Params.named("adult#001")), Patient(Params.named("adult#001"), sensitivity=1.3)
    for p in (normal, sensitive):
        p.step(p.params.basal + 4.0)
        hours(p, 3)
    assert sensitive.glucose < normal.glucose - 10


def test_same_seed_and_decisions_give_the_same_life():
    a, b = World("adult#001", seed=3), World("adult#001", seed=3)
    for i in range(288):
        alert = i % 50 == 0
        a.advance(alert), b.advance(alert)
    assert a.history == b.history


def test_an_alert_treats_a_low_and_bothers_a_normal_glucose():
    w = World("adult#001", seed=0)
    w.history[-1] = 80.0
    m = w.advance(True)
    assert m.treated == 15.0 and not m.nuisance
    w = World("adult#001", seed=0)
    w.history[-1] = 160.0
    m = w.advance(True)
    assert m.treated == 0.0 and m.nuisance and m.reward < 0


def test_rule_alerts_on_a_falling_reading():
    w = World("adult#001", seed=0)
    w.history[-4:] = [120.0, 110.0, 100.0, 85.0]
    assert C.Rule().decide(w, None)["alert"]
    w.history[-4:] = [120.0, 120.0, 121.0, 120.0]
    assert not C.Rule().decide(w, None)["alert"]


def test_senses_have_the_declared_width_and_range():
    x = C.senses(World("adult#001", seed=0))
    assert x.shape == (1, C.INPUTS)
    assert np.all(x >= 0) and np.all(x <= 1)


def test_the_dog_continues_identically_after_save_and_load(tmp_path):
    life = Life(World("adult#001", seed=5), C.Dog(C.compose(0)))
    life.run(0.25)
    assert life.companion.brain.pending_feedback  # an issued alert decision awaits its outcome
    path = life.companion.save(tmp_path / "dog.npz")
    twin = Life(copy.deepcopy(life.world), C.Dog.load(path), reward=life.reward)
    for _ in range(60):
        a, _ = life.moment()
        b, _ = twin.moment()
        assert a.alert == b.alert
    assert life.world.history == twin.world.history


def test_apprenticeship_executes_the_trainers_decisions():
    w = World("adult#001", seed=2)
    life = Life(w, C.Dog(C.compose(0)), trainer=C.Rule(), stages=(("kennel", 0, 1, 1.0),))
    rule_world = World("adult#001", seed=2)
    rule = Life(rule_world, C.Rule(), stages=(("kennel", 0, 1, 1.0),))
    life.run(0.5)
    rule.run(0.5)
    assert w.history == rule_world.history
    assert life.tallies["kennel"].lessons > 0


@pytest.mark.parametrize("kind", ["frozen", "always"])
def test_controls_answer_from_a_brain(kind):
    brain = C.compose(0)
    companion = C.Frozen(brain) if kind == "frozen" else C.Always(brain)
    life = Life(World("adult#001", seed=1), companion)
    life.run(0.1)
    assert life.tallies["arrival"].moments == 28
