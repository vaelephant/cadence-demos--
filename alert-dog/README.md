# Alert Dog: a Cadence brain lives with a virtual patient

Some people with type 1 diabetes live with a trained alert dog that notices a coming low
before they do. Here the dog is one continuing Cadence brain. It is trained in a kennel by
a trainer, then placed with a virtual patient it has never met. Every five minutes it reads
the patient's sensor and decides whether to stay quiet or alert "going low". After an
alert, the patient checks and eats 15 g of sugar if the glucose is low or falling toward
low. When nothing surprises it, the dog settles once and learns nothing. When an outcome
surprises it, it is aroused, and that moment samples, learns and writes memory.

The patient is the UVA/Padova 2008 model of type 1 diabetes, in the open-source
[simglucose](https://github.com/jxx123/simglucose) implementation. **This is a research
demo with a virtual patient. It is not a medical device and gives no treatment advice.**

Built with `cadence-net` 0.80.0.

```sh
cd alert-dog
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python server.py        # http://localhost:8672
.venv/bin/python evaluate.py      # the measurement below, about 6 minutes on 8 cores
.venv/bin/python -m pytest tests -q
```

## The page

Three households live the same day, with the same patient (`adult#001`), the same seeded
routine and the same interventions. Only the companion differs:

- **the Cadence alert dog**, kennel seed 10;
- **the monitor's trend rule**, which was also the dog's trainer;
- **no companion**, where the patient notices a low only below 60 mg/dL awake and
  45 mg/dL asleep.

An alert changes the patient's day, so the three curves part after the first alert.

The buttons act on all three households at once:

- **多打一针胰岛素** (an extra insulin dose): one meal's worth of bolus, with no meal.
- **去运动** (go exercise): 50 minutes of exercise, followed by a few hours of heightened
  insulin sensitivity.
- **加一顿饭** (add a meal): 60 g with its bolus.
- **变得对胰岛素更敏感** (become more insulin-sensitive): the body becomes 1.3 times as
  sensitive to insulin. **恢复原样** (back to normal) restores it.

The dog's card shows:

- whether this moment was calm (平静) or aroused (警觉);
- its two motor neurons, quiet and alert;
- a trace of arousal;
- its age, its share of aroused moments and the settling sweeps it spent.

The page also shows the measurement below, read from `evidence/`.

## What the dog is

The dog is one `Brain.compose` life: 36 sensory inputs, one 48-neuron cortex and two motor
neurons, with 4,384 parameters. Once placed, its only entry is `Brain.live`, which runs
with arousal and episodic memory. Each moment it senses population codes of:

- the reading;
- the 15-minute trend;
- insulin and carbohydrate on board;
- the time of day;
- the time since exercise;
- whether the patient is asleep;
- whether the patient was just treated.

It is paid what the patient lived through after its decision. Every five minutes below
70 mg/dL costs 1, and below 54 mg/dL costs 2 more (both divided by 5). Every alert costs 0.2
by day and 0.5 at night. Nobody tells it when to alert, and it is never shown the future.

**The kennel.** A newborn lives 6 days with each of 10 other virtual patients: 60 days in
all, with the trainer at the controls. The dog watches the trainer. Its lessons are every
trainer alert plus one in twenty of the trainer's quiet moments, taught with
`brain.stimulus` and `brain.learner.step`. That curriculum is supplied by hand, like a dog
trainer's. A seed's kennel gives about 1,100 lessons. A lesson whose free phase runs out of
budget before settling is counted in `brains/kennel-founder-s*.json`
(`unsettled_lessons`).

**Leaving the kennel.** The dog grows up (`ADULT` in `alertdog/companions.py`):

- no arousal heat;
- sampling temperature 0.3 → 0.15;
- actor step 0.03 → 0.003.

This mirrors the robot arena's ring stage. The step was chosen on development seeds 0–2,
between 0.03 and 0.003, so that learning at home refines the kennel's training instead of
overwriting it (`evidence/dev-kennel.json`, `evidence/dev-adult-eta.json`). The founder
genes come from the robot arena's nursery and were not tuned here.

**Memory.**

| Memory | Writes | Reads | Retention and reset |
| --- | --- | --- | --- |
| Weights | Kennel lessons; aroused moments at home | Every settle | Saved whole after the kennel (`brains/kennel-*.npz`). The page restarts from the saved brain whenever the server starts |
| Episodic records | Aroused moments | Every settle | Saved in the same file |
| Pending feedback | Every `live` decision | The next moment's reward | Kept until the outcome arrives |

## Controls

Every companion lives the same staged life, and each control is declared:

| Control | What it is | What it controls for |
| --- | --- | --- |
| `rule` | Alert below 75 mg/dL, or when the 15-minute trend projects below 70 within 20 minutes | The trend alarm of real monitors, and the dog's trainer |
| `none` | Never alerts | The patient on their own |
| `random` | Alerts 2% of moments | Alerting without looking |
| `wild` | The same brain without the kennel, learning from outcomes from birth | The kennel |
| `frozen` | The kennel dog answering greedily and never learning again | Learning at home |
| `always` | The kennel dog learning from every outcome through `Brain.step` | Learning only when aroused |

## The measurement

There is one patient the kennel never saw (`adult#001`) and a 49-day life in four stages:

1. **arrival**: days 0–7.
2. **routine**: days 7–21.
3. **changed**: days 21–35, with the body 1.3 times as sensitive to insulin and the doses
   unchanged.
4. **recovered**: days 35–49, back to normal.

The life seeds 10–19 were not used while designing. Seeds 0–2 were used to choose the
design. Every number below is the mean of 10 lives, from
`evidence/adult001-seeds10-19.json`.

Share of time below 70 mg/dL, in %:

| Companion | arrival | routine | changed | recovered |
| --- | ---: | ---: | ---: | ---: |
| dog | 0.24 | 0.33 | 2.25 | 0.59 |
| rule | 0.10 | 0.08 | 0.58 | 0.06 |
| none | 8.07 | 7.65 | 26.39 | 8.77 |
| random | 4.70 | 4.11 | 21.51 | 4.71 |
| wild | 1.64 | 2.10 | 12.29 | 3.20 |
| frozen | 0.95 | 0.97 | 3.53 | 0.99 |
| always | 0.19 | 0.30 | 1.16 | 0.17 |

Alerts per day, with the share that needed no sugar in brackets:

| Companion | routine | changed | recovered |
| --- | ---: | ---: | ---: |
| dog | 14.6 (64%) | 28.3 (63%) | 14.1 (62%) |
| rule | 2.9 (36%) | 12.5 (49%) | 3.2 (41%) |
| frozen | 13.2 (69%) | 29.4 (69%) | 12.9 (68%) |
| always | 23.2 (66%) | 33.8 (64%) | 21.6 (64%) |

What these numbers say:

- **The dog keeps an unseen patient out of most lows.** Without a companion, the patient
  spends 7.7% of routine time below 70 mg/dL and 26% once the body changes. With the dog,
  the figures are 0.33% and 2.25%. Without the kennel, the same brain learning from outcomes
  alone is 5 to 6 times worse (`wild`).
- **The dog does not beat its trainer.** The rule has fewer lows in every stage. It alerts
  a fifth as often in routine, and under half as often once the body changes. Close to two of every three of the dog's alerts need no sugar.
  The dog learned when lows come, but not to be as sparing as its trainer.
- **Learning at home: better on average, but not consistently per life.** On the mean, the
  learning dog has fewer lows than its frozen copy in every stage. Seed by seed the picture
  is mixed:
  - routine: lower in 1 life, tied in 5, higher in 4;
  - changed: lower in 5, tied in 2, higher in 3;
  - recovered: lower in 2, tied in 5, higher in 3.

  Most of the mean difference comes from seed 13, where the frozen dog spent 9.3% of
  routine time low. This run does not show that learning at home reliably helps.
- **Learning only when aroused is cheaper but not better.** In routine, the dog was aroused
  in 15% of moments and spent 1.15 learning sweeps per moment. Learning from every outcome
  (`always`) spent 4.8 learning sweeps per moment, had slightly fewer lows, and alerted
  more.
- **Retention after the change is partial.** Back to the normal body, the dog spends
  0.59% of the time low, against 0.33% before the change.

**Costs.** One 49-day life (14,112 moments) took 26 CPU seconds for the dog, 23 for the
frozen copy, 29 for `always` and 1.3 for the rule. That is about 1.8 ms per dog moment. The
ten kennels took 78 s wall time on 8 processes. In the seed-10 life, the dog settled for
434,000 sweeps, plus 28,000 learning sweeps.

**Kennel lessons.** Seeds 10–19 had 1,050–1,126 lessons each, 10,848 in all. One lesson
(seed 14) ran out of its free-phase budget before settling. It was learned anyway and is
counted as unsettled.

The `pass` of a test or a lower number in a table certifies only these declared tests on
virtual patients. It is not a claim about real patients or about monitors in use.

## What is borrowed and what is invented

- **The patient model** comes from simglucose 0.2.11 (MIT, Jinyu Xie; see
  `alertdog/data/LICENSE-simglucose`). The parameters of 30 virtual patients and their
  therapy settings are copied from it. The ODE is ported to plain Python with a one-minute
  RK4 step. When checked against simglucose during development, it agreed within 0.03 mg/dL.
- **Illustrative, not from the model:**
  - the insulin-sensitivity factor (a multiplier on insulin-dependent glucose use);
  - exercise (extra insulin-independent use, then a decaying afterglow);
  - the patient's habits;
  - the self-noticing thresholds;
  - the 15 g treatment rule.
- **Hand-set:** the kennel curriculum, the reward weights and the trend rule.

## Files

- `alertdog/patient.py`: the UVA/Padova model.
- `alertdog/world.py`: the patient's day and what an alert does.
- `alertdog/companions.py`: the dog and its controls.
- `alertdog/life.py`: stages and tallies.
- `alertdog/kennel.py`: the apprenticeship.
- `evaluate.py`: the measurement.
- `server.py` and `static/index.html`: the page.
- `kennel_reports.py`: rewrites a kennel's lesson report by retraining. Training is
  deterministic, and a report is written only if the retrained brain equals the saved one.
- `evidence/dev-*.json` and `spike*.json`: design runs on seeds 0–2, kept as the record of
  choices.
