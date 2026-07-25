---
title: "MLOps Basics"
stage: 6
order: 31
minutes: 45
difficulty: intermediate
prerequisites: ["first-ml-project"]
tags: ["mlops", "experiment-tracking", "drift"]
summary: "Experiment tracking, model and data versioning, reproducibility, and drift monitoring — what keeps a shipped model trustworthy after it leaves your notebook."
---

# MLOps Basics

## Why this matters

Lesson 17 took one project from a raw question to a written-up result: a
baseline, a few compared models, a chosen metric, a final number. That
pipeline is real, but it's also a single run, on a laptop, that lives in
your head and a notebook. The moment you try more than one variant
seriously — a different `class_weight`, a different model, a reworked
feature — and want to know *which run actually produced which number*,
"I remember it was the second one" stops being good enough. And the
moment a model like that goes anywhere near production, a second problem
appears that lesson 17 never had to face: the world it was trained on
keeps moving after training stops. This lesson is the layer between "a
model that worked once" and "a model you can trust, rerun, and know when
to retrain" — the practices that turn Stage 3's one-off pipeline into
something an actual team could run for a year.

## The concept

**Experiment tracking** is the direct fix for "which run was that." Every
time you train a variant, record what went in (hyperparameters, which
features, which data) and what came out (metrics) as a durable record,
not a scrollback buffer. The minimum viable version is a line of JSON
appended to a file per run — exactly what "In code" builds below. Once a
team outgrows that, dedicated tools (MLflow, Weights & Biases) do the same
job with a UI, artifact storage, and richer comparison — same idea, more
infrastructure. Either way, the point is identical: never trust memory
over a written record when comparing runs.

**Experiment tracking is not a model registry, and neither is
monitoring — three different jobs.** Tracking records *every* run you
ever try, including the ones that were worse, because "worse" is only
knowable by comparison. A **model registry** is a much smaller,
curated list: the handful of models that graduated from "an experiment"
to "a versioned candidate for deployment," each with a version number and
a status like staging or production. **Monitoring** is different again —
it's not about training runs at all, it's about watching a model that's
*already deployed* to see whether it's still behaving the way it did when
it was evaluated. Confusing these three is a common real mistake: logging
every experiment nowhere, or treating your only copy of a deployed model
as also being its full experiment history, or assuming that because a
model passed evaluation once, it needs no further watching. They're three
separate, complementary layers, and a real ML system needs all three.

**Model and data versioning.** A trained model is a function of three
things: the code, the hyperparameters, and the data it was fit on.
Change any one and you get a different model, even if you don't notice
immediately. Code versioning is git, already familiar. Hyperparameters
are what experiment tracking already logs. Data is the one teams skip
most often — retraining "the same script" against a table that has since
had rows added, fixed, or deleted silently produces a different model,
with no error and no warning. The fix is to pin what a given run actually
trained on: a copy, a hash (`hashlib.sha256` over the file is the DIY
version), or a dedicated tool like **DVC**, which versions large
data/model files alongside git commits the same way git versions code.

**Reproducibility** is the property that lets you rerun a logged
experiment and get the same answer back — and it doesn't happen for
free. Three concrete sources of drift between "I ran this Tuesday" and "I
ran this again Thursday": an unseeded random number generator (every
`random_state=` in this entire roadmap exists to prevent exactly this);
an unpinned dependency (a library's default behavior changes between
versions, silently, unless `requirements.txt` or a lockfile pins the
exact version, not just the name); and simply not knowing which code
version produced a result (a git commit SHA logged alongside the run
answers "which version of my code was this," the same way experiment
tracking answers "which hyperparameters was this").

**Monitoring for drift.** A deployed model doesn't crash when the world
changes around it — it just keeps returning confident predictions that
quietly get worse, because nothing about *how* it produces output changes
when its assumptions stop holding. **Data drift** (also called covariate
shift) is when the distribution of the *input features* your model sees
in production shifts away from what it was trained on — a pricing change
that pushes everyone's `monthly_charges` up, a new signup channel that
brings in a different mix of customers. **Concept drift** is subtler: the
input distribution can look the same while the *relationship* between
features and label changes underneath it — the same `tenure_months` and
`contract_type` used to predict low churn risk, but after a competitor's
price cut, they don't anymore. Both are silent by default. Statistical
tests over feature distributions (the Kolmogorov-Smirnov test used below
is one of the simplest) catch data drift automatically, without needing
ground-truth labels. Concept drift is harder to catch this way, since the
inputs alone don't reveal it — the most direct check is comparing
predictions to real outcomes once they're known, which is often the
entire reason a monitoring system needs to wait and re-check later
rather than judging a model once at deployment and considering it done.

## In code

**Part 1 — a lightweight experiment logger, run against three real model
variants.** No dependency beyond the standard library's `json` — this is
the idea MLflow and W&B build on, in its simplest possible form: append
one fact per run to a file.

```python
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

LOG_PATH = Path("experiments.jsonl")
if LOG_PATH.exists():
    LOG_PATH.unlink()


def log_run(run_name, params, metrics):
    """Append one experiment run as a line of JSON: easy to append to,
    easy to grep, easy to load back with pandas -- the whole idea of
    experiment tracking, minus the UI."""
    record = {
        "run_name": run_name,
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params,
        "metrics": metrics,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


# Same shape of churn problem as lesson 17, regenerated here so this
# lesson is self-contained -- the exercise has you point this same
# logger at your actual lesson 17 pipeline instead.
rng = np.random.default_rng(42)
n = 400
contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
tenure_months = np.clip(rng.exponential(24, n), 0, 72).round(0)
monthly_charges = np.clip(rng.normal(70, 25, n), 20, 150).round(2)
contract_risk = np.select(
    [contract_type == "Month-to-month", contract_type == "One year", contract_type == "Two year"],
    [0.9, 0.0, -0.5])
churn_logit = -1.3 + contract_risk - 0.02 * tenure_months + 0.006 * (monthly_charges - 70) + rng.normal(0, 0.3, n)
churned = (rng.random(n) < 1 / (1 + np.exp(-churn_logit))).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure_months, "contract_type": contract_type,
    "monthly_charges": monthly_charges, "churned": churned,
})
feature_df = pd.get_dummies(df.drop(columns=["churned"]), columns=["contract_type"], drop_first=True)
X = feature_df.values.astype(float)
y = df["churned"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Three candidate runs -- each one gets its params and metrics logged
# instead of just printed and forgotten the moment the next cell runs.
runs = [
    ("logreg_default", LogisticRegression(max_iter=2000), X_train_scaled,
     {"model": "LogisticRegression", "class_weight": "none", "seed": 42}),
    ("logreg_balanced", LogisticRegression(max_iter=2000, class_weight="balanced"), X_train_scaled,
     {"model": "LogisticRegression", "class_weight": "balanced", "seed": 42}),
    ("random_forest", RandomForestClassifier(n_estimators=300, random_state=42), X_train,
     {"model": "RandomForestClassifier", "n_estimators": 300, "seed": 42}),
]

for run_name, model, X_fit, params in runs:
    acc = cross_val_score(model, X_fit, y_train, cv=cv, scoring="accuracy").mean()
    auc = cross_val_score(model, X_fit, y_train, cv=cv, scoring="roc_auc").mean()
    metrics = {"cv_accuracy": round(float(acc), 4), "cv_roc_auc": round(float(auc), 4)}
    log_run(run_name, params, metrics)
    print(f"logged {run_name}: {metrics}")

print(f"\n--- {LOG_PATH} now contains ---")
print(LOG_PATH.read_text())
```

```
logged logreg_default: {'cv_accuracy': 0.7812, 'cv_roc_auc': 0.6678}
logged logreg_balanced: {'cv_accuracy': 0.6312, 'cv_roc_auc': 0.6643}
logged random_forest: {'cv_accuracy': 0.7031, 'cv_roc_auc': 0.4724}

--- experiments.jsonl now contains ---
{"run_name": "logreg_default", "logged_at": "2026-07-25T14:14:29", "params": {"model": "LogisticRegression", "class_weight": "none", "seed": 42}, "metrics": {"cv_accuracy": 0.7812, "cv_roc_auc": 0.6678}}
{"run_name": "logreg_balanced", "logged_at": "2026-07-25T14:14:29", "params": {"model": "LogisticRegression", "class_weight": "balanced", "seed": 42}, "metrics": {"cv_accuracy": 0.6312, "cv_roc_auc": 0.6643}}
{"run_name": "random_forest", "logged_at": "2026-07-25T14:14:30", "params": {"model": "RandomForestClassifier", "n_estimators": 300, "seed": 42}, "metrics": {"cv_accuracy": 0.7031, "cv_roc_auc": 0.4724}}
```

That's the entire mechanism. `cv_roc_auc` differs meaningfully between the
three runs, and now that difference is a permanent, timestamped fact on
disk instead of something you'd have to rerun everything to recover.

**Part 2 — loading the log back and comparing runs**, continuing right
after Part 1 in the same session:

```python
records = [json.loads(line) for line in LOG_PATH.read_text().splitlines()]
rows = [{"run_name": r["run_name"], **r["params"], **r["metrics"]} for r in records]
runs_df = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
print(runs_df[["run_name", "model", "cv_accuracy", "cv_roc_auc"]].to_string(index=False))
print(f"\nBest run by cv_roc_auc: {runs_df.iloc[0]['run_name']}")
```

```
       run_name                  model  cv_accuracy  cv_roc_auc
 logreg_default     LogisticRegression       0.7812      0.6678
logreg_balanced     LogisticRegression       0.6312      0.6643
  random_forest RandomForestClassifier       0.7031      0.4724

Best run by cv_roc_auc: logreg_default
```

This is the entire point of tracking made concrete: `random_forest` looks
respectable on accuracy alone (70.3%) but has the worst `cv_roc_auc` of
the three (0.4724 — barely better than random) — exactly the accuracy-can-
mislead trap from lesson 13, now caught by a comparison table instead of
a single glance at one run's printout.

**Part 3 — reproducibility, demonstrated rather than asserted:**

```python
rep_rng = np.random.default_rng(0)
Xr = rep_rng.normal(size=(300, 4))
yr = (Xr[:, 0] + 0.5 * Xr[:, 1] + rep_rng.normal(scale=1.5, size=300) > 0).astype(int)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=1)


def fit_and_summarize(random_state):
    m = RandomForestClassifier(n_estimators=50, random_state=random_state)
    m.fit(Xr_train, yr_train)
    return m.score(Xr_test, yr_test), m.feature_importances_.round(4).tolist()


print("Same seed (random_state=42), run A vs run B:")
acc_a, imp_a = fit_and_summarize(random_state=42)
acc_b, imp_b = fit_and_summarize(random_state=42)
print(f"  run A: accuracy={acc_a:.4f}  feature_importances={imp_a}")
print(f"  run B: accuracy={acc_b:.4f}  feature_importances={imp_b}")
print(f"  identical: {acc_a == acc_b and imp_a == imp_b}")

print("\nNo fixed seed (random_state=None), run C vs run D:")
acc_c, imp_c = fit_and_summarize(random_state=None)
acc_d, imp_d = fit_and_summarize(random_state=None)
print(f"  run C: accuracy={acc_c:.4f}  feature_importances={imp_c}")
print(f"  run D: accuracy={acc_d:.4f}  feature_importances={imp_d}")
print(f"  identical: {acc_c == acc_d and imp_c == imp_d}")
```

```
Same seed (random_state=42), run A vs run B:
  run A: accuracy=0.7000  feature_importances=[0.3527, 0.2217, 0.1985, 0.2271]
  run B: accuracy=0.7000  feature_importances=[0.3527, 0.2217, 0.1985, 0.2271]
  identical: True

No fixed seed (random_state=None), run C vs run D:
  run C: accuracy=0.7000  feature_importances=[0.365, 0.2215, 0.1888, 0.2247]
  run D: accuracy=0.7000  feature_importances=[0.353, 0.2214, 0.1877, 0.2379]
  identical: False
```

Same code, same data, two different outcomes depending on one argument.
`random_state=42` gives byte-for-byte identical feature importances on
both runs; `random_state=None` gives two runs that happen to land on the
same rounded accuracy but genuinely different importances underneath —
a believable trap if you only glance at the headline metric. This is
exactly what "pin your seeds" in *The concept* is protecting you from.

**Part 4 — a real drift check with `scipy.stats.ks_2samp`,** the
two-sample Kolmogorov-Smirnov test: are two samples plausibly drawn from
the same distribution?

```python
from scipy.stats import ks_2samp

drift_rng = np.random.default_rng(7)
training_charges = drift_rng.normal(loc=70, scale=25, size=1000)

# "today's" traffic, still genuinely the same distribution -- no drift
fresh_batch = drift_rng.normal(loc=70, scale=25, size=300)

# a later batch where the underlying population actually shifted --
# e.g. a pricing change pushed everyone's plan up
drifted_batch = drift_rng.normal(loc=95, scale=25, size=300)


def check_drift(reference, current, feature_name, alpha=0.05):
    statistic, p_value = ks_2samp(reference, current)
    drifted = p_value < alpha
    verdict = "DRIFT DETECTED" if drifted else "no drift"
    print(f"{feature_name}: KS statistic={statistic:.4f}  p-value={p_value:.3e}  -> {verdict}")
    return drifted


print(f"training mean={training_charges.mean():.2f}  fresh mean={fresh_batch.mean():.2f}  drifted mean={drifted_batch.mean():.2f}\n")
check_drift(training_charges, fresh_batch, "monthly_charges (fresh batch)")
check_drift(training_charges, drifted_batch, "monthly_charges (drifted batch)")
```

```
training mean=68.19  fresh mean=69.58  drifted mean=95.80

monthly_charges (fresh batch): KS statistic=0.0480  p-value=6.449e-01  -> no drift
monthly_charges (drifted batch): KS statistic=0.4173  p-value=1.265e-36  -> DRIFT DETECTED
```

A p-value of 0.64 says the fresh batch is entirely consistent with random
sampling noise around the training distribution — correctly, since it was
drawn from the identical distribution. A p-value of 1.3 × 10⁻³⁶ says the
drifted batch essentially could not have come from that same
distribution — correctly again, since its mean is a full standard
deviation higher. This is exactly the kind of check a monitoring job runs
on a schedule against fresh production data, with no labels required,
catching data drift before anyone notices the model quietly getting
worse.

## Build this

Add experiment tracking to your lesson 17 project. Using the `log_run`
pattern above, log every model variant you compared there (at minimum the
baseline and the two-plus real models) with its hyperparameters and
cross-validated metrics into a JSON-lines file. Then write a short script
that loads the log back with pandas and prints a comparison table sorted
by your chosen metric, the way Part 2 does — confirm it correctly
identifies the model you actually picked in lesson 17's write-up.

**Stretch:** add a real drift check to the same project. Using
`ks_2samp` from Part 4, compare the distribution of one numeric feature
(e.g. `monthly_charges`) between your train and test splits — it should
report no drift, since it's a random split of the same data. Then build
one artificially drifted batch (shift the mean, or resample from a
different subgroup) and confirm your check correctly flags it. Write down
both p-values.

## Go deeper

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — the most widely used open-source experiment tracking and model registry tool; a natural next step past the JSON-lines logger above.
- [DVC Documentation](https://doc.dvc.org/) — Data Version Control, for versioning datasets and model artifacts alongside git commits.
- [Weights & Biases Documentation](https://docs.wandb.ai/) — a hosted alternative to MLflow with a strong comparison-dashboard UI.
- [Google: Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — battle-tested, practical engineering rules for production ML, including several on monitoring and gradual rollout.
- [Chip Huyen: MLOps Guide](https://huyenchip.com/mlops/) — a free, structured path through MLOps topics at increasing depth, from someone who has written extensively on designing ML systems.

**Next:** [Deploying a Model](32-deploying-models.md)
