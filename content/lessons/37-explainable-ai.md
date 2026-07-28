---
title: "Explainable AI — Opening the Black Box"
stage: 3
order: 37
minutes: 45
difficulty: intermediate
prerequisites: ["model-evaluation"]
tags: ["interpretability", "scikit-learn"]
summary: "Why a model made a prediction — feature importance, permutation importance, and reading a model you can't see inside."
---

# Explainable AI — Opening the Black Box

## Why this matters

A model that scores 96% accuracy and a model you can *explain* are two
different achievements. The first tells you it works on your test set.
The second tells you *why* — and why is what you need when a loan
applicant asks why they were rejected, a doctor asks why the model flagged
a scan, a regulator asks you to prove the model isn't discriminating, or
your own model quietly breaks in production and you need to know which
input changed. A model you can't explain is a model you can't fully
trust, debug, or defend — no matter how good its score is.

## The concept

**Global vs local explanations: two different questions.** A **global**
explanation answers "which features matter to this model overall, across
every prediction it makes?" A **local** explanation answers a narrower,
often more useful question: "why did the model make *this specific*
prediction, for *this one* row?" Feature importance (this lesson's focus)
is global. SHAP and LIME (below) are the standard tools for local,
per-prediction explanations.

**Built-in feature importance: fast, but biased.** Tree-based models
(`RandomForestClassifier`, decision trees, gradient boosting) come with a
free `.feature_importances_` attribute. It's computed from **mean
decrease in impurity (MDI)**: every time a feature is used to split a
node, the tree records how much that split reduced impurity (how much
"purer" the resulting groups became), and a feature's importance is the
total of that credit across every tree. It's convenient — no extra
computation — but it has a real bias: it's measured on the *training*
data the trees were fit to, so a feature the model has overfit to (learned
spurious patterns from, rather than genuine signal) can still rack up
impurity-reduction credit and look important, even though it explains
nothing on new data.

**Permutation importance: model-agnostic, and it doesn't lie the same
way.** The idea is simple and doesn't care what kind of model you're
using: take a trained model and a held-out test set, measure its score,
then shuffle *one column* of the test set (breaking whatever relationship
that feature had with the target, while leaving every other column
untouched) and measure the score again. If shuffling a feature barely
hurts the score, the model wasn't really relying on it. If shuffling it
tanks the score, the model depends on it heavily. Because this is measured
on held-out data using the model's actual predictive performance — not an
internal training-time bookkeeping number — it's a far more trustworthy
signal of what the model has actually learned to use, and it works on
*any* model: linear, tree, neural network, anything with a `.predict()`.

**SHAP and LIME: explaining one prediction at a time.** Feature and
permutation importance both answer the global question. When you need the
local one — "why did the model flag *this* transaction, *this* patient" —
the standard tools are **SHAP** (SHapley Additive exPlanations, which
borrows a concept from cooperative game theory to fairly split credit for
one prediction among its input features) and **LIME** (Local Interpretable
Model-agnostic Explanations, which fits a simple, interpretable model
around just the neighborhood of one prediction to approximate why the
complex model landed there). Both are widely used in production
interpretability work; this lesson keeps the runnable code to permutation
importance, which ships with scikit-learn and needs no extra install, but
you should know SHAP and LIME by name for local explanations.

## In code

A `RandomForestClassifier` on a real dataset, read with its free built-in
importance:

```python
import warnings
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print("test accuracy:", round(model.score(X_test, y_test), 4))

built_in = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 8 features by built-in (impurity) importance:")
print(built_in.head(8).round(4))
```

```
test accuracy: 0.9561

Top 8 features by built-in (impurity) importance:
worst perimeter         0.1331
worst area              0.1281
worst concave points    0.1081
mean concave points     0.0944
worst radius            0.0906
mean radius             0.0587
mean perimeter          0.0552
mean area               0.0499
dtype: float64
```

That looks reasonable on its own — but here's the bias made concrete.
Two genuinely predictive features (`x1`, `x2`) plus six columns of *pure
random noise*, carrying zero real information about the target:

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n = 500
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
y = ((x1 + 0.5 * x2 + rng.normal(0, 0.3, n)) > 0).astype(int)

noise = rng.normal(0, 1, size=(n, 6))
noise_names = [f"noise_{i}" for i in range(6)]
X = pd.DataFrame(np.column_stack([x1, x2, noise]), columns=["x1", "x2"] + noise_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print("test accuracy:", round(model.score(X_test, y_test), 4))

built_in = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nbuilt-in (impurity) importance:")
print(built_in.round(4))

perm = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
perm_s = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("\npermutation importance (test set):")
print(perm_s.round(4))
```

```
test accuracy: 0.9467

built-in (impurity) importance:
x1         0.4657
x2         0.2001
noise_1    0.0649
noise_0    0.0572
noise_3    0.0559
noise_5    0.0545
noise_2    0.0532
noise_4    0.0486
dtype: float64

permutation importance (test set):
x1         0.4013
x2         0.1184
noise_0    0.0080
noise_5    0.0064
noise_1    0.0058
noise_3    0.0058
noise_4   -0.0007
noise_2   -0.0009
dtype: float64
```

Look at the noise columns. Built-in importance hands them a combined
**~0.33** — nearly a third of the total credit — to features that, by
construction, carry no information whatsoever. That's the trees
overfitting to random fluctuations and getting rewarded for it in-sample.
Permutation importance, measured on held-out data, correctly puts every
noise feature near zero (some are even slightly *negative* — shuffling
them randomly helped, which is exactly what you'd expect from pure
noise). This is the mechanism behind the warning above: built-in
importance can promote junk features, and it does it silently.

Back to the real dataset — permutation importance on the breast-cancer
model, compared against its own built-in ranking:

```python
import warnings
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring="accuracy")
perm_df = pd.DataFrame({
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}, index=X.columns).sort_values("importance_mean", ascending=False)

print("Top 8 features by permutation importance:")
print(perm_df.head(8).round(4))
print("\nBottom 5 features by permutation importance:")
print(perm_df.tail(5).round(4))
```

```
Top 8 features by permutation importance:
                     importance_mean  importance_std
worst area                    0.0094          0.0085
worst perimeter               0.0085          0.0095
mean area                     0.0082          0.0022
mean concave points           0.0082          0.0068
compactness error             0.0079          0.0026
mean radius                   0.0070          0.0035
mean smoothness               0.0067          0.0037
worst radius                  0.0064          0.0060

Bottom 5 features by permutation importance:
                         importance_mean  importance_std
mean texture                         0.0             0.0
mean compactness                     0.0             0.0
fractal dimension error              0.0             0.0
texture error                        0.0             0.0
mean symmetry                        0.0             0.0
```

The overall story roughly agrees with built-in importance here — `worst
area`, `worst perimeter`, and `mean concave points` show up as important
in both — which makes sense, since this model isn't overfit to noise the
way the synthetic example was. But the two rankings aren't identical
(`compactness error` and `mean smoothness` appear in the permutation top
8 but not the built-in one), and the `importance_std` column is telling
you something built-in importance never could: how *stable* that estimate
is across repeated shuffles. A model this well-behaved is exactly when
you'd expect the two methods to mostly agree — permutation importance is
still the one you should trust when they don't.

## Build this

Take a model from an earlier lesson — the `LogisticRegression` from
[Model Evaluation](13-model-evaluation.md) works well — and run
`permutation_importance` on its test set the way the code above does.
Print the top 5 features by importance and write two or three sentences
interpreting them: do they match your intuition about the dataset, and
does the ranking agree with any built-in coefficients or importances that
model exposes?

**Stretch:** re-run permutation importance with a different `scoring`
argument (try `"roc_auc"` if your model is a classifier) and see whether
the feature ranking changes. Then increase `n_repeats` from 30 to 100 and
check whether `importance_std` shrinks — that's the shuffling noise
averaging out, not the model becoming more interpretable.

## Go deeper

- [scikit-learn: Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) — the official guide, including the exact pitfall this lesson demonstrated with correlated and noisy features.
- [scikit-learn: `inspection.permutation_importance` API reference](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) — full parameter reference.
- [Christoph Molnar: Interpretable Machine Learning (free online book)](https://christophm.github.io/interpretable-ml-book/) — the standard, thorough reference covering permutation importance, SHAP, LIME, and more, all free to read.
- [SHAP documentation](https://shap.readthedocs.io/en/latest/) — official docs for the most widely used local-explanation library.

**Next:** [Time-Series Forecasting](38-time-series.md)
