---
title: "Model Evaluation"
stage: 3
order: 13
minutes: 45
difficulty: intermediate
prerequisites: ["classification"]
tags: ["machine-learning", "evaluation", "metrics"]
summary: "The confusion matrix, precision, recall, F1, ROC-AUC, cross-validation, and why accuracy alone can make a worthless model look great."
---

# Model Evaluation

## Why this matters

You've been printing "accuracy" as the score for every classifier so far.
It's about to stop being trustworthy. The moment one class is much rarer
than another — fraud, disease, equipment failure, anything you'd actually
build a classifier to catch — accuracy can hit 97% while catching almost
none of what you care about. This lesson gives you the metrics that don't
lie in that situation, and builds a model that proves the point.

## The concept

**The confusion matrix: every outcome, counted.** For a binary classifier,
there are exactly four possible outcomes for any prediction, arranged in a
2×2 grid:

|                  | predicted negative | predicted positive |
|---|---|---|
| **actually negative** | True Negative (TN) | False Positive (FP) |
| **actually positive** | False Negative (FN) | True Positive (TP) |

Every other metric in this lesson is built from these four counts. A
**false positive** is a false alarm (predicted positive, actually
negative); a **false negative** is a miss (predicted negative, actually
positive) — and which one costs you more depends entirely on the problem
(a missed cancer diagnosis and a missed spam email are not equally bad
false negatives).

**Precision: of what you flagged, how much was right?**
`precision = TP / (TP + FP)`. High precision means when the model says
"positive," you can trust it — few false alarms.

**Recall: of what was actually true, how much did you catch?**
`recall = TP / (TP + FN)`. High recall means the model rarely misses a real
positive case. Precision and recall trade off against each other: a model
that predicts "positive" for almost everything gets perfect recall (it
never misses one) but terrible precision (most of its alarms are false).

**F1: one number that punishes ignoring either one.** F1 is the
**harmonic mean** of precision and recall,
`2 * (precision * recall) / (precision + recall)`. Unlike a plain average,
the harmonic mean is pulled down hard by whichever of the two is lower —
a model with 0.9 precision and 0.1 recall gets an F1 near 0.18, not 0.5 —
so F1 is a fast way to catch a model that's gaming one metric at the total
expense of the other.

**ROC-AUC: how well does the model rank, at every possible threshold?**
Every metric so far assumes a fixed decision threshold (usually 0.5). The
**ROC curve** instead sweeps through *every* possible threshold and plots
the true positive rate against the false positive rate at each one;
**AUC** ("area under the curve") condenses that whole curve into one
number from 0.5 (no better than random guessing) to 1.0 (perfect ranking:
every true positive is scored higher than every true negative). Because it
never fixes a threshold, ROC-AUC keeps working even when accuracy and a
fixed-threshold precision/recall become misleading — it's measuring
whether the model's underlying scores are any good, independent of where
you draw the line.

**Cross-validation: don't trust a single split.** A single train/test
split gives you one score, which could be a little lucky or unlucky
depending on exactly which rows landed in the test set. **k-fold
cross-validation** splits the data into `k` roughly equal parts, trains `k`
separate times (holding out a different part as the test set each time),
and reports all `k` scores — giving you both an average and a sense of how
much that average could reasonably vary.

**Why accuracy misleads on imbalanced data.** If 97% of examples are
negative, a model that *always* predicts negative — one that has learned
nothing at all — scores 97% accuracy by never once being asked to
generalise. Accuracy answers "what fraction did I get right overall,"
which is a bad question the moment the classes aren't roughly balanced;
precision, recall, and ROC-AUC all answer better, more specific questions
that don't get fooled by a lopsided class count.

## In code

The confusion matrix, on a real, only-mildly-imbalanced dataset:

```python
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)

data = load_breast_cancer()
X, y = data.data, data.target   # 0 = malignant, 1 = benign
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"true negatives:  {tn}  (correctly said malignant)")
print(f"false positives: {fp}  (said benign, was actually malignant - dangerous)")
print(f"false negatives: {fn}  (said malignant, was actually benign)")
print(f"true positives:  {tp}  (correctly said benign)")
```

```
[[39  3]
 [ 1 71]]
true negatives:  39  (correctly said malignant)
false positives: 3  (said benign, was actually malignant - dangerous)
false negatives: 1  (said malignant, was actually benign)
true positives:  71  (correctly said benign)
```

Precision, recall, and F1, computed manually and cross-checked against
scikit-learn:

```python
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings("ignore", category=RuntimeWarning)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
predictions = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

precision_manual = tp / (tp + fp)
print("precision (manual):", round(precision_manual, 4), " sklearn:", round(precision_score(y_test, predictions), 4))

recall_manual = tp / (tp + fn)
print("recall (manual):   ", round(recall_manual, 4), " sklearn:", round(recall_score(y_test, predictions), 4))

f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual)
print("F1 (manual):       ", round(f1_manual, 4), " sklearn:", round(f1_score(y_test, predictions), 4))

print()
print(classification_report(y_test, predictions, target_names=data.target_names))
```

```
precision (manual): 0.9595  sklearn: 0.9595
recall (manual):    0.9861  sklearn: 0.9861
F1 (manual):        0.9726  sklearn: 0.9726

              precision    recall  f1-score   support

   malignant       0.97      0.93      0.95        42
      benign       0.96      0.99      0.97        72

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114
```

Now the point of the whole lesson: a deliberately imbalanced problem —
97% "not fraud," 3% "fraud" — where accuracy stops being informative:

```python
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=4, n_redundant=0,
    weights=[0.97, 0.03], class_sep=1.5, flip_y=0.0, random_state=42,
)
print("class balance:", np.bincount(y), "->", round(y.mean() * 100, 2), "% positive (fraud)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def report(name, y_true, y_pred):
    print(f"\n{name}")
    print("  accuracy: ", round(accuracy_score(y_true, y_pred), 4))
    print("  precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
    print("  recall:   ", round(recall_score(y_true, y_pred, zero_division=0), 4))
    print("  f1:       ", round(f1_score(y_true, y_pred, zero_division=0), 4))
    print("  confusion matrix:\n", confusion_matrix(y_true, y_pred))

# The "worthless" model: always predicts the majority class, learns nothing
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
report("DUMMY (always predicts 'not fraud')", y_test, dummy.predict(X_test))

# A real model, trained normally
default_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
report("LOGISTIC REGRESSION (default)", y_test, default_model.predict(X_test))

# Telling it to pay attention to the rare class changes everything except accuracy
balanced_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
report("LOGISTIC REGRESSION (class_weight='balanced')", y_test, balanced_model.predict(X_test))
```

```
class balance: [4850  150] -> 3.0 % positive (fraud)

DUMMY (always predicts 'not fraud')
  accuracy:  0.97
  precision: 0.0
  recall:    0.0
  f1:        0.0
  confusion matrix:
 [[1455    0]
 [  45    0]]

LOGISTIC REGRESSION (default)
  accuracy:  0.9687
  precision: 0.0
  recall:    0.0
  f1:        0.0
  confusion matrix:
 [[1453    2]
 [  45    0]]

LOGISTIC REGRESSION (class_weight='balanced')
  accuracy:  0.782
  precision: 0.1039
  recall:    0.8222
  f1:        0.1845
  confusion matrix:
 [[1136  319]
 [   8   37]]
```

The dummy model — which learned literally nothing — scores 97% accuracy,
tying the "real" default-trained logistic regression, which *also* catches
zero fraud cases (recall 0.0). Both models are worthless for the actual
job, and accuracy alone would never have told you that. Only once we tell
the model to weigh the rare class properly (`class_weight="balanced"`)
does recall jump to 82% — at a real, visible cost: accuracy *drops* to
78% and precision falls to 10%, because catching more real fraud means
tolerating more false alarms. That tradeoff, not a single accuracy number,
is what you're actually choosing between in an imbalanced problem.

Cross-validation and ROC-AUC, back on the breast cancer dataset:

```python
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)

data = load_breast_cancer()
X, y = data.data, data.target

model = LogisticRegression(max_iter=5000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("accuracy per fold:", np.round(scores, 4))
print("mean accuracy:", round(scores.mean(), 4), " std:", round(scores.std(), 4))

auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print("ROC-AUC per fold:", np.round(auc_scores, 4))
print("mean ROC-AUC:", round(auc_scores.mean(), 4))
```

```
accuracy per fold: [0.9649 0.9211 0.9649 0.9474 0.9735]
mean accuracy: 0.9543  std: 0.0187
ROC-AUC per fold: [0.9928 0.9882 0.9854 0.9924 0.9997]
mean ROC-AUC: 0.9917
```

Five folds, five slightly different accuracy scores (0.9211 to 0.9735) —
a single split could easily have landed you on either end of that range
and given you a misleadingly confident (or pessimistic) picture.

## Build this

Using `make_classification`, build your own imbalanced dataset with a
different imbalance ratio than the example above (try `weights=[0.99, 0.01]`
for a rarer positive class, or `weights=[0.9, 0.1]` for a milder one).
Compare a `DummyClassifier(strategy="most_frequent")` against a
`LogisticRegression()` trained normally: report accuracy, precision,
recall, and F1 for both, and write two or three sentences explaining
whether the "real" model is actually better than the dummy one here, and
what evidence in your numbers supports your answer.

**Stretch:** for your logistic regression model, try `class_weight="balanced"`
and report how all four metrics move. Then try adjusting the decision
threshold manually instead — use `model.predict_proba(X_test)[:, 1] > 0.2`
in place of `model.predict(X_test)` — and compare the resulting precision
and recall to the `class_weight="balanced"` version. Which lever feels
more predictable to you as a way of trading precision for recall?

## Go deeper

- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) — the official reference for every classification and regression metric, including all the ones in this lesson.
- [StatQuest: Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) and [ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) — clear, visual walkthroughs of both topics.
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — the official guide to k-fold and other cross-validation strategies.
- [Google's Machine Learning Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification) — covers thresholds, precision, recall, and ROC-AUC from the model-building side.

**Next:** [Feature Engineering](14-feature-engineering.md)
