---
title: "Trees & Ensembles"
stage: 3
order: 16
minutes: 50
difficulty: intermediate
prerequisites: ["model-evaluation"]
tags: ["machine-learning", "decision-trees", "ensembles"]
summary: "Decision trees and how splits are chosen, random forests as bagging, gradient boosting, and feature importance."
---

# Trees & Ensembles

## Why this matters

Decision trees and the ensembles built from them — random forests and
gradient boosting — are the most-used models for real-world tabular data
(spreadsheets and databases, as opposed to images or text), and for good
reason: they need little feature engineering, handle nonlinear
relationships naturally, and, in their boosted form, win more Kaggle
competitions on structured data than any other model family. This lesson
builds up from a single tree's split rule to the ensembles that dominate
practice.

## The concept

**A decision tree asks a sequence of yes/no questions.** Each internal
**node** in the tree tests one feature against a threshold ("is tenure
> 12 months?"), sending each example left or right depending on the
answer; each **leaf** (the end of a path) makes a prediction. Reading a
trained tree from root to leaf is reading the exact rule it learned — this
is the rare model family you can literally print out and follow by hand.

**How a split is chosen: minimising impurity.** At each node, a tree tries
every feature and every possible threshold, and picks whichever split
makes the two resulting groups as "pure" (as dominated by one class) as
possible. **Gini impurity** is the standard measure: the probability that
two randomly picked points from a group would have different labels. It's
0 for a perfectly pure group (every point the same class) and reaches its
maximum, 0.5, for a 50/50 split between two classes. A tree computes the
weighted-average Gini impurity *after* each candidate split, compares it
to the impurity *before*, and greedily takes whichever split reduces
impurity the most — repeating this at every node until it runs out of
useful splits or hits a stopping rule (like maximum depth).

**A single, unrestricted tree overfits.** Left alone, a tree keeps
splitting until every leaf is perfectly pure — which, on real data, usually
means it has memorised the training set, including its noise. This is the
same high-variance, low-bias failure mode from the ML Fundamentals lesson,
and it's the single biggest weakness of decision trees on their own.

**Random forests: bagging many overfit trees into one good model.** A
**random forest** builds many trees — often hundreds — each trained on a
random bootstrap sample of the rows *and* considering only a random subset
of features at each split. Individually, each tree still overfits its own
sample. But because each tree sees different data and different features,
their mistakes are largely uncorrelated, so averaging their predictions
(a technique called **bagging**, short for *bootstrap aggregating*)
cancels out much of that noise. Bagging trades a little bias for a large
reduction in variance — the opposite failure mode from a single deep tree.

**Gradient boosting: build trees sequentially to fix mistakes.** Where
bagging builds trees independently and averages them, **boosting** builds
trees one at a time, in sequence, and each new tree is trained specifically
to correct the errors (technically, the residuals) the trees built so far
are still making. Each tree's contribution is added in with a small
weight (the **learning rate**), so the ensemble improves gradually and
deliberately rather than all at once. Where bagging mainly attacks
variance, boosting mainly attacks bias — it directly targets whatever the
model currently gets wrong.

**XGBoost, LightGBM, and scikit-learn's own histogram-based boosting.**
XGBoost and LightGBM are optimised, industry-standard implementations of
gradient boosting — faster and more tunable than a naive implementation,
and the default choice for tabular-data competitions and production
systems alike. scikit-learn ships its own fast implementation of the same
core idea, `HistGradientBoostingClassifier`/`Regressor`, built on the same
histogram-based trick that makes LightGBM fast. It needs no extra
installation, which is what this lesson's code uses — everything you learn
about it here carries directly over to XGBoost or LightGBM if you reach
for them later.

**Feature importance: which inputs is the model actually using?** Once a
model is trained, you often want to know which features drove its
predictions. **Permutation importance** measures this directly and
model-agnostically: shuffle one feature's values (breaking its relationship
with the target, keeping everything else intact), and measure how much the
model's score drops. A big drop means the model relied on that feature
heavily; almost no drop means it barely mattered. This is more reliable
than a tree ensemble's built-in `.feature_importances_`, which can be
skewed by features with many possible split points, and it works for any
model, not just trees.

## In code

Gini impurity, computed by hand on a tiny toy split, to see exactly what a
tree is optimising when it picks where to divide the data:

```python
import numpy as np

labels_before = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])   # 5 churned, 5 stayed

def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = np.mean(labels)             # fraction that are class 1
    return 1 - p**2 - (1 - p)**2

# Candidate split: "tenure > 12 months?" sends 6 people left, 4 right
left = np.array([1, 1, 1, 0, 1, 0])
right = np.array([0, 1, 0, 0])

gini_before = gini(labels_before)
gini_left, gini_right = gini(left), gini(right)
gini_after = (len(left) / len(labels_before)) * gini_left + (len(right) / len(labels_before)) * gini_right

print("Gini before split:", round(gini_before, 3))
print("Gini left group:  ", round(gini_left, 3), " Gini right group:", round(gini_right, 3))
print("Gini after split (weighted):", round(gini_after, 3))
print("impurity reduction:", round(gini_before - gini_after, 3))
```

```
Gini before split: 0.5
Gini left group:   0.444  Gini right group: 0.375
Gini after split (weighted): 0.417
impurity reduction: 0.083
```

A real tree tries every feature and every threshold, computes this exact
reduction for each candidate, and greedily takes whichever split reduces
impurity the most.

A single unrestricted tree versus a random forest, on the breast cancer
dataset — watch the train/test gap:

```python
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

single_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
train_acc = accuracy_score(y_train, single_tree.predict(X_train))
test_acc = accuracy_score(y_test, single_tree.predict(X_test))
print(f"single tree       train acc={train_acc:.4f}  test acc={test_acc:.4f}  depth={single_tree.get_depth()}")

forest = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
train_acc_f = accuracy_score(y_train, forest.predict(X_train))
test_acc_f = accuracy_score(y_test, forest.predict(X_test))
print(f"random forest      train acc={train_acc_f:.4f}  test acc={test_acc_f:.4f}")
```

```
single tree       train acc=1.0000  test acc=0.9123  depth=7
random forest      train acc=1.0000  test acc=0.9561
```

Both models fit the training data perfectly, but the single tree's
train/test gap (1.0 → 0.912, a drop of 8.8 points) is about twice the
forest's (1.0 → 0.956, a drop of 4.4 points) — bagging's variance
reduction, visible in the numbers.

Gradient boosting, compared against logistic regression and the random
forest above, using 5-fold cross-validation:

```python
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

data = load_breast_cancer()
X, y = data.data, data.target
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
}

for name, model in models.items():
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"{name:20s} accuracy={acc.mean():.4f}  ROC-AUC={auc.mean():.4f}")
```

```
Logistic Regression  accuracy=0.9543  ROC-AUC=0.9917
Random Forest        accuracy=0.9543  ROC-AUC=0.9896
Gradient Boosting    accuracy=0.9578  ROC-AUC=0.9907
```

Gradient boosting edges out the other two on accuracy here, though the
margin is small — this particular dataset is close to linearly separable
to begin with, so logistic regression already performs almost as well as
it can. Boosted trees tend to pull further ahead of linear models on
messier, more nonlinear tabular data; the honest lesson from this table is
"try more than one model and cross-validate," not "boosting always wins."

Feature importance for the boosted model, via permutation importance
(needed here because `HistGradientBoostingClassifier` doesn't expose a
built-in `.feature_importances_` the way a single tree or random forest
does):

```python
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)

result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)
order = np.argsort(result.importances_mean)[::-1][:5]

print("top 5 features by permutation importance:")
for i in order:
    print(f"  {data.feature_names[i]:24s} importance={result.importances_mean[i]:.4f}"
          f"  (+/- {result.importances_std[i]:.4f})")
```

```
top 5 features by permutation importance:
  worst concave points     importance=0.0364  (+/- 0.0139)
  worst texture            importance=0.0193  (+/- 0.0094)
  worst perimeter          importance=0.0184  (+/- 0.0096)
  worst smoothness         importance=0.0184  (+/- 0.0087)
  mean texture             importance=0.0175  (+/- 0.0092)
```

`worst concave points` stands out well above the rest — shuffling it
alone costs the model roughly 3.6 percentage points of accuracy, about
twice the next most important feature.

## Build this

Using the breast cancer dataset, beat this lesson's best cross-validated
score (Gradient Boosting's 0.9578 accuracy / 0.9907 ROC-AUC from the table
above). Try adjusting `HistGradientBoostingClassifier`'s hyperparameters —
`max_iter`, `learning_rate`, `max_depth` — or `RandomForestClassifier`'s
`n_estimators` and `max_depth`, using the same `cross_val_score` pattern
shown in this lesson. Report the best configuration you found and its
score, then run `permutation_importance` on your best model and explain
its top five features in plain English — for each one, look up what it
measures (`data.DESCR` or the [dataset documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)) and write one sentence connecting it to "why would this predict malignant vs. benign?"

**Stretch:** compare your best model's top 5 features (by permutation
importance) against `RandomForestClassifier`'s built-in
`.feature_importances_` on the same data. Do the two methods agree on
which features matter most? Write two or three sentences about any
differences you find.

## Go deeper

- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html) — the official guide to how trees are built and split, including impurity criteria beyond Gini.
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) and [Gradient Boost](https://www.youtube.com/watch?v=3CC4N4z3GJc) — clear, step-by-step visual walkthroughs of both ensemble methods.
- [scikit-learn: Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html) — the official reference covering bagging, random forests, and gradient boosting in depth.
- [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/) — the standard, widely-used boosting library this lesson's concepts transfer directly to.

**Next:** [First ML Project](17-first-ml-project.md)
