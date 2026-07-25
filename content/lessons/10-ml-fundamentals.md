---
title: "ML Fundamentals"
stage: 3
order: 10
minutes: 45
difficulty: beginner
prerequisites: ["numpy", "probability-stats"]
tags: ["machine-learning", "fundamentals", "model-evaluation"]
summary: "Supervised vs unsupervised learning, the train/validation/test split, and overfitting vs underfitting through the bias-variance tradeoff."
---

# ML Fundamentals

## Why this matters

You've built a linear model by hand, computed gradients, and worked
through the probability behind uncertainty. This lesson is where those
pieces turn into a discipline: the vocabulary and workflow that every
lesson from here to the end of this roadmap assumes you already have.
Get "overfitting," "train/test split," and "bias-variance" solid now,
because every remaining lesson in Stage 3 uses them without re-explaining
them.

## The concept

**Supervised vs unsupervised learning.** In **supervised learning**, every
training example comes with a known answer — a **label** — and the model's
job is to learn the mapping from inputs to that answer. Predicting a house
price from its features (a **label** you have for past sales) or
classifying an email as spam (a **label** a human already assigned) are
both supervised. In **unsupervised learning**, there's no label at all —
the model looks for structure in the data on its own, like grouping
customers into segments with nothing telling it what the "right" groups
are. Stage 3 is mostly supervised learning; clustering, later in this
stage, is your first unsupervised example.

**Features and labels, `X` and `y`.** The convention you've seen used
loosely in earlier lessons now gets a name: `X` is the matrix of
**features** (the input columns a model gets to see — one row per
example, one column per feature, exactly the data matrix from the linear
algebra lesson), and `y` is the vector of **labels** (the answer the model
is trying to predict, one value per row of `X`). Every supervised
scikit-learn function you'll call from this lesson onward — `.fit(X, y)`
— takes exactly these two things.

**The train/validation/test split.** You cannot trust a model's score on
data it was trained on — it's had every opportunity to simply memorise the
answers, the way a student who's seen the exam questions in advance would
ace a test without having learned anything. So before doing anything else
with a dataset, you split it: the **training set** is what the model
learns from; the **test set** is held back, untouched, until the very end,
to measure how the model performs on data it has genuinely never seen. A
**validation set** is a third slice, carved out of the training data, used
to make decisions *while* building the model — which settings to use, when
to stop — without ever letting those decisions peek at the test set. The
rule that makes all of this meaningful: the test set is touched exactly
once, at the end, and if you look at it twice and change something in
between, it has quietly become a second validation set.

**Underfitting: too simple to learn the pattern.** An underfit model is
too weak for the problem — a straight line trying to fit a curve — and it
does *poorly on both the training set and the test set*, because it
genuinely never captured the underlying relationship in the first place.

**Overfitting: too good at the training set specifically.** An overfit
model has learned the training data's noise and quirks, not just its
signal — like a student who memorised the exact practice questions instead
of the underlying concept. It shows *low training error but high test
error* — a big gap between the two is the signature of overfitting, and
it's the single most common failure mode in applied machine learning.

**The bias-variance tradeoff.** These two failure modes trade off against
each other, and naming them precisely matters. **Bias** is the error from
a model being too simple to capture the true pattern — systematic
wrongness, present even with infinite data (a straight line will never
perfectly fit a curve, no matter how much data you give it). **Variance**
is the error from a model being too sensitive to the specific training
data it happened to see — retrain it on a slightly different sample and
its predictions swing wildly. A too-simple model has high bias, low
variance (underfitting). A too-complex model has low bias, high variance
(overfitting). The goal is never "zero error" — it's finding the
complexity level where the *combination* of bias and variance is smallest,
which "In code" below makes visible by watching both errors as model
complexity increases.

## In code

`X`, `y`, and the train / validation / test split:

```python
import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# Features (X) and labels (y): predict churn from two features
n = 300
tenure_months = rng.uniform(0, 60, n)
monthly_charges = rng.uniform(20, 120, n)
X = np.column_stack([tenure_months, monthly_charges])

churn_score = -0.05 * tenure_months + 0.02 * monthly_charges + rng.normal(0, 1, n)
y = (churn_score > np.median(churn_score)).astype(int)

print("X shape (rows=examples, cols=features):", X.shape)
print("y shape (one label per example):", y.shape)

# Split into train and test FIRST - the test set stays unseen until the end
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("train:", X_train.shape, " test:", X_test.shape)

# A validation set carves a further slice out of the TRAINING data, used to
# tune choices without ever touching the test set
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42   # 0.25 of 80% = 20% of the total
)
print("train:", X_train2.shape, " val:", X_val.shape, " test:", X_test.shape)
```

```
X shape (rows=examples, cols=features): (300, 2)
y shape (one label per example): (300,)
train: (240, 2)  test: (60, 2)
train: (180, 2)  val: (60, 2)  test: (60, 2)
```

Underfitting and overfitting, made visible: fit polynomial models of
increasing complexity to a curved, noisy relationship, and watch the gap
between training error and test error open up.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)

# A curved relationship on a small dataset - few points make it easy for a
# high-degree model to memorise noise instead of learning the true curve
x = np.linspace(0, 1, 40)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, size=x.shape[0])
X = x.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'degree':>6} {'train MSE':>10} {'test MSE':>10}")
for degree in [1, 2, 4, 8, 15]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression().fit(X_train_poly, y_train)

    train_mse = mean_squared_error(y_train, model.predict(X_train_poly))
    test_mse = mean_squared_error(y_test, model.predict(X_test_poly))
    print(f"{degree:>6} {train_mse:>10.3f} {test_mse:>10.3f}")
```

```
degree  train MSE   test MSE
     1      0.257      0.276
     2      0.242      0.338
     4      0.069      0.074
     8      0.048      0.186
    15      0.021     63.704
```

Read this table as the whole lesson in one place. Degree 1 (a straight
line) is high bias: it can't bend to fit the curve, so both errors are
high and close together — underfitting. Degree 4 is the sweet spot: train
and test error are both low and close together. By degree 8 the gap has
opened (train error still falling, test error rising) — variance creeping
in. By degree 15 the model has essentially memorised the 28 training
points exactly (train MSE near zero) while becoming worthless on new data
(test MSE of 63.7, wildly worse than just guessing degree 1) — textbook
high-variance overfitting.

## Build this

Reuse the polynomial-degree code above with your own dataset: generate `x`
from `np.linspace` over a range of your choice, define `y` as some curved
function of `x` (try `x**3`, or `np.cos`, anything nonlinear) plus noise
from `rng.normal`. Split into train/test, then loop over degrees
`[1, 2, 3, 5, 10, 20]`, printing train and test MSE for each. Identify by
eye which degree is underfitting, which is overfitting, and which looks
like the best tradeoff — write one sentence justifying your answer using
the train/test gap.

**Stretch:** for your best-tradeoff degree, refit the model 5 times on 5
different random train/test splits (change `random_state` each time) and
print the test MSE each time. If your "best" degree were actually
overfitting, you'd expect to see much more of that number bouncing around
between runs than for an underfitting one — do you?

## Go deeper

- [scikit-learn: Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html) — the official worked example this lesson's polynomial demo is based on.
- [scikit-learn: `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — the full reference for every option the function supports.
- [Google's Machine Learning Crash Course: Generalization](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) — a short, clear treatment of overfitting and why it happens.
- [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — a visual, step-by-step explanation of the bias-variance tradeoff.

**Next:** [Regression](11-regression.md)
