---
title: "Classification"
stage: 3
order: 12
minutes: 50
difficulty: beginner
prerequisites: ["regression"]
tags: ["machine-learning", "classification", "scikit-learn"]
summary: "Logistic regression and the sigmoid, decision boundaries, k-nearest neighbours, and strategies for more than two classes."
---

# Classification

## Why this matters

Regression predicts a number; **classification** predicts a category —
spam or not spam, which digit a handwritten image shows, which species a
flower belongs to. It's at least as common a problem as regression in real
ML work, and it turns out to need surprisingly little new machinery: the
same weighted sum from the linear algebra lesson, squeezed through one new
function, gets you most of the way there.

## The concept

**The sigmoid function turns any number into a probability.** Defined as
`sigmoid(z) = 1 / (1 + e^-z)`, it takes any real number — however large,
however negative — and squashes it into the open interval `(0, 1)`.
`sigmoid(0) = 0.5` exactly; large positive inputs push the output toward 1,
large negative inputs push it toward 0. That range, `(0, 1)`, is exactly
what a probability needs to look like.

**Logistic regression: linear regression's weighted sum, read as a
probability.** Despite the name, logistic regression is a *classification*
algorithm. It computes the same `w · x + b` weighted sum as linear
regression, then passes the result through sigmoid, so the output is
"probability this example belongs to the positive class" instead of a raw
number. Training it means finding the weights that make that probability
close to 1 for positive examples and close to 0 for negative ones.

**The decision boundary is where the model is exactly 50/50.** To turn a
probability into an actual predicted class, pick a threshold — by default,
0.5. The **decision boundary** is the line (or curve, or surface, in more
than two dimensions) in feature space where the predicted probability
crosses exactly that threshold; on one side, the model predicts one class,
on the other, the other. For plain logistic regression with two features,
this boundary is always a straight line, because `w · x + b = 0` is a
linear equation — logistic regression can only ever draw straight
(or flat, in higher dimensions) dividing lines between classes, however
curved the sigmoid that sits on top of it looks.

**k-Nearest Neighbours (k-NN): classify by asking your neighbours.** k-NN
takes a completely different approach with no training phase at all: it
just remembers the entire training set. To classify a new point, it finds
the `k` closest training points (by distance in feature space) and predicts
whichever class is most common among them. Small `k` (like 1) makes the
model very sensitive to individual noisy points close to the boundary;
large `k` smooths predictions out by averaging over more neighbours, at
the cost of blurring genuinely sharp boundaries between classes. Because
k-NN relies entirely on *distance*, features on very different scales can
silently dominate it — a preview of the feature-scaling lesson still to
come.

**Multi-class strategies: what to do with more than two classes.** Sigmoid
and a single 0.5 threshold only make sense for two classes. For more, two
common strategies exist. **One-vs-rest** trains one binary classifier per
class, each answering "is this example class K, or not?", and predicts
whichever classifier is most confident. **Softmax** (also called
"multinomial") generalises sigmoid directly: instead of one probability, it
outputs one probability per class, all positive and summing to exactly 1,
letting you read off "70% setosa, 25% versicolor, 5% virginica" in one
step. Modern scikit-learn's `LogisticRegression` picks a sensible strategy
automatically based on your data and solver — the three rows of weights
you'll see below, one per class, are exactly what that multi-class fit
looks like under the hood.

## In code

The sigmoid function itself:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for z in [-10, -2, -1, 0, 1, 2, 10]:
    print(f"z={z:4d}  sigmoid(z)={sigmoid(z):.4f}")
```

```
z= -10  sigmoid(z)=0.0000
z=  -2  sigmoid(z)=0.1192
z=  -1  sigmoid(z)=0.2689
z=   0  sigmoid(z)=0.5000
z=   1  sigmoid(z)=0.7311
z=   2  sigmoid(z)=0.8808
z=  10  sigmoid(z)=1.0000
```

`z = 0` (a perfectly undecided weighted sum) maps to exactly the 0.5
threshold — the natural cutoff between the two classes.

Logistic regression on the Iris dataset, using two features so a decision
boundary is easy to see, predicting all three species at once:

```python
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

iris = load_iris()
X = iris.data[:, [2, 3]]   # petal length, petal width
y = iris.target            # 0=setosa, 1=versicolor, 2=virginica

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
predictions = model.predict(X_test)
print("test accuracy:", round(accuracy_score(y_test, predictions), 4))

# One row of weights per class - how much petal length/width pushes toward each
for class_name, coefs, intercept in zip(iris.target_names, model.coef_, model.intercept_):
    print(f"{class_name:12s} weights={np.round(coefs, 3)}  bias={intercept:.3f}")

sample = [[1.5, 0.3]]   # a short-petalled flower
probs = model.predict_proba(sample)[0]
for class_name, p in zip(iris.target_names, probs):
    print(f"P({class_name}) = {p:.4f}")
```

```
test accuracy: 0.9667
setosa       weights=[-2.575 -1.088]  bias=10.416
versicolor   weights=[ 0.111 -0.843]  bias=2.834
virginica    weights=[2.463 1.931]  bias=-13.250
P(setosa) = 0.9701
P(versicolor) = 0.0299
P(virginica) = 0.0000
```

The three rows of weights are the one-model-per-class picture from "The
concept" made concrete: `virginica`'s weights are strongly positive
(bigger petals push toward virginica), `setosa`'s are strongly negative
(bigger petals push away from setosa) — matching what you'd expect, since
setosa flowers have the smallest petals of the three species.

Drawing the decision boundary by predicting across a grid of points and
colouring by predicted class:

```python
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_predictions = model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, grid_predictions, alpha=0.3, cmap="viridis")
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", edgecolor="k")
ax.set_xlabel(iris.feature_names[2])
ax.set_ylabel(iris.feature_names[3])
ax.set_title("Logistic regression decision boundary (Iris)")
fig.savefig("iris_decision_boundary.png")
print("saved iris_decision_boundary.png")
print("grid shape:", grid_predictions.shape, " unique regions:", np.unique(grid_predictions))
```

```
saved iris_decision_boundary.png
grid shape: (200, 200)  unique regions: [0 1 2]
```

Every point on a 200×200 grid got its own prediction, and colouring each
one by its predicted class draws the three regions logistic regression
carved the feature space into — straight-line boundaries between all three,
as "The concept" predicted.

k-NN at several values of `k`, compared against logistic regression:

```python
import warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for k in [1, 3, 5, 9, 15]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"k={k:2d}  test accuracy={acc:.4f}")

logreg_acc = accuracy_score(y_test, LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))
print("logistic regression test accuracy:", round(logreg_acc, 4))
```

```
k= 1  test accuracy=1.0000
k= 3  test accuracy=0.9667
k= 5  test accuracy=0.9667
k= 9  test accuracy=0.9667
k=15  test accuracy=0.9667
logistic regression test accuracy: 0.9667
```

`k=1` happens to score perfectly here — but be careful reading too much
into that: with only 30 test examples, one extra correct guess is worth
more than 3 percentage points, so small test sets like this one can make a
single lucky (or unlucky) `k` look more meaningful than it is. This is
exactly the kind of judgement call the model evaluation lesson, next,
gives you proper tools for.

## Build this

Using `load_iris()`, pick a *different* pair of two features from
`iris.data` (there are six possible pairs — try sepal length and sepal
width instead of petal length and width) and repeat the logistic
regression fit, accuracy check, and decision boundary plot from this
lesson. Compare the accuracy and the shape of the boundary to the
petal-based version above, and write one sentence about which pair of
features seems to separate the three species more cleanly, and why that
might be (look at the plotted points' spread).

**Stretch:** for your chosen feature pair, sweep `k` from 1 to 20 in steps
of 1 for k-NN, plot test accuracy against `k`, and identify the range of
`k` values that gives the most stable (not just the single highest) accuracy.

## Go deeper

- [scikit-learn: Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) — the official reference, including the multi-class strategies mentioned above.
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — a clear, visual walkthrough of the sigmoid and how logistic regression is fit.
- [scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html) — the official guide to k-NN, including how distance metrics and `k` affect results.
- [scikit-learn: Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) — a visual gallery showing how different classifiers, including logistic regression and k-NN, draw very different decision boundaries on the same data.

**Next:** [Model Evaluation](13-model-evaluation.md)
