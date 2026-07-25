---
title: "Linear Algebra"
stage: 2
order: 7
minutes: 50
difficulty: beginner
prerequisites: ["how-to-learn-ai"]
tags: ["math", "linear-algebra", "numpy"]
summary: "Vectors and matrices as the data structures behind every model, dot products as weighted sums, and matrix multiplication as batch prediction."
---

# Linear Algebra

## Why this matters

Every model in this roadmap — from the five-point line you fit in the first
lesson to the largest language model — computes its predictions the same
way underneath: multiply some numbers together, add them up. Linear algebra
is just the name for the rules that make that multiply-and-add fast and
precise when you're doing it with thousands of numbers instead of five. You
already used a NumPy array in the first two stages; this lesson is about
finally understanding what it represents and why `X @ weights` is the
single most common line of code in machine learning.

## The concept

**A vector is a list of numbers that describes one thing.** A house with
80 square metres, 3 bedrooms, and 15 years of age can be written as the
vector `[80, 3, 15]` — one number per feature, in a fixed order everyone
agrees on. Two vectors of the same length can be added (add each matching
pair of numbers) or scaled (multiply every number by the same constant).
Both operations do exactly what you'd expect: adding two houses' feature
vectors combines their totals; scaling a vector by 2 doubles every feature.

**The dot product is a weighted sum, and a weighted sum is a prediction.**
Given a feature vector `x = [x1, x2, x3]` and a same-length **weight
vector** `w = [w1, w2, w3]`, the dot product is
`x1*w1 + x2*w2 + x3*w3` — multiply matching entries, then add up the
results. This single number is exactly what "What AI... Actually Are"
computed when it predicted a house price from its size: a linear model's
prediction is nothing more than `dot(features, weights) + bias`, where each
weight says "how much this feature pushes the prediction up or down per
unit." Once you can read a dot product, you can read the core computation
of linear regression, logistic regression, and the first layer of every
neural network you'll meet later in this roadmap.

**A vector's length (norm) measures how big it is.** The most common
measure, the L2 norm, is `sqrt(x1**2 + x2**2 + ... )` — Pythagoras'
theorem generalised beyond two dimensions. You'll meet this again soon:
Ridge regression's penalty term is built directly from this norm.

**A matrix is a table of vectors — usually, a whole dataset at once.**
Stack four houses' feature vectors as rows and you get a matrix with shape
`(4, 3)`: 4 rows (examples), 3 columns (features). This is exactly the
`X` you've been passing to `pd.DataFrame` and NumPy arrays since the Stage 1
lessons — a dataset *is* a matrix, and now you know why it's shaped
(rows, columns).

**Matrix multiplication applies a dot product to every row at once.**
Multiplying a `(m, n)` matrix by an `(n, p)` matrix produces an `(m, p)`
result — the "inner" dimensions (`n` and `n`) must match, and they
disappear from the answer's shape. When `p = 1` (multiplying a matrix by a
plain weight vector), this is just "take the dot product of every row with
the same weights" — one matrix multiplication predicts every example in a
dataset in a single step, with no loop. That's not a minor convenience: it's
the difference between code that takes microseconds and code that takes
seconds once your dataset has millions of rows, and it's why every ML
library represents "predict on this whole dataset" as one matrix multiply.

**The transpose flips a matrix on its diagonal.** `X.T` turns a
`(rows, columns)` matrix into `(columns, rows)` — the same numbers, reread
the other way. You'll need this shortly: the next lesson's gradient
calculations and the regression lesson's closed-form solution both use a
transpose to make the shapes line up correctly.

## In code

Vectors, addition, scaling, and the dot product as a weighted sum:

```python
import numpy as np

house_a = np.array([80, 3, 15])   # size (m2), bedrooms, age (years)
house_b = np.array([50, 2, 5])

print(house_a + house_b)
print(house_a * 2)

weights = np.array([2.5, 10.0, -1.0])   # price impact per unit of each feature
bias = 50.0

manual = house_a[0] * weights[0] + house_a[1] * weights[1] + house_a[2] * weights[2] + bias
via_dot = np.dot(house_a, weights) + bias

print(manual)
print(via_dot)

print(np.linalg.norm(house_a))   # the vector's length (L2 norm)
```

```
[130   5  20]
[160   6  30]
265.0
265.0
81.44937077718895
```

`np.dot` gives the exact same answer as multiplying and adding term by
term, because that's literally its definition — it's just faster to write
and, for large vectors, faster to run.

Predicting an entire dataset of houses at once — the slow loop versus one
matrix multiplication:

```python
import numpy as np

# Four houses (rows), three features each (columns): size, bedrooms, age
X = np.array([
    [80,  3, 15],
    [50,  2,  5],
    [110, 4, 30],
    [65,  2, 10],
])
weights = np.array([2.5, 10.0, -1.0])
bias = 50.0

# The slow way: loop over rows, dot product each one by hand
predictions_loop = []
for row in X:
    predictions_loop.append(np.dot(row, weights) + bias)
print(np.array(predictions_loop))

# The matrix way: one matrix-vector multiply predicts all four houses at once
predictions_matmul = X @ weights + bias
print(predictions_matmul)

print(X.shape, weights.shape, predictions_matmul.shape)
```

```
[265.  190.  335.  222.5]
[265.  190.  335.  222.5]
(4, 3) (3,) (4,)
```

`@` is Python's matrix-multiplication operator; `X @ weights` reads as
"apply these weights to every row of X." Same numbers, same answer, no
Python-level loop.

Matrix-by-matrix multiplication, and the transpose:

```python
import numpy as np

X = np.array([
    [80,  3, 15],
    [50,  2,  5],
])   # shape (2, 3): two houses, three features

# Two "pricing models" stacked as columns: each column is one model's weights
W = np.array([
    [2.5,  1.0],
    [10.0, 5.0],
    [-1.0, 0.0],
])   # shape (3, 2)

result = X @ W   # shape (2, 2): row = house, column = which model's raw score
print(result)
print(X.shape, "@", W.shape, "->", result.shape)

X4 = np.array([
    [80,  3, 15],
    [50,  2,  5],
    [110, 4, 30],
    [65,  2, 10],
])
print(X4.shape, "->", X4.T.shape)
print(X4.T)
```

```
[[215.  95.]
 [140.  60.]]
(2, 3) @ (3, 2) -> (2, 2)
(4, 3) -> (3, 4)
[[ 80  50 110  65]
 [  3   2   4   2]
 [ 15   5  30  10]]
```

Notice the rule: `(2, 3) @ (3, 2)` matches on the inner `3`s, and those
disappear from the result's shape `(2, 2)`. Whenever `X @ W` raises a shape
error, this rule is the first thing to check.

## Build this

Using the four-house `X` matrix above, invent your own three weights and a
bias representing a *different* pricing model (change the numbers). Compute
the predictions two ways — a Python loop calling `np.dot` on each row, and
a single `X @ weights + bias` — and confirm with `np.allclose(...)` that
both give the same answer.

**Stretch:** add a fifth house (a new row) with any feature values you like,
re-run the matrix version, and print how the output shape changes.
Then compute that fifth house's L2 norm and explain in one sentence what a
*larger* norm tells you about a feature vector (hint: think about a house
with unusually large numbers in every feature).

## Go deeper

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — the best visual intuition for vectors, dot products, and matrix multiplication that exists, free.
- [NumPy: Linear algebra (`numpy.linalg`)](https://numpy.org/doc/stable/reference/routines.linalg.html) — the official reference for norms, matrix multiplication, and more.
- [Khan Academy: Linear algebra](https://www.khanacademy.org/math/linear-algebra) — a full free course if you want more worked practice problems than fit in one lesson.
- [Google's Machine Learning Crash Course: Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression) — sees the same dot-product idea from the model-building side.

**Next:** [Calculus](08-calculus.md)
