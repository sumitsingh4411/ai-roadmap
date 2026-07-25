---
title: "Calculus"
stage: 2
order: 8
minutes: 50
difficulty: intermediate
prerequisites: ["linear-algebra"]
tags: ["math", "calculus", "gradient-descent"]
summary: "Derivatives as slope, the chain rule, and gradient descent implemented by hand in NumPy — how a model actually learns."
---

# Calculus

## Why this matters

"Learning," for almost every model in this roadmap, means one specific
thing: nudging a set of weights, over and over, so that a cost function
gets smaller. Calculus is what tells you *which direction* to nudge them.
This lesson builds gradient descent — the algorithm behind linear
regression, logistic regression, and every neural network you'll train
later — from nothing but the idea of a slope, and by the end you'll have
written it yourself in NumPy and watched it converge.

## The concept

**A derivative is the slope of a function at one point.** If you zoom in
close enough on any smooth curve, it looks like a straight line — the
derivative is the slope of that line. For `f(x) = x**2`, the derivative is
`2x`: at `x = 3`, the curve is rising with slope 6; at `x = 0`, the slope is
flat (0), which is exactly the bottom of the bowl-shaped curve `x**2`
draws. You can approximate any derivative numerically by nudging `x` a tiny
amount and seeing how much `f(x)` moves — that's the *finite difference*
method, and it's a useful sanity check even once you know the formula.

**A cost function turns "how wrong is my model" into a curve you can
descend.** Take the simplest possible model, `prediction = w * x`, and
the mean squared error cost from the last two lessons,
`loss(w) = mean((w*x - y)**2)`. For a fixed dataset, `loss` is a function
of one number, `w` — and if you plot it, it draws a bowl: high loss for
bad guesses of `w`, low loss near the best one, curving smoothly between
them. Training a model is finding the bottom of that bowl.

**The gradient is the derivative generalised to more than one variable.**
A real model has many weights, not one, so `loss` is a function of a whole
vector `w`. The gradient is a vector of partial derivatives — one number
per weight, each answering "if I nudge *just this weight* and hold the
others still, does the loss go up or down, and how fast?" It points in the
direction of steepest increase, which means the *opposite* direction is
the fastest way downhill.

**The chain rule lets you differentiate a function built out of smaller
functions.** Our loss is a composition: `prediction = w * x`, then
`loss = (prediction - y)**2`. The chain rule says the derivative of the
whole thing is the product of the derivatives of the parts:
`d(loss)/dw = d(loss)/d(prediction) * d(prediction)/dw`. Work out each
piece — `d(loss)/d(prediction) = 2*(prediction - y)` and
`d(prediction)/dw = x` — multiply them, and you have the gradient with no
guesswork. This is exactly the calculation "In code" below performs, and
it's the same rule (applied to much deeper compositions of functions) that
makes training a 50-layer neural network possible at all.

**Gradient descent: take a small step downhill, repeat.** The update rule
is `w := w - learning_rate * gradient`. Because the gradient points uphill,
subtracting it moves `w` downhill. The **learning rate** controls the step
size: too large and you overshoot the bottom and bounce around (or
diverge); too small and training crawls. Repeat this update enough times on
a bowl-shaped (convex) loss like MSE, and `w` converges to the minimum —
no calculus textbook required at runtime, just the same arithmetic
repeated thousands of times.

## In code

The derivative as a slope — a numerical (finite-difference) approximation
checked against the known formula:

```python
import numpy as np

def f(x):
    return x ** 2

def numerical_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

for x in [0, 1, 3, -2]:
    approx = numerical_derivative(f, x)
    exact = 2 * x   # the known formula: derivative of x^2 is 2x
    print(f"x={x:>3}  numerical={approx:.4f}  exact (2x)={exact}")
```

```
x=  0  numerical=0.0000  exact (2x)=0
x=  1  numerical=2.0000  exact (2x)=2
x=  3  numerical=6.0000  exact (2x)=6
x= -2  numerical=-4.0000  exact (2x)=-4
```

The loss as a bowl — sweeping `w` across a range and watching the cost:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)   # hours studied
y = np.array([9, 15, 26, 31, 42], dtype=float)   # test score

def mse_loss(w):
    predictions = w * x
    return np.mean((predictions - y) ** 2)

w_values = np.linspace(0, 16, 9)
for w in w_values:
    print(f"w={w:5.1f}  loss={mse_loss(w):8.2f}")

best_w = w_values[np.argmin([mse_loss(w) for w in w_values])]
print("lowest loss in this sweep at w =", best_w)
```

```
w=  0.0  loss=  741.40
w=  2.0  loss=  424.60
w=  4.0  loss=  195.80
w=  6.0  loss=   55.00
w=  8.0  loss=    2.20
w= 10.0  loss=   37.40
w= 12.0  loss=  160.60
w= 14.0  loss=  371.80
w= 16.0  loss=  671.00
lowest loss in this sweep at w = 8.0
```

Loss falls all the way to `w = 8` and rises again after — the bowl shape,
confirmed. Gradient descent finds that bottom without a brute-force sweep,
using the chain rule to compute exactly which way is downhill:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([9, 15, 26, 31, 42], dtype=float)

def mse_loss(w):
    return np.mean((w * x - y) ** 2)

def gradient(w):
    # Chain rule: loss = (prediction - y)^2, prediction = w * x
    # d(loss)/d(prediction) = 2 * (prediction - y)
    # d(prediction)/d(w)    = x
    # d(loss)/d(w)          = d(loss)/d(prediction) * d(prediction)/d(w)
    prediction = w * x
    return np.mean(2 * (prediction - y) * x)

w = 0.0            # start anywhere
learning_rate = 0.01
history = []

for step in range(200):
    grad = gradient(w)
    w = w - learning_rate * grad     # gradient descent update
    if step % 40 == 0 or step == 199:
        history.append((step, round(w, 4), round(mse_loss(w), 4)))

for step, w_val, loss_val in history:
    print(f"step={step:3d}  w={w_val:7.4f}  loss={loss_val:8.4f}")

print("final w:", round(w, 3))

# Sanity check the gradient itself against a plain finite-difference estimate
def numerical_gradient(w, h=1e-6):
    return (mse_loss(w + h) - mse_loss(w - h)) / (2 * h)

print("analytical grad at w=3:", round(gradient(3.0), 4))
print("numerical  grad at w=3:", round(numerical_gradient(3.0), 4))
```

```
step=  0  w= 1.8040  loss=451.7570
step= 40  w= 8.1997  loss=  1.7600
step= 80  w= 8.2000  loss=  1.7600
step=120  w= 8.2000  loss=  1.7600
step=160  w= 8.2000  loss=  1.7600
step=199  w= 8.2000  loss=  1.7600
final w: 8.2
analytical grad at w=3: -114.4
numerical  grad at w=3: -114.4
```

Gradient descent lands on `w ≈ 8.2` — matching the sweep — after starting
from 0 and taking nothing but small downhill steps. The two gradient
values matching confirms the chain-rule formula is correct: this "check
the analytical gradient against a numerical one" trick is a real technique
used to debug gradient calculations in production ML code.

Now with two features and a bias, checked against scikit-learn's own
linear regression:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)

n = 200
x1 = rng.uniform(0, 10, n)
x2 = rng.uniform(0, 10, n)
noise = rng.normal(0, 1, n)
y = 3 * x1 - 2 * x2 + 5 + noise   # true rule the model has to discover

X = np.column_stack([x1, x2])   # shape (200, 2)

def gradient_descent(X, y, learning_rate=0.01, steps=2000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(steps):
        predictions = X @ w + b
        error = predictions - y
        grad_w = (2 / n_samples) * (X.T @ error)   # chain rule, one feature at a time
        grad_b = (2 / n_samples) * np.sum(error)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    return w, b

w, b = gradient_descent(X, y)
print("gradient descent weights:", np.round(w, 3), "bias:", round(b, 3))

sklearn_model = LinearRegression().fit(X, y)
print("sklearn weights:         ", np.round(sklearn_model.coef_, 3), "bias:", round(sklearn_model.intercept_, 3))
```

```
gradient descent weights: [ 3.042 -2.009] bias: 4.782
sklearn weights:          [ 3.04  -2.011] bias: 4.802
```

Our from-scratch gradient descent lands within hundredths of both
scikit-learn's answer and the true rule (`3`, `-2`, `5`) we generated the
data from. `X.T @ error` is exactly last lesson's matrix multiplication:
one line computes every feature's gradient at once, across all 200
examples, with no Python loop.

## Build this

Using the two-feature `gradient_descent` function above, generate your own
synthetic dataset with a *different* true rule (pick your own coefficients
and bias) and confirm the function recovers them. Then deliberately set
`learning_rate=0.5` and re-run — print the loss every 10 steps for the
first 100 steps and describe in one sentence what you observe happening to
`w` (this is "overshooting," the failure mode a too-large learning rate
causes).

**Stretch:** modify `gradient_descent` to also record `mse_loss` (write a
small loss function for this multi-feature case) at every step into a
list, then find the smallest number of steps after which the loss no
longer meaningfully improves for your dataset. That step count is roughly
how long training "needs" to run for this problem.

## Go deeper

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) — the same visual-intuition treatment as the linear algebra series, this time for derivatives and the chain rule.
- [StatQuest: Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — a clear, patient walkthrough of exactly the algorithm you just implemented.
- [Google's Machine Learning Crash Course: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent) — the same idea explained from the model-training side, with interactive visualisations.
- [Khan Academy: Derivatives](https://www.khanacademy.org/math/ap-calculus-ab/ab-derivatives-first-principles) — a full grounding in derivatives from first principles, if the slope intuition above wasn't enough on its own.

**Next:** [Probability & Statistics](09-probability-stats.md)
