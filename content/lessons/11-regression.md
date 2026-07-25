---
title: "Regression"
stage: 3
order: 11
minutes: 50
difficulty: beginner
prerequisites: ["ml-fundamentals", "linear-algebra"]
tags: ["machine-learning", "regression", "scikit-learn"]
summary: "Linear regression from the normal equation to scikit-learn, the MSE/MAE/R² metrics, and Ridge/Lasso regularisation to fight overfitting."
---

# Regression

## Why this matters

Regression — predicting a continuous number, like a price, a temperature,
or a duration — is where linear algebra, calculus, and the bias-variance
tradeoff you just learned all meet in one working model. This lesson closes
the loop: you'll solve linear regression exactly (not by gradient descent
this time, but by a direct formula), do the same fit in one line of
scikit-learn, learn the three metrics you'll use to judge every regression
model for the rest of this roadmap, and use regularisation to fix
overfitting in a model with real, correlated features.

## The concept

**The normal equation: solving linear regression exactly, in one step.**
Gradient descent (last lesson) finds the minimum of the MSE cost function
by taking small steps downhill. For plain linear regression, there's also
a direct formula — the **normal equation** — that jumps straight to the
bottom of the bowl in one calculation: `w = (Xᵀ X)⁻¹ Xᵀ y`,
where `X` has an extra column of 1s prepended so the bias gets solved for
along with the weights. This uses exactly the matrix operations from the
linear algebra lesson: transpose, matrix multiplication, and matrix
inversion. It's not how large models are trained in practice (inverting a
huge matrix is expensive, and gradient descent scales better), but for a
small number of features it's exact, fast, and a good way to confirm you
understand what "fitting" a linear model actually computes.

**The cost function, again.** Both the normal equation and
`LinearRegression().fit(...)` are solving the exact same problem you built
gradient descent for last lesson: minimise mean squared error between
predictions and actual values. Different algorithm, same target.

**Three ways to score a regression model.** **MSE** (mean squared error) —
the average squared gap between predictions and actual values — is what
the model is trained to minimise; squaring punishes big misses hard, but
it also means MSE isn't in the same units as your target, which makes it
hard to interpret on its own. **MAE** (mean absolute error) — the average
*absolute* gap — stays in the target's original units (dollars, if you're
predicting price), so "MAE of $53,000" is directly readable, and it's less
dominated by a handful of huge outlier errors than MSE is. **R²**
("R-squared") is different in kind: it's the fraction of the target's
variance the model explains, from 1.0 (perfect predictions) down through
0.0 (no better than always predicting the mean) and even negative (worse
than that). R² is scale-free, which makes it the easiest of the three to
compare across different problems.

**Regularisation: penalising a model for having large weights.** A linear
model with many features can overfit the same way a high-degree polynomial
did last lesson — by fitting large, precisely-tuned weights that chase
noise in the training data. **Ridge regression** fixes this by adding a
penalty to the cost function equal to the sum of the *squared* weights
(scaled by a strength parameter, `alpha`): the model now has to trade off
"fit the data well" against "keep weights small," and larger `alpha` means
a stronger preference for small weights. **Lasso regression** does the
same thing with the sum of *absolute* weights instead of squared weights —
a difference that sounds small but has a striking consequence: Lasso can
push some weights to *exactly* zero, effectively removing those features
from the model entirely, while Ridge only ever shrinks weights toward zero
without eliminating them. That makes Lasso useful when you suspect some
features are irrelevant and want the model to say so; Ridge is the safer
default when you believe most features matter at least a little.

## In code

The normal equation, from scratch, checked against scikit-learn:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)

n = 100
x1 = rng.uniform(0, 10, n)
x2 = rng.uniform(0, 10, n)
y = 3 * x1 - 2 * x2 + 5 + rng.normal(0, 1, n)   # true weights: 3, -2, bias 5

X = np.column_stack([x1, x2])

# w = (X_b^T X_b)^-1 X_b^T y, where X_b has an extra column of 1s for the bias
X_b = np.column_stack([np.ones(n), X])
w_full = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
bias, w1, w2 = w_full
print("normal equation:  bias=%.3f  w1=%.3f  w2=%.3f" % (bias, w1, w2))

sklearn_model = LinearRegression().fit(X, y)
print("sklearn:          bias=%.3f  w1=%.3f  w2=%.3f" % (
    sklearn_model.intercept_, sklearn_model.coef_[0], sklearn_model.coef_[1]))
```

```
normal equation:  bias=4.309  w1=3.095  w2=-1.964
sklearn:          bias=4.309  w1=3.095  w2=-1.964
```

Exactly matching numbers — the normal equation and scikit-learn's solver
are computing the same thing.

Now on a real dataset — California housing prices — with all three
metrics, and the fitted coefficients:

```python
import warnings
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Some BLAS backends emit a harmless divide-by-zero warning during plain
# matrix multiplication on certain hardware; it doesn't affect the result.
warnings.filterwarnings("ignore", category=RuntimeWarning)

housing = fetch_california_housing()
X, y = housing.data, housing.target   # y = median house value, in $100,000s
feature_names = housing.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mse_manual = np.mean((predictions - y_test) ** 2)
print("MSE (sklearn):", round(mse, 4), " MSE (manual):", round(mse_manual, 4))

mae = mean_absolute_error(y_test, predictions)
print("MAE:", round(mae, 4), "-> off by about $%.0f on average" % (mae * 100_000))

r2 = r2_score(y_test, predictions)
print("R^2:", round(r2, 4))

print()
print("coefficients:")
for name, coef in sorted(zip(feature_names, model.coef_), key=lambda t: -abs(t[1])):
    print(f"  {name:12s} {coef:8.4f}")
print("  intercept   ", round(model.intercept_, 4))
```

```
MSE (sklearn): 0.5559  MSE (manual): 0.5559
MAE: 0.5332 -> off by about $53320 on average
R^2: 0.5758

coefficients:
  AveBedrms      0.7831
  MedInc         0.4487
  Longitude     -0.4337
  Latitude      -0.4198
  AveRooms      -0.1233
  HouseAge       0.0097
  AveOccup      -0.0035
  Population    -0.0000
  intercept    -37.0233
```

Reading the coefficients: `MedInc` (median income) pushes price up, which
matches intuition, and `Latitude`/`Longitude` being large and negative
just reflects California's geography (house prices vary a lot by
location). But look at `AveRooms` (-0.12) versus `AveBedrms` (+0.78) — a
bedroom is a room, so it's suspicious that more rooms would *lower* the
predicted price while more bedrooms *raises* it. This is **multicollinearity**:
`AveRooms` and `AveBedrms` are highly correlated with each other, so the
model can't cleanly tell which one deserves credit for the effect, and
splits it unstably between them. This is exactly why "a bigger coefficient
means a more important feature" is not a safe rule to apply blindly —
correlated features are the most common way it breaks.

Ridge and Lasso, on the same dataset (features scaled first, since a
penalty on "weight size" only makes sense when every feature is on a
comparable scale):

```python
import warnings
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear": LinearRegression(),
    "Ridge(a=100)": Ridge(alpha=100),
    "Lasso(a=0.05)": Lasso(alpha=0.05),
}
fitted = {name: model.fit(X_train_scaled, y_train) for name, model in models.items()}

print(f"{'feature':12s}" + "".join(f"{name:>14s}" for name in models))
for i, feature in enumerate(feature_names):
    row = "".join(f"{fitted[name].coef_[i]:14.3f}" for name in models)
    print(f"{feature:12s}{row}")

print()
for name, model in fitted.items():
    r2 = r2_score(y_test, model.predict(X_test_scaled))
    n_zero = np.sum(np.abs(model.coef_) < 1e-8)
    print(f"{name:14s} test R^2={r2:.4f}  zero coefficients={n_zero}")
```

```
feature             Linear  Ridge(a=100) Lasso(a=0.05)
MedInc               0.854         0.848         0.742
HouseAge             0.123         0.130         0.140
AveRooms            -0.294        -0.275        -0.000
AveBedrms            0.339         0.315         0.000
Population          -0.002        -0.000         0.000
AveOccup            -0.041        -0.041        -0.000
Latitude            -0.897        -0.828        -0.259
Longitude           -0.870        -0.800        -0.216

Linear         test R^2=0.5758  zero coefficients=0
Ridge(a=100)   test R^2=0.5778  zero coefficients=0
Lasso(a=0.05)  test R^2=0.5305  zero coefficients=4
```

Ridge shrinks every coefficient a little without eliminating any of them,
and even nudges test R² up very slightly here. Lasso, at this strength,
zeroes out four features entirely — including `AveRooms` and `AveBedrms`,
the exact pair multicollinearity was making unstable above — at the cost
of a lower R². That's the tradeoff regularisation strength controls: push
`alpha` higher and you trade some fit quality for a simpler, more stable
model.

## Build this

Using the California housing dataset above, fit a plain `LinearRegression`
and interpret the top 3 coefficients by magnitude in your own words — for
each one, write a sentence explaining what it means for a feature to have
that sign and size (e.g., "as median income rises by one unit, predicted
price rises by about $45,000, holding other features constant"). Then fit
`Ridge` at three different `alpha` values of your choosing (try one much
smaller and one much larger than 100) and report how the test R² changes
across the three.

**Stretch:** using `Lasso`, sweep `alpha` from `0.001` up to `0.2` in
5-6 steps, and for each one print the number of zeroed coefficients and
the test R². Find the smallest `alpha` at which at least one coefficient
first hits exactly zero.

## Go deeper

- [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) — the official user guide covering ordinary least squares, Ridge, and Lasso in depth.
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) — a clear walkthrough of what fitting a line actually minimises.
- [StatQuest: Regularization Part 1 (Ridge)](https://www.youtube.com/watch?v=Q81RR3yKn30) and [Part 2 (Lasso)](https://www.youtube.com/watch?v=NGf0voTMlcs) — the clearest available explanation of why Ridge shrinks and Lasso zeroes out.
- [scikit-learn: California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) — the official documentation for the dataset used throughout this lesson.

**Next:** [Classification](12-classification.md)
