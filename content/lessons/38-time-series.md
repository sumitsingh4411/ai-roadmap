---
title: "Time-Series Forecasting"
stage: 3
order: 38
minutes: 50
difficulty: intermediate
prerequisites: ["feature-engineering"]
tags: ["time-series", "forecasting"]
summary: "Predicting what happens next — why time data breaks normal ML, lag features, a proper time-aware split, and honest baselines."
---

# Time-Series Forecasting

## Why this matters

Every model so far has assumed the rows are interchangeable — shuffle
them, split them randomly, nothing breaks. Time-series data breaks that
assumption completely. Order isn't incidental, it's the whole point:
today's value depends on yesterday's, this week looks like last week, and
the one thing you're actually being asked to predict — the future — is
by definition data you've never seen. Get the split wrong here and your
model will look brilliant in testing and fail the moment it meets a real
future. This lesson shows you exactly how that failure happens, and how
to evaluate honestly instead.

## The concept

**Why time series is different.** In a normal ML dataset, row 401 and row
402 have no special relationship — they could be shuffled and nothing
about the *problem* would change. In a time series, row 402 came right
after row 401, and that adjacency carries information: **autocorrelation**
means nearby points in time tend to be similar (today's temperature
predicts tomorrow's far better than a random day from another month
would). On top of that, most real series have **trend** (a slow drift up
or down over the long run) and **seasonality** (a pattern that repeats on
a fixed schedule — daily, weekly, yearly). A model that ignores order
throws away exactly the structure that makes forecasting possible.

**Why a random split leaks the future.** `train_test_split(shuffle=True)`
is the default you've used all roadmap — and it's actively wrong for time
series. Shuffling scatters points across time, so a test-set row from,
say, day 400 can end up with day 399 and day 401 sitting right there in
the *training* set. Because of autocorrelation, those neighbors are
almost the same value as the row being tested — the model isn't
forecasting an unseen future, it's interpolating between two points it
was already shown. The resulting test score looks better than the model
actually is, and that gap won't show up until it meets real, genuinely
future data with no neighbors in the training set at all. The fix is a
**time-aware split**: train only on the past, test only on a later block
the model never saw any part of, before or after.

**The naive baseline: the bar any real model must clear.** Before
trusting any forecasting model, compare it to the simplest possible one —
the **persistence** (or "naive") baseline: predict that the next value
will equal the last observed value. It sounds too simple to matter, but
for many real series (especially ones with strong autocorrelation and
little noise) it's a genuinely hard baseline to beat. If your carefully
engineered model can't beat "just guess yesterday's number," it isn't
adding value.

**Lag features: turning a series into a supervised problem.** Ordinary
ML models (linear regression, trees) don't understand "time" as a
concept — they just take a row of numbers and predict a target. **Lag
features** bridge the gap: for each day, add columns for the value 1 day
ago, 2 days ago, and so on. Once you've done that, forecasting becomes an
ordinary supervised regression problem — predict `value` from
`lag_1, lag_2, ..., lag_7` — and every model you already know how to use
applies directly. The only new rule is that every lag feature must be
built from values *strictly before* the row it's predicting; a feature
that peeks at the current or a future value is data leakage, no
different in kind from the leakage in the Feature Engineering lesson.

**Evaluating with MAE and RMSE.** Forecasting error is usually reported
as **MAE** (mean absolute error — the average size of your miss, in the
original units) or **RMSE** (root mean squared error — like MAE but
squares errors before averaging, so a few large misses are punished more
than many small ones). Both should always be read next to the naive
baseline's score on the *same* test block — a forecasting model's error
number means nothing in isolation.

## In code

A synthetic two-year daily series with a rising trend and a weekly
seasonal pattern — generated, not real, but built the same way real
seasonal data behaves:

```python
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n_days = 730
t = np.arange(n_days)
trend = 0.03 * t
weekly = 5 * np.sin(2 * np.pi * t / 7)
noise = rng.normal(0, 1.5, n_days)
value = 50 + trend + weekly + noise

dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
series = pd.Series(value, index=dates, name="value")
print(series.head(7).round(2))

print("\nfirst 30 days mean:", round(series.iloc[:30].mean(), 2))
print("last 30 days mean: ", round(series.iloc[-30:].mean(), 2), " <- trend: the level is rising")

by_day = pd.DataFrame({"value": series, "day_of_week": series.index.day_name()})
print("\nmean value by day of week (the seasonal pattern):")
print(by_day.groupby("day_of_week")["value"].mean().round(2))
```

```
2023-01-01    50.46
2023-01-02    52.38
2023-01-03    56.06
2023-01-04    53.67
2023-01-05    45.02
2023-01-06    43.32
2023-01-07    46.46
Freq: D, Name: value, dtype: float64

first 30 days mean: 50.59
last 30 days mean:  71.19  <- trend: the level is rising

mean value by day of week (the seasonal pattern):
day_of_week
Friday       56.03
Monday       64.84
Saturday     56.82
Sunday       61.00
Thursday     58.61
Tuesday      65.61
Wednesday    63.13
```

The level rose from ~50.6 to ~71.2 across the two years (the trend), and
the day-of-week averages swing between ~56 and ~66 in a repeating pattern
(the weekly seasonality) — exactly the two structures described above.

The naive baseline — "tomorrow looks like today" — measured across the
whole series:

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n_days = 730
t = np.arange(n_days)
trend = 0.03 * t
weekly = 5 * np.sin(2 * np.pi * t / 7)
noise = rng.normal(0, 1.5, n_days)
value = 50 + trend + weekly + noise
dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
series = pd.Series(value, index=dates, name="value")

naive_pred = series.shift(1)
naive_mae = mean_absolute_error(series.iloc[1:], naive_pred.iloc[1:])
print("naive persistence baseline MAE (predict = yesterday's value):", round(naive_mae, 4))
```

```
naive persistence baseline MAE (predict = yesterday's value): 3.0784
```

Now the leakage demonstration this lesson is built around. Turn the
series into lag features, then fit the *same* `LinearRegression` two
ways — once with a random shuffled split, once with a proper time-aware
split:

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n_days = 730
t = np.arange(n_days)
trend = 0.03 * t
weekly = 5 * np.sin(2 * np.pi * t / 7)
noise = rng.normal(0, 1.5, n_days)
value = 50 + trend + weekly + noise
dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
series = pd.Series(value, index=dates, name="value")

# Turn the series into a supervised problem: predict today's value from
# the previous 7 days (lag features)
df = pd.DataFrame({"value": series})
for lag in range(1, 8):
    df[f"lag_{lag}"] = df["value"].shift(lag)
df = df.dropna()   # first 7 rows have no full history yet

feature_cols = [f"lag_{i}" for i in range(1, 8)]
X = df[feature_cols]
y = df["value"]

# WRONG: shuffle the rows before splitting, like you would for tabular data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
model_r = LinearRegression().fit(X_train_r, y_train_r)
mae_random = mean_absolute_error(y_test_r, model_r.predict(X_test_r))
print("random shuffled split  - test MAE:", round(mae_random, 4))

# RIGHT: split by time - train only on the past, test only on a later block
split_idx = int(len(df) * 0.8)
X_train_c, X_test_c = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_c, y_test_c = y.iloc[:split_idx], y.iloc[split_idx:]
model_c = LinearRegression().fit(X_train_c, y_train_c)
mae_chrono = mean_absolute_error(y_test_c, model_c.predict(X_test_c))
print("chronological split    - test MAE:", round(mae_chrono, 4))
```

```
random shuffled split  - test MAE: 1.2632
chronological split    - test MAE: 1.5741
```

The random split reports a lower (better-looking) error than the
chronological one — not because that model is actually better at
forecasting, but because shuffling let it train on rows sitting right
next to its test rows in time. The chronological score is the honest
one: it's the only one measuring performance on a block of time the
model never had a neighbor inside of.

The full, honest evaluation — time-aware split, MAE and RMSE, compared
directly against the naive baseline on the *same* held-out block:

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n_days = 730
t = np.arange(n_days)
trend = 0.03 * t
weekly = 5 * np.sin(2 * np.pi * t / 7)
noise = rng.normal(0, 1.5, n_days)
value = 50 + trend + weekly + noise
dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
series = pd.Series(value, index=dates, name="value")

df = pd.DataFrame({"value": series})
for lag in range(1, 8):
    df[f"lag_{lag}"] = df["value"].shift(lag)
df = df.dropna()

feature_cols = [f"lag_{i}" for i in range(1, 8)]
X = df[feature_cols]
y = df["value"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"lag-feature model  - MAE: {mae:.4f}  RMSE: {rmse:.4f}")

# naive baseline, evaluated on the exact same held-out block, for a fair comparison
naive_pred = df["lag_1"].iloc[split_idx:]
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
print(f"naive baseline      - MAE: {naive_mae:.4f}  RMSE: {naive_rmse:.4f}")
```

```
lag-feature model  - MAE: 1.5741  RMSE: 1.9151
naive baseline      - MAE: 3.1530  RMSE: 3.7894
```

On the same held-out future block, the lag-feature model roughly halves
the naive baseline's error (MAE 1.57 vs 3.15). That's the comparison that
actually matters — not the model's score in isolation, but its score
against the simplest thing it had to beat, measured on data that truly
comes after everything it was trained on.

## Build this

Using the same synthetic series (or generate your own with a different
`rng` seed, trend slope, or seasonal period), build lag features
`lag_1` through `lag_7`, split chronologically (80/20, no shuffling), and
train a `LinearRegression` or `DecisionTreeRegressor` on the lag
features. Report MAE and RMSE for your model *and* for the naive
baseline, evaluated on the identical held-out block, and confirm your
model beats it. Write one or two sentences on how much of the improvement
you think comes from the model versus simply how predictable this
particular series is.

**Stretch:** add one more feature — a 7-day rolling mean, computed as
`df["value"].shift(1).rolling(window=7).mean()`. Note the `shift(1)`
before `.rolling()`: without it, the rolling window would include the
current row's own value, leaking the answer into the feature — the same
kind of mistake as the `listing_estimate` leakage in
[Feature Engineering](14-feature-engineering.md), just in a time-series
shape instead of a tabular one. Re-evaluate on the same chronological
split and see whether the rolling-mean feature moves your MAE.

## Go deeper

- [Forecasting: Principles and Practice (free online textbook, Hyndman & Athanasopoulos)](https://otexts.com/fpp3/) — the standard, thorough, completely free reference for time-series forecasting.
- [pandas: Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html) — the official guide to `shift`, `rolling`, `resample`, and date handling used throughout this lesson.
- [scikit-learn: `TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — the built-in cross-validation splitter that generalizes the single chronological split in this lesson to multiple time-ordered folds.
- [Google's Machine Learning Crash Course: time series basics](https://developers.google.com/machine-learning/crash-course) — for a refresher on the general train/test split concepts this lesson builds on.

**Next:** [Reinforcement Learning](39-reinforcement-learning.md)
