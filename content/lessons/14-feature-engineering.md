---
title: "Feature Engineering"
stage: 3
order: 14
minutes: 45
difficulty: intermediate
prerequisites: ["pandas", "model-evaluation"]
tags: ["machine-learning", "feature-engineering", "pandas"]
summary: "Scaling, encoding categoricals, dates, binning, interaction terms, and data leakage — raising a model's score with features, not a new algorithm."
---

# Feature Engineering

## Why this matters

Every lesson so far has handed you numeric features on a plate. Real data
looks nothing like that: a `brand` column full of text, a `sale_date`
column your model can't use as-is, features on wildly different scales.
**Feature engineering** — reshaping raw columns into a form a model can
actually learn from — routinely moves a model's score more than switching
algorithms does. This lesson raises a baseline model's score using nothing
but better features, and ends with the single most common way a model
lies to you about how good it is: leakage.

## The concept

**Scaling and normalisation: putting features on comparable footing.**
`StandardScaler` rescales every column to mean 0, standard deviation 1;
`MinMaxScaler` rescales every column to a fixed range like `[0, 1]`. Scaling
matters enormously for **distance-based** models (k-NN, k-means — a
feature ranging into the thousands silently dominates the distance
calculation over one ranging 0–10) and for **regularised** models (Ridge
and Lasso penalise coefficient *size*, which only means the same thing
across features if they're on the same scale). It matters far less for
plain linear regression or a single tree: rescaling a feature doesn't
change the *relationships* a tree's splits or an unregularised line can
fit, only the units the numbers happen to be in.

**Encoding categoricals: turning text into numbers a model can use.**
`pd.get_dummies` (or `sklearn`'s `OneHotEncoder`) creates one new binary
(0/1) column per category — "is this row Toyota? Honda? Ford?" — instead
of one text column. This avoids implying an order that doesn't exist: if
you encoded `brand` as `0, 1, 2, 3`, a linear model would treat "Ford" as
literally twice "Honda," which is nonsense for an unordered category.

**Dates: not usable directly, full of usable structure.** A raw
`datetime` column isn't a number a model can multiply by a weight, but it
hides real signal — a `month` extracted from it, a `day_of_week`, or a
derived flag like "is this in the high season?" These derived columns
*are* plain numbers, and turn an unusable column into several useful ones.

**Binning: turning a continuous or fine-grained column into buckets.**
Sometimes the *category* matters more than the exact value — grouping ages
into `"18-25"`, `"26-40"`, `"41-65"` (with `pd.cut`) instead of using raw
age can help a model that struggles to find a smooth relationship find a
simpler, chunkier one instead. Extracting "which season" from a date,
below, is binning applied to a date column.

**Interaction terms: features that only mean something together.**
Sometimes two features' *combination* carries information that neither one
carries alone — "old AND high-mileage" might devalue a car far more than
"old" or "high-mileage" separately would predict. A **linear** model can
only ever add each feature's individual effect; it cannot represent a
genuine multiplicative relationship unless you hand it one directly, as a
new column equal to the product of the two originals.

**Leakage: when a feature secretly contains the answer.** **Data leakage**
happens when information that wouldn't actually be available at prediction
time sneaks into a feature — most dangerously, information derived from
the target itself. A model trained with a leaked feature looks
spectacular in testing and then fails in the real world, because the
leaked information won't exist yet when you actually need a prediction.
It is the single most common reason a beginner's model score looks too
good to be true — because it is.

## In code

A baseline model using only the raw numeric columns, ignoring everything
that needs engineering — used cars, with `price` as the target:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
n = 800

brands = rng.choice(["Toyota", "Honda", "Ford", "BMW"], size=n, p=[0.35, 0.3, 0.2, 0.15])
brand_premium = {"Toyota": 0, "Honda": 500, "Ford": -1000, "BMW": 6000}
age_years = rng.uniform(0, 15, n)
mileage_km = np.clip(age_years * rng.uniform(8000, 18000, n) + rng.normal(0, 5000, n), 0, None)
doors = rng.choice([2, 4], size=n, p=[0.25, 0.75])
sale_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
month = sale_date.month
season_boost = np.where((month >= 4) & (month <= 8), 800, 0)   # sells for more in spring/summer
price = (28000 - 900 * age_years - 0.05 * mileage_km
         - 0.02 * age_years * mileage_km / 100
         + np.array([brand_premium[b] for b in brands]) + season_boost + rng.normal(0, 1200, n))
price = np.clip(price, 2000, None).round(0)

df = pd.DataFrame({
    "brand": brands, "age_years": age_years.round(2), "mileage_km": mileage_km.round(0),
    "doors": doors, "sale_date": sale_date, "price": price,
})
print(df.head())
print(df.shape)

X_baseline = df[["age_years", "mileage_km", "doors"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)

baseline_model = LinearRegression().fit(X_train, y_train)
baseline_r2 = r2_score(y_test, baseline_model.predict(X_test))
print("baseline R^2 (raw numeric columns only):", round(baseline_r2, 4))
```

```
    brand  age_years  mileage_km  doors  sale_date    price
0    Ford      11.08    193068.0      4 2023-07-05   7868.0
1   Honda       4.76     47829.0      4 2024-11-22  19987.0
2     BMW      13.35    197092.0      2 2023-11-14  11744.0
3    Ford       8.91    144796.0      4 2024-04-20  11166.0
4  Toyota       1.89     13645.0      4 2024-05-24  25880.0
(800, 6)
baseline R^2 (raw numeric columns only): 0.8883
```

Raising that score with features alone — same `LinearRegression`, no model
change — by encoding `brand` and turning the unusable `sale_date` column
into a binned "high season" flag:

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n = 800
brands = rng.choice(["Toyota", "Honda", "Ford", "BMW"], size=n, p=[0.35, 0.3, 0.2, 0.15])
brand_premium = {"Toyota": 0, "Honda": 500, "Ford": -1000, "BMW": 6000}
age_years = rng.uniform(0, 15, n)
mileage_km = np.clip(age_years * rng.uniform(8000, 18000, n) + rng.normal(0, 5000, n), 0, None)
doors = rng.choice([2, 4], size=n, p=[0.25, 0.75])
sale_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
month = sale_date.month
season_boost = np.where((month >= 4) & (month <= 8), 800, 0)
price = (28000 - 900*age_years - 0.05*mileage_km - 0.02*age_years*mileage_km/100
         + np.array([brand_premium[b] for b in brands]) + season_boost + rng.normal(0,1200,n))
price = np.clip(price, 2000, None).round(0)
df = pd.DataFrame({"brand": brands, "age_years": age_years.round(2), "mileage_km": mileage_km.round(0),
                    "doors": doors, "sale_date": sale_date, "price": price})

def evaluate(features_df, label):
    X_train, X_test, y_train, y_test = train_test_split(features_df, df["price"], test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"{label:45s} R^2 = {r2:.4f}")
    return r2

evaluate(df[["age_years", "mileage_km", "doors"]], "baseline (raw numeric only)")

# One-hot encode the categorical brand column
brand_dummies = pd.get_dummies(df["brand"], prefix="brand", drop_first=True)
step1 = pd.concat([df[["age_years", "mileage_km", "doors"]], brand_dummies], axis=1)
evaluate(step1, "+ one-hot encoded brand")

# Bin the date into a meaningful season flag - a raw timestamp is unusable,
# but this derived column is a plain 0/1 the model can weigh
df["high_season"] = df["sale_date"].dt.month.between(4, 8).astype(int)
step2 = pd.concat([step1, df[["high_season"]]], axis=1)
evaluate(step2, "+ high_season flag from sale_date")

print("\ncolumns in final feature set:", list(step2.columns))
```

```
baseline (raw numeric only)                   R^2 = 0.8883
+ one-hot encoded brand                       R^2 = 0.9740
+ high_season flag from sale_date             R^2 = 0.9771

columns in final feature set: ['age_years', 'mileage_km', 'doors', 'brand_Ford', 'brand_Honda', 'brand_Toyota', 'high_season']
```

R² jumped from 0.888 to 0.977 without touching the model at all — brand
and season both carried real signal the raw numeric columns simply
couldn't express.

Interaction terms, isolated in a small example where the true relationship
is *purely* the product of two features — a case a linear model given only
the two features separately cannot solve at all:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
n = 500
x1 = rng.uniform(-1, 1, n)
x2 = rng.uniform(-1, 1, n)
y = 10 * x1 * x2 + rng.normal(0, 0.5, n)   # pure interaction, no individual linear effect

X_plain = np.column_stack([x1, x2])
X_interaction = np.column_stack([x1, x2, x1 * x2])

for name, X in [("x1, x2 only", X_plain), ("x1, x2, x1*x2", X_interaction)]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    print(f"{name:16s} R^2 = {r2_score(y_test, model.predict(X_test)):.4f}")
```

```
x1, x2 only      R^2 = 0.0041
x1, x2, x1*x2    R^2 = 0.9819
```

`x1` and `x2` alone explain essentially nothing (R² near 0) — because on
their own, neither one correlates with `y` at all, by construction. Adding
their product as a third feature takes R² to 0.98. This is the sharpest
possible illustration of what "a linear model can't see interactions
unless you hand it one" means.

Why scaling matters for a distance-based model, even though it did nothing
for the linear models above — k-NN on the used-car features, unscaled
versus scaled:

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n = 800
brands = rng.choice(["Toyota", "Honda", "Ford", "BMW"], size=n, p=[0.35, 0.3, 0.2, 0.15])
brand_premium = {"Toyota": 0, "Honda": 500, "Ford": -1000, "BMW": 6000}
age_years = rng.uniform(0, 15, n)
mileage_km = np.clip(age_years * rng.uniform(8000, 18000, n) + rng.normal(0, 5000, n), 0, None)
doors = rng.choice([2, 4], size=n, p=[0.25, 0.75])
price = (28000 - 900*age_years - 0.05*mileage_km - 0.02*age_years*mileage_km/100
         + np.array([brand_premium[b] for b in brands]) + rng.normal(0,1200,n))
price = np.clip(price, 2000, None).round(0)
df = pd.DataFrame({"age_years": age_years, "mileage_km": mileage_km, "doors": doors, "price": price})

X = df[["age_years", "mileage_km", "doors"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("feature ranges (train):")
print(X_train.describe().loc[["min", "max"]])

# Unscaled, mileage_km (up to ~260,000) completely swamps doors (2-4) and
# age_years (0-15) in the raw Euclidean distance k-NN uses
knn_unscaled = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train)
r2_unscaled = r2_score(y_test, knn_unscaled.predict(X_test))

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn_scaled = KNeighborsRegressor(n_neighbors=10).fit(X_train_scaled, y_train)
r2_scaled = r2_score(y_test, knn_scaled.predict(X_test_scaled))

print(f"\nk-NN R^2, unscaled features: {r2_unscaled:.4f}")
print(f"k-NN R^2, scaled features:   {r2_scaled:.4f}")
```

```
feature ranges (train):
     age_years     mileage_km  doors
min   0.016012       0.000000    2.0
max  14.991329  263893.348015    4.0

k-NN R^2, unscaled features: 0.8467
k-NN R^2, scaled features:   0.8819
```

Same data, same model, only the scale changed — and R² improved, because
`doors` and `age_years` finally get to influence "which points are
nearest" instead of being drowned out by `mileage_km`'s much larger raw
numbers.

Leakage, made concrete: a column that looks like a feature but is
secretly built from the answer.

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
n = 800
brands = rng.choice(["Toyota", "Honda", "Ford", "BMW"], size=n, p=[0.35, 0.3, 0.2, 0.15])
brand_premium = {"Toyota": 0, "Honda": 500, "Ford": -1000, "BMW": 6000}
age_years = rng.uniform(0, 15, n)
mileage_km = np.clip(age_years * rng.uniform(8000, 18000, n) + rng.normal(0, 5000, n), 0, None)
doors = rng.choice([2, 4], size=n, p=[0.25, 0.75])
sale_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
month = sale_date.month
season_boost = np.where((month >= 4) & (month <= 8), 800, 0)
price = (28000 - 900*age_years - 0.05*mileage_km - 0.02*age_years*mileage_km/100
         + np.array([brand_premium[b] for b in brands]) + season_boost + rng.normal(0,1200,n))
price = np.clip(price, 2000, None).round(0)
df = pd.DataFrame({"brand": brands, "age_years": age_years.round(2), "mileage_km": mileage_km.round(0),
                    "doors": doors, "sale_date": sale_date, "price": price})
brand_dummies = pd.get_dummies(df["brand"], prefix="brand", drop_first=True)
df["high_season"] = df["sale_date"].dt.month.between(4, 8).astype(int)

# LEAKAGE: this "listing_estimate" was calculated by the dealership FROM
# the final sale price (a small markdown applied afterwards) - it encodes
# the answer. In a real pipeline, this is exactly how leakage sneaks in: a
# system that only populates a field once the outcome is already known.
df["listing_estimate"] = df["price"] * 0.98 + rng.normal(0, 50, n)

honest_features = pd.concat(
    [df[["age_years", "mileage_km", "doors", "high_season"]], brand_dummies], axis=1
)
leaky_features = pd.concat([honest_features, df[["listing_estimate"]]], axis=1)

for name, X in [("honest features", honest_features), ("WITH leaky listing_estimate", leaky_features)]:
    X_train, X_test, y_train, y_test = train_test_split(X, df["price"], test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"{name:32s} R^2 = {r2:.4f}")
```

```
honest features                  R^2 = 0.9771
WITH leaky listing_estimate      R^2 = 1.0000
```

R² of 1.0000 from adding one column is not a win to celebrate — it's a
sign that column is a disguised copy of the target. A model this "good" in
testing will fail the moment it meets a real car whose future
`listing_estimate` doesn't exist yet.

## Build this

Start from the used-car `df` in the second code block above (raw numeric
features only, R² ≈ 0.888). Apply exactly two engineering steps of your
choice from this lesson — for example, one-hot encoding `brand` plus a
binned feature of your own design from `age_years` (try
`pd.cut(df["age_years"], bins=[0, 3, 8, 15], labels=["new", "mid", "old"])`
then one-hot encode that) — and report the before/after R² using the same
`evaluate` pattern shown above. In one or two sentences, explain which of
your two changes contributed more, and how you can tell from the numbers.

**Stretch:** deliberately engineer one *leaky* feature of your own into
this dataset (something derived from `price`), confirm it inflates R²
unrealistically the way `listing_estimate` did, then remove it and confirm
the score returns to the honest level. Write one sentence about what
real-world column in a dataset you've worked with (or can imagine) might
carry this same risk.

## Go deeper

- [scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — the official reference for `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, and more.
- [Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering) — a free, hands-on course covering encoding, interactions, and more feature-engineering techniques than fit in this lesson.
- [scikit-learn: `ColumnTransformer`](https://scikit-learn.org/stable/modules/compose.html#columntransformer) — the standard tool for applying different preprocessing to different columns in one clean pipeline, useful once you're engineering more than a couple of features.
- [Google's Machine Learning Crash Course: Data leakage](https://developers.google.com/machine-learning/crash-course/production-ml-systems) — practical guidance on spotting leakage in real production pipelines, not just toy examples.

**Next:** [Clustering & PCA](15-clustering-pca.md)
