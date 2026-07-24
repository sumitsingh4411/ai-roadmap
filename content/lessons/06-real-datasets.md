---
title: "Real Datasets"
stage: 1
order: 6
minutes: 45
difficulty: beginner
prerequisites: ["pandas"]
tags: ["python", "pandas", "data-cleaning"]
summary: "Where to find datasets, loading messy CSVs, fixing types and dates, removing duplicates, spotting outliers, and a reusable cleaning checklist."
---

# Real Datasets

## Why this matters

The `employees` and `scores` data in the last two lessons was clean on
purpose, so you could focus on syntax. Real data is never that polite:
numbers arrive as text with currency symbols attached, dates arrive in three
different formats in the same column, the same record gets entered twice,
and one typo turns "29" into "290." Every model you'll ever train is only as
good as the cleaning that happened before it, so this lesson is entirely
about that step.

## The concept

**Where to find datasets.** You don't need your own data to practice.
[Kaggle Datasets](https://www.kaggle.com/datasets) has tens of thousands of
free, real-world CSVs across every topic. The
[UCI Machine Learning Repository](https://archive.ics.uci.edu/) is the
classic academic source, still widely used for teaching. Government open-data
portals (search "[your country] open data portal") publish everything from
census figures to transit schedules for free. For quick practice without
downloading anything, scikit-learn and seaborn both ship a handful of small
built-in datasets — you'll meet those in a later stage.

**Loading messy CSVs.** `pd.read_csv` does its best to guess each column's
type from the text it sees, and it guesses wrong constantly. A column of
prices written as `$120.50` gets read as text (`object` dtype), not a number,
because of the `$` sign — pandas has no way to know that's meant to be
currency. The fix is always the same two-step: load it as-is, then look at
`df.dtypes` to see what pandas actually decided, before assuming any column
means what its name suggests.

**Types and parsing dates.** Once you've spotted a column that's the wrong
type, you fix it explicitly. For text-that-should-be-numbers, strip the
non-numeric characters first (`.str.replace("$", "")`), then convert
(`.astype(float)`). For dates, `pd.to_datetime(df["col"])` parses a column of
date-like text into real datetime values — after that, you can do things a
plain string can't do, like sort chronologically, subtract two dates, or pull
out just the month.

**Duplicates.** `df.duplicated()` returns `True` for any row that's an exact
repeat of an earlier one; `df.duplicated().sum()` counts them, and
`df.drop_duplicates()` removes them, keeping the first occurrence by default.
Duplicates creep in from double-submitted forms, merged files, or bugs in
whatever collected the data — always check for them before trusting a count
or a sum computed from the data.

**Outliers.** An outlier is a value far outside the range the rest of the
data occupies — sometimes a genuine (if unusual) data point, sometimes a
data-entry error (an age of `150`, a price of `-40`). One common way to flag
them: compute the interquartile range (IQR), the gap between the 25th and
75th percentile of a column, then treat anything more than 1.5×IQR beyond
that range as an outlier worth a second look. Finding an outlier isn't the
same as deleting it — first find out *why* it's there, because sometimes
the outlier is the most important row in the dataset.

**A reusable cleaning checklist.** Run this, roughly in order, on every new
dataset before you do anything else with it:

1. `df.head()` and `df.shape` — what am I looking at, and how big is it?
2. `df.dtypes` — does every column's type match what its name implies?
3. `df.isna().sum()` — where is data missing, and how much?
4. `df.duplicated().sum()` — are there exact repeat rows?
5. `df.describe()` — do the min/max of each numeric column look plausible?

Five checks, a couple of minutes, and you'll catch the great majority of
problems before they quietly corrupt an analysis three steps later.

## In code

Loading a messy CSV and seeing what pandas actually inferred:

```python
import io
import pandas as pd

csv_text = """order_id,customer,amount,order_date
1,Alice,$120.50,2024-01-05
2,Bob,$85.00,2024-01-06
3,Carol,$200.00,2024-01-06
3,Carol,$200.00,2024-01-06
4,Dave,$95.25,2024-01-07
5,Eve,$15000.00,2024-01-08
6,Frank,$60.00,2024-01-09
"""

df = pd.read_csv(io.StringIO(csv_text))
print(df.dtypes)
print(df)
```

```
order_id       int64
customer      object
amount        object
order_date    object
dtype: object
   order_id customer     amount  order_date
0         1    Alice    $120.50  2024-01-05
1         2      Bob     $85.00  2024-01-06
2         3    Carol    $200.00  2024-01-06
3         3    Carol    $200.00  2024-01-06
4         4     Dave     $95.25  2024-01-07
5         5      Eve  $15000.00  2024-01-08
6         6    Frank     $60.00  2024-01-09
```

`amount` and `order_date` both came in as `object` (text) — exactly the two
columns that need fixing:

```python
df["amount"] = df["amount"].str.replace("$", "", regex=False).astype(float)
df["order_date"] = pd.to_datetime(df["order_date"])

print(df.dtypes)
print(df)
```

```
order_id               int64
customer              object
amount               float64
order_date    datetime64[ns]
dtype: object
   order_id customer    amount order_date
0         1    Alice    120.50 2024-01-05
1         2      Bob     85.00 2024-01-06
2         3    Carol    200.00 2024-01-06
3         3    Carol    200.00 2024-01-06
4         4     Dave     95.25 2024-01-07
5         5      Eve  15000.00 2024-01-08
6         6    Frank     60.00 2024-01-09
```

Duplicates and outliers:

```python
print("duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print(df)

q1 = df["amount"].quantile(0.25)
q3 = df["amount"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print("normal range:", round(lower, 2), "to", round(upper, 2))

outliers = df[(df["amount"] < lower) | (df["amount"] > upper)]
print(outliers)
```

```
duplicate rows: 1
   order_id customer    amount order_date
0         1    Alice    120.50 2024-01-05
1         2      Bob     85.00 2024-01-06
2         3    Carol    200.00 2024-01-06
4         4     Dave     95.25 2024-01-07
5         5      Eve  15000.00 2024-01-08
6         6    Frank     60.00 2024-01-09
normal range: -51.28 to 318.97
   order_id customer   amount order_date
5         5      Eve  15000.0 2024-01-08
```

Carol's duplicate order is gone, and Eve's $15,000 order is flagged as an
outlier against the rest of the (much smaller) orders. Whether that's a
data-entry typo or a genuinely huge order is a judgement call the code can't
make for you — only a look at where the data came from can.

## Build this

Clean this deliberately messy signups dataset:

```python
import io
import pandas as pd

csv_text = """signup_id,name,age,email,signup_date
1,  Alice Kim ,29,alice@example.com,2024-02-01
2,Bob Lee,34,BOB@EXAMPLE.COM,2024-02-02
3,Carol Diaz,150,carol@example.com,2024-02-02
4,Dave Chen,41,dave@example.com,2024-02-03
5,Eve Wong,27,eve@example.com,2024-02-05
5,Eve Wong,27,eve@example.com,2024-02-05
6,Frank Ito,31,frank@example.com,
"""

df = pd.read_csv(io.StringIO(csv_text))
```

Run the five-step checklist from this lesson, then fix what it finds:

- Strip stray whitespace from `name` (`.str.strip()`).
- Normalise `email` to lowercase (`.str.lower()`) so the same address in
  different cases doesn't look like two different people.
- Parse `signup_date` with `pd.to_datetime` and decide what to do about
  Frank's missing date (drop the row? fill it? leave it as a missing date?).
- Find and remove the exact duplicate row.
- Use the IQR method to flag Carol's age as an outlier, and decide what to
  do about it (an age of 150 is almost certainly a data-entry error, not a
  real value).

For **every** decision — each fix you make and each one you deliberately
choose *not* to make — write a one-line comment in your script explaining
why. "Documented every decision" means someone reading only your comments,
not your code, could explain what happened to this dataset and why.

**Stretch:** turn the five-step checklist into a function,
`explore(df)`, that runs all five checks and prints their results with
labels. Run it against both this lesson's dataset and the pandas lesson's
employees dataset to confirm it works on more than one shape of data.

## Go deeper

- [Kaggle Datasets](https://www.kaggle.com/datasets) — thousands of free real-world datasets to practice on.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — the long-standing academic source for benchmark datasets.
- [pandas: Essential basic functionality — duplicates](https://pandas.pydata.org/docs/user_guide/duplicates.html) — the official guide to `duplicated()` and `drop_duplicates()`.
- [pandas: Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html) — the full reference for `pd.to_datetime` and everything you can do once a column is a real date.
- [Kaggle Learn: Data Cleaning](https://www.kaggle.com/learn/data-cleaning) — a free, hands-on course that goes deeper into missing values, inconsistent entries, and outliers than fits here.

**Next:** [Linear Algebra](07-linear-algebra.md)
