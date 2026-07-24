---
title: "Pandas"
stage: 1
order: 4
minutes: 50
difficulty: beginner
prerequisites: ["numpy"]
tags: ["python", "pandas", "data-wrangling"]
summary: "Series and DataFrames, loading CSVs, selecting and filtering rows, grouping, and handling missing values."
---

# Pandas

## Why this matters

Real-world data doesn't arrive as a tidy NumPy array of numbers — it arrives
as a spreadsheet or CSV with names, dates, missing entries, and columns of
mixed types. Pandas is the library that sits between "a file on disk" and "an
array a model can use," and it's what you'll reach for every single time you
start a new project with real data, including the very next lesson.

## The concept

**`Series` vs `DataFrame`.** A `Series` is a single labelled column of
data — think of it as a NumPy array with an index attached. A `DataFrame` is
a table: a collection of `Series` that share the same index, each with its
own column name. If NumPy's array is the building block, a DataFrame is the
spreadsheet built out of them — rows are records (one employee, one house,
one transaction), columns are the fields describing each record.

**Loading data with `read_csv`.** `pd.read_csv(...)` reads a CSV file (or, as
you'll do in the exercises below, CSV text) into a DataFrame, using the first
line as column names by default. This is almost always the first line of code
in a real data project.

**`head`, `info`, `describe`.** These three are how you get oriented in a
new dataset before doing anything else. `df.head(n)` shows the first `n` rows
so you can eyeball what the data looks like. `df.info()` lists every column
with its dtype and how many non-missing values it has — your fastest way to
spot missing data. `df.describe()` computes count, mean, std, min, max, and
quartiles for every numeric column at once. Run all three, in that order,
on any dataset you haven't seen before.

**Selecting with `loc` and `iloc`.** `df.loc[...]` selects by *label* —
row/column names, like `df.loc[0, "name"]` or a boolean condition. `df.iloc[...]`
selects by *integer position*, like `df.iloc[1:3]` for "the 2nd and 3rd
rows," regardless of what their labels are. When the index is the default
0, 1, 2, ... they often look interchangeable — the difference becomes real
once you filter or sort a DataFrame and the labels no longer match the row
positions. As a rule: reach for `.loc` when you're thinking in column
names and conditions, `.iloc` when you're thinking "the Nth row."

**Filtering.** Putting a condition inside square brackets, like
`df[df["salary"] > 80000]`, keeps only the rows where that condition is
`True` — this is called boolean indexing, and it's the same broadcasting idea
from the NumPy lesson applied to a DataFrame. Combine conditions with `&`
(and) and `|` (or), and wrap each condition in its own parentheses.

**`groupby`.** `df.groupby("department")["salary"].mean()` splits the
DataFrame into one group per unique value in `"department"`, computes the
mean salary within each group, and returns the results indexed by
department. This "split, apply, combine" pattern — split into groups, apply
some calculation, combine into one result — answers most of the questions
you actually have about a dataset ("average by category," "count per group,"
"max per group").

**Handling missing values.** Missing data shows up as `NaN` (Not a Number).
`df.isna().sum()` counts missing values per column, so you know what you're
dealing with before deciding what to do about it. Your two main options are
`df.dropna()` (remove rows with any missing value — simple, but throws away
data) and `df["col"].fillna(value)` (replace missing values with something —
often the column's median or mean, since those are less skewed by outliers
than replacing with 0). There's no universally correct choice; it depends on
*why* the value is missing and how much of the column is missing. This
lesson's exercise asks you to fill; the next lesson on real datasets goes
deeper into deciding when to fill versus drop.

## In code

`Series` and `DataFrame`:

```python
import pandas as pd

salaries = pd.Series([95000, 87000, 72000, 68000], name="salary")
print(salaries)

data = {
    "name": ["Alice", "Bob", "Carol", "Dave"],
    "department": ["Engineering", "Engineering", "Sales", "Sales"],
    "salary": [95000, 87000, 72000, 68000],
}
df = pd.DataFrame(data)
print(df)
```

```
0    95000
1    87000
2    72000
3    68000
Name: salary, dtype: int64
    name   department  salary
0  Alice  Engineering   95000
1    Bob  Engineering   87000
2  Carol        Sales   72000
3   Dave        Sales   68000
```

Loading a CSV and getting oriented. We build the CSV as text first so this
example is fully self-contained — in a real project, `csv_text` would
instead be a path like `"employees.csv"`:

```python
import io
import pandas as pd

csv_text = """name,department,salary,years_experience
Alice,Engineering,95000,5
Bob,Engineering,87000,3
Carol,Sales,72000,4
Dave,Sales,68000,
Eve,Marketing,75000,6
Frank,Marketing,71000,2
Grace,Engineering,102000,8
"""

df = pd.read_csv(io.StringIO(csv_text))

print(df.head(3))
print()
df.info()
print()
print(df.describe())
```

```
    name   department  salary  years_experience
0  Alice  Engineering   95000               5.0
1    Bob  Engineering   87000               3.0
2  Carol        Sales   72000               4.0

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7 entries, 0 to 6
Data columns (total 4 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   name              7 non-null      object 
 1   department        7 non-null      object 
 2   salary            7 non-null      int64  
 3   years_experience  6 non-null      float64
dtypes: float64(1), int64(1), object(2)
memory usage: 352.0+ bytes

              salary  years_experience
count       7.000000          6.000000
mean    81428.571429          4.666667
std     13277.263057          2.160247
min     68000.000000          2.000000
25%     71500.000000          3.250000
50%     75000.000000          4.500000
75%     91000.000000          5.750000
max    102000.000000          8.000000
```

Notice `years_experience` shows only 6 non-null out of 7 rows — that's
Dave's missing value, and `describe()`'s count of 6.0 for that column
confirms it. We'll deal with it below.

Selecting with `loc`/`iloc` and filtering (continuing with the same `df`):

```python
print(df.loc[0, "name"])
print(df.loc[df["department"] == "Engineering", "name"].tolist())
print(df.iloc[0])
print(df.iloc[1:3])

engineers = df[df["department"] == "Engineering"]
print(engineers)

well_paid_sales = df[(df["department"] == "Sales") & (df["salary"] > 70000)]
print(well_paid_sales)
```

```
Alice
['Alice', 'Bob', 'Grace']
name                      Alice
department          Engineering
salary                    95000
years_experience            5.0
Name: 0, dtype: object
    name   department  salary  years_experience
1    Bob  Engineering   87000               3.0
2  Carol        Sales   72000               4.0
    name   department  salary  years_experience
0  Alice  Engineering   95000               5.0
1    Bob  Engineering   87000               3.0
6  Grace  Engineering  102000               8.0
    name department  salary  years_experience
2  Carol      Sales   72000               4.0
```

Missing values and `groupby` (continuing with the same `df`):

```python
print(df.isna().sum())

avg_salary_by_dept = df.groupby("department")["salary"].mean()
print(avg_salary_by_dept)

df["years_experience"] = df["years_experience"].fillna(df["years_experience"].median())
print(df)
```

```
name                0
department          0
salary              0
years_experience    1
dtype: int64
department
Engineering    94666.666667
Marketing      73000.000000
Sales          70000.000000
Name: salary, dtype: float64
    name   department  salary  years_experience
0  Alice  Engineering   95000               5.0
1    Bob  Engineering   87000               3.0
2  Carol        Sales   72000               4.0
3   Dave        Sales   68000               4.5
4    Eve    Marketing   75000               6.0
5  Frank    Marketing   71000               2.0
6  Grace  Engineering  102000               8.0
```

Dave's missing `years_experience` became `4.5` — the median of the other six
values — instead of leaving a gap or dropping his row entirely.

## Build this

Build your own small CSV as text (at least 6 rows, one categorical column to
group by, and at least one numeric column with a missing value — reuse the
employees example or invent your own, like products with a category and
price). Load it with `pd.read_csv(io.StringIO(...))` and answer three
questions using `groupby`:

1. What is the average value of your numeric column per group?
2. How many rows are in each group? (`.size()` or `.count()`)
3. What is the max (or min) value per group?

Print all three results with a label so it's clear which answer is which.

**Stretch:** before grouping, filter out rows below some threshold you
choose (like "only rows where the value is above the overall median") and
re-run your three `groupby` questions on the filtered data. Compare the
results to the unfiltered version and write one sentence about what changed
and why.

## Go deeper

- [pandas: 10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) — the official quick tour, covering everything in this lesson.
- [pandas: Group by: split-apply-combine](https://pandas.pydata.org/docs/user_guide/groupby.html) — the full reference for `groupby`, once you want more than the basics.
- [pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html) — the official guide to `dropna`, `fillna`, and how pandas represents missing values.
- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas) — a free, hands-on course with exercises graded automatically.

**Next:** [Data Visualization](05-data-visualization.md)
