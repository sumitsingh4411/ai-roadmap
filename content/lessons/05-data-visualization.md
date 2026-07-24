---
title: "Data Visualization"
stage: 1
order: 5
minutes: 40
difficulty: beginner
prerequisites: ["pandas"]
tags: ["python", "matplotlib", "visualization"]
summary: "Choosing the right chart type, matplotlib basics, plotting straight from pandas, and reading what a histogram tells you."
---

# Data Visualization

## Why this matters

`df.describe()` gives you numbers, but numbers hide shape: two datasets can
share the exact same mean and still look completely different once plotted —
one clustered tightly, one full of outliers, one with two separate humps.
Before you train any model on a dataset, you should be able to look at it.
This lesson is a small, permanent toolkit for that — one you'll reuse for
every dataset in the rest of this roadmap.

## The concept

**When to use each chart type.** Most beginner charting mistakes are really
"wrong chart for the question," not a coding error.

| Chart | Question it answers | Example |
|---|---|---|
| Line | How does one value change in order (usually over time)? | Stock price by day |
| Bar | How do totals or averages compare across categories? | Average salary by department |
| Histogram | What's the shape of one numeric column's values? | Distribution of test scores |
| Scatter | How do two numeric columns relate to each other? | Years of experience vs. salary |

Pick the chart from the question you're asking, not the other way around.

**Matplotlib basics.** `fig, ax = plt.subplots()` gives you a `Figure` (the
whole image) and an `Axes` (the actual plot area you draw on) — almost every
example you'll see, in this lesson and elsewhere, starts with that line. You
draw onto `ax` (`ax.plot(...)`, `ax.bar(...)`, `ax.hist(...)`,
`ax.scatter(...)`), then either `plt.show()` to display it interactively or
`fig.savefig("name.png")` to write it to a file. The code examples below use
`savefig` so they run the same way in any environment, including one with no
display attached.

**Plotting straight from pandas.** You rarely need to hand matplotlib raw
arrays. A pandas `Series` or `DataFrame` has a `.plot()` method built on top
of matplotlib, so `avg_salary.plot(kind="bar")` draws a bar chart directly
from a `groupby` result in one line, and `df.plot.scatter(x=..., y=...)`
draws a scatter plot straight from two columns. Pass `ax=ax` to have pandas
draw onto an `Axes` you created yourself, so you can still add labels and a
title with matplotlib's normal methods afterward.

**Labelling axes properly.** An unlabelled chart is a picture, not
information — the reader has to guess what the numbers mean. Every chart you
make from now on should call `ax.set_xlabel(...)`, `ax.set_ylabel(...)`, and
`ax.set_title(...)` with real units, not just variable names. "Salary ($)" is
a label; "salary" alone, on an axis someone else has to interpret, is not.
This is a two-second habit that separates a chart you can hand to someone
else from one only you can read.

**What a histogram tells you about a feature.** A histogram splits a numeric
column into equal-width bins and counts how many values fall in each one, so
its shape *is* the shape of your data. A single central hump means most
values cluster around one typical value (roughly what your data would look
like if it followed a normal, "bell curve" distribution). A long tail
stretching to one side means the column is *skewed* — a handful of very
large (or very small) values pulling the mean away from where most of the
data actually sits, which is exactly the situation where the median is a
more honest summary than the mean. Two separate humps (*bimodal*) often means
you're actually looking at two different groups mixed into one column — for
example, salaries from two very different job levels lumped together. Learning
to read these shapes at a glance is most of what "exploring" a dataset means
in practice.

## In code

Matplotlib basics — a labelled line plot:

```python
import matplotlib
matplotlib.use("Agg")  # renders to a file with no display window needed
import matplotlib.pyplot as plt

days = [1, 2, 3, 4, 5]
study_minutes = [45, 45, 90, 0, 45]

fig, ax = plt.subplots()
ax.plot(days, study_minutes, marker="o")
ax.set_xlabel("Day")
ax.set_ylabel("Minutes studied")
ax.set_title("Study time, week 1")

fig.savefig("study_time.png")
print("saved study_time.png with", len(days), "points")
print("total minutes:", sum(study_minutes))
```

```
saved study_time.png with 5 points
total minutes: 225
```

A histogram, and what its shape tells you:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=42)
test_scores = rng.normal(loc=72, scale=10, size=200)
test_scores = np.clip(test_scores, 0, 100)

fig, ax = plt.subplots()
counts, bin_edges, _ = ax.hist(test_scores, bins=10, edgecolor="white")
ax.set_xlabel("Score")
ax.set_ylabel("Number of students")
ax.set_title("Distribution of test scores")

fig.savefig("test_scores_hist.png")

print("counts per bin:", counts.astype(int))
print("mean score:", round(test_scores.mean(), 1))
print("std dev:", round(test_scores.std(), 1))
```

```
counts per bin: [ 6 15 30 40 41 35 19 10  3  1]
mean score: 71.7
std dev: 8.8
```

The counts rise smoothly from 6 up to a peak of 41 and back down to 1 — a
single central hump with a slightly longer tail on the low side. That's what
a roughly-normal distribution centred near 72 looks like as a histogram.

Plotting straight from a pandas `groupby` result:

```python
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

csv_text = """name,department,salary,years_experience
Alice,Engineering,95000,5
Bob,Engineering,87000,3
Carol,Sales,72000,4
Dave,Sales,68000,4.5
Eve,Marketing,75000,6
Frank,Marketing,71000,2
Grace,Engineering,102000,8
"""
df = pd.read_csv(io.StringIO(csv_text))

avg_salary = df.groupby("department")["salary"].mean()
print(avg_salary)

fig, ax = plt.subplots()
avg_salary.plot(kind="bar", ax=ax)
ax.set_xlabel("Department")
ax.set_ylabel("Average salary ($)")
ax.set_title("Average salary by department")
fig.tight_layout()
fig.savefig("avg_salary_by_dept.png")
print("saved avg_salary_by_dept.png")
```

```
department
Engineering    94666.666667
Marketing      73000.000000
Sales          70000.000000
Name: salary, dtype: float64
saved avg_salary_by_dept.png
```

A scatter plot, for a relationship between two numeric columns:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=42)
study_hours = rng.uniform(0, 10, size=50)
test_scores = 50 + 4 * study_hours + rng.normal(0, 8, size=50)

fig, ax = plt.subplots()
ax.scatter(study_hours, test_scores)
ax.set_xlabel("Hours studied")
ax.set_ylabel("Test score")
ax.set_title("Study hours vs test score")
fig.savefig("study_vs_score.png")

correlation = np.corrcoef(study_hours, test_scores)[0, 1]
print("correlation:", round(correlation, 2))
```

```
correlation: 0.88
```

Points trend clearly upward and a correlation of 0.88 confirms it: more
study hours are strongly associated with higher scores in this (synthetic)
data.

## Build this

Using the same employees CSV from the pandas lesson (rebuild it with
`io.StringIO` as in the examples above), make two charts:

1. A **histogram** of the `salary` column. Write one sentence describing its
   shape — is it clustered, spread out, skewed toward one side?
2. A **scatter plot** of `years_experience` (x-axis) against `salary`
   (y-axis). Write one sentence about the relationship you see — does more
   experience look associated with higher salary here?

Label both charts' axes and give each a title. Save both to files with
`fig.savefig(...)`.

**Stretch:** compute `df["years_experience"].corr(df["salary"])` and compare
the number to what your eye told you from the scatter plot — with only 7
rows, does a single correlation number feel trustworthy? Add three more rows
of invented data and see whether the correlation and the chart's shape
change much.

## Go deeper

- [Matplotlib: Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html) — the official introduction to `Figure`, `Axes`, and the plotting API used throughout this lesson.
- [pandas: Chart visualization](https://pandas.pydata.org/docs/user_guide/visualization.html) — the official reference for `.plot()` on Series and DataFrames.
- [Matplotlib: Choosing colormaps and chart types](https://matplotlib.org/stable/plot_types/index.html) — a visual gallery of chart types, useful for building intuition beyond the four in this lesson's table.
- [Storytelling with Data: chart guide](https://www.storytellingwithdata.com/chart-guide) — a free, practical guide to picking and labelling charts well.

**Next:** [Real Datasets](06-real-datasets.md)
