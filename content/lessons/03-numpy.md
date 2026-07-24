---
title: "NumPy"
stage: 1
order: 3
minutes: 45
difficulty: beginner
prerequisites: ["python-basics"]
tags: ["python", "numpy", "arrays"]
summary: "Why arrays beat lists for numeric work, plus shape, dtype, indexing, broadcasting, and vectorised math."
---

# NumPy

## Why this matters

Every machine learning library you'll touch — pandas, scikit-learn, PyTorch —
stores its numbers in arrays that behave like NumPy's. If you understand a
NumPy array, you already understand the data structure underneath almost
everything else in this roadmap. If you don't, every later lesson will feel
like it's skipping a step. This lesson is that step.

## The concept

**Why arrays beat lists.** A Python list is a general-purpose container — it
can hold numbers, strings, other lists, anything, in any mix. That flexibility
has a cost: to double every number in a list, Python has to loop over it one
item at a time, checking each item's type as it goes. A NumPy array
(`ndarray`) gives that flexibility up on purpose: every element has the same
type, packed together in memory. That uniformity lets NumPy do the loop in
fast, compiled code instead of slow Python, and lets you write `prices * 2`
directly instead of a list comprehension. For a few numbers the difference is
invisible; for the millions of numbers a real dataset or model involves, it's
the difference between seconds and minutes.

**Creating arrays and reading `shape`/`dtype`.** `np.array([[1, 2], [3, 4]])`
turns nested Python lists into a 2D array. Two properties tell you almost
everything about an array at a glance: `.shape` is a tuple of its size along
each dimension — `(4, 3)` means 4 rows, 3 columns — and `.dtype` is the single
type every element shares, like `int64` or `float64`. Whenever an array isn't
behaving how you expect, print `.shape` first; a mismatched shape is the most
common bug in array code.

**Indexing and slicing.** A 1D array indexes like a list: `arr[0]` is the
first element. A 2D array takes two indices, `arr[row, column]`, and each can
be a single number or a `start:stop` slice. `arr[:, 1]` means "every row,
column 1" — the colon alone means "all of this dimension." `arr[1:3, 0:2]`
means "rows 1 up to (not including) 3, columns 0 up to (not including) 2."
Reading array shapes and slices fluently takes practice; the exercise below
is designed to build it.

**Broadcasting.** Broadcasting is NumPy's rule for combining arrays of
different shapes without you writing a loop. `scores + 5` "stretches" the
single number 5 across every element of `scores` automatically. The same rule
lets you combine a 2D array with a 1D array — like subtracting each column's
mean from every value in that column — as long as the shapes are compatible.
The rule in plain terms: compare shapes from the right; dimensions match if
they're equal or one of them is 1. You don't need to memorize the full
algorithm yet — just recognise that "one small thing gets applied across a
big thing" is broadcasting.

**Vectorised math and `axis`.** "Vectorised" means applying an operation to
a whole array at once instead of element by element in a Python loop —
`scores + 5`, `scores.mean()`, `scores * scores` are all vectorised. Many
NumPy functions take an `axis` argument that controls which direction they
collapse. For a 2D array, `axis=0` moves *down* the rows, collapsing each
column to one value (so `scores.mean(axis=0)` gives you one mean per column);
`axis=1` moves *across* the columns, collapsing each row to one value. If you
mix these up — and everyone does at first — check your result's `.shape`
against what you expected.

## In code

Lists versus arrays, doing the same job:

```python
import numpy as np

prices = [10, 20, 30, 40]

doubled_list = [p * 2 for p in prices]  # a Python loop in disguise
print(doubled_list)

prices_arr = np.array(prices)
doubled_arr = prices_arr * 2            # vectorised, no loop written
print(doubled_arr)
```

```
[20, 40, 60, 80]
[20 40 60 80]
```

Creating a 2D array and reading its shape, dtype, and slices:

```python
import numpy as np

scores = np.array([[70, 85, 90],
                    [60, 75, 95],
                    [80, 90, 100],
                    [50, 65, 70]])

print(scores.shape)       # (rows, columns)
print(scores.dtype)

print(scores[0])          # first row
print(scores[:, 1])       # every row, column 1
print(scores[1:3, 0:2])   # rows 1-2, columns 0-1
```

```
(4, 3)
int64
[70 85 90]
[85 75 90 65]
[[60 75]
 [80 90]]
```

Broadcasting and `axis`:

```python
import numpy as np

scores = np.array([[70, 85, 90],
                    [60, 75, 95],
                    [80, 90, 100],
                    [50, 65, 70]])

curved = scores + 5              # broadcasting: 5 applied to every element
print(curved)

column_means = scores.mean(axis=0)  # one mean per column
print(column_means)

row_means = scores.mean(axis=1)     # one mean per row
print(row_means)
```

```
[[ 75  90  95]
 [ 65  80 100]
 [ 85  95 105]
 [ 55  70  75]]
[65.   78.75 88.75]
[81.66666667 76.66666667 90.         61.66666667]
```

## Build this

Using the `scores` array above, **normalise it column-wise without writing a
loop**: for each column, subtract that column's mean and divide by that
column's standard deviation, so every column ends up with mean 0. You'll need
`scores.mean(axis=0)` and `scores.std(axis=0)` — broadcasting handles the rest
in one expression. Convert `scores` to floats first (`scores.astype(float)`)
so the division doesn't round to whole numbers. Print the result and confirm
`result.mean(axis=0)` comes out as (approximately) all zeros.

**Stretch:** do the same normalisation row-wise instead (`axis=1`), and write
one sentence explaining why row-wise wouldn't make sense for this particular
dataset (think about what a row and a column each represent here — a student
and a test, respectively).

## Go deeper

- [NumPy: Absolute Beginners' Guide](https://numpy.org/doc/stable/user/absolute_beginners.html) — the official starting point, written for exactly this stage.
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — the full rule, with diagrams, once the plain-terms version above feels too vague.
- [NumPy: Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html) — every slicing trick beyond what fit in this lesson.
- [Real Python: NumPy Array Programming](https://realpython.com/numpy-array-programming/) — more worked examples of vectorised thinking.

**Next:** [Pandas](04-pandas.md)
