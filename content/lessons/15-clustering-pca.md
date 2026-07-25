---
title: "Clustering & PCA"
stage: 3
order: 15
minutes: 45
difficulty: intermediate
prerequisites: ["ml-fundamentals", "linear-algebra"]
tags: ["machine-learning", "unsupervised-learning", "clustering"]
summary: "k-means clustering, choosing k with the elbow method, hierarchical clustering, and PCA as compression with explained variance."
---

# Clustering & PCA

## Why this matters

Every model since the ML Fundamentals lesson has been supervised — trained
on examples that already came with the right answer. This lesson is your
first look at **unsupervised learning**: finding structure in data with no
labels at all. It answers two very different, very common questions —
"which of these examples are naturally alike?" (clustering) and "can I
describe this data with fewer numbers and lose almost nothing?" (PCA) —
and both come up constantly once you're working with real, unlabelled
data.

## The concept

**k-means clustering: group points by nearest centroid, then repeat.**
k-means partitions data into `k` groups through a simple, iterative loop:
pick `k` starting points (**centroids**), assign every data point to its
nearest centroid, move each centroid to the mean of the points now
assigned to it, and repeat until nothing changes. There's no label telling
it what a "correct" group looks like — it just keeps tightening groups
around their own centres until they stabilise.

**Choosing k with the elbow method.** k-means needs you to decide `k` in
advance, and there's no single formula that hands you the right answer.
**Inertia** — the total squared distance from every point to its assigned
centroid — always goes down as `k` increases (more clusters can only fit
the data at least as well), so you can't just pick the `k` with the lowest
inertia; that's always the largest `k` you try. Instead, plot inertia
against `k` and look for the **elbow**: the point where adding another
cluster stops buying you much improvement, and the curve bends from a
steep drop into a flatter one.

**Hierarchical (agglomerative) clustering: build a tree of merges.**
Instead of starting with `k` centroids, hierarchical clustering starts
with *every point as its own cluster*, then repeatedly merges the two
closest clusters until only `k` remain. It needs no random starting point
(unlike k-means, which can land in different places depending on where its
centroids started), and the sequence of merges can be drawn as a tree (a
**dendrogram**) that shows the data's structure at every possible number of
clusters simultaneously, not just one.

**PCA: compression by keeping the directions of most spread.**
**Principal Component Analysis** finds new axes — **principal
components** — that are combinations of your original features, chosen so
the first axis captures as much of the data's variance (spread) as
possible, the second captures as much of what's left as possible while
staying perpendicular to the first, and so on. Keeping only the first few
components is a form of lossy compression: fewer numbers per row, with
most of the original information kept. This matters for the same reason
compressing a photo does — fewer dimensions are faster to compute with,
easier to plot, and often easier for a model to learn from, as long as you
haven't thrown away too much.

**Explained variance: how much did you actually keep?** Each principal
component comes with an **explained variance ratio** — the fraction of the
original data's total spread that component alone accounts for. Summing
the ratios for the components you kept tells you exactly how much
information survived the compression — "95.8% of the variance kept with 2
components instead of 4" is a precise, checkable claim, not a vague one.

## In code

k-means and the elbow method, on synthetic customers described by annual
spend and visit frequency (three segments built in on purpose, so we have
a known answer to check the clustering against):

```python
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
segment_centers = {"budget": (300, 6), "regular": (900, 18), "vip": (2200, 30)}
rows = []
for spend_mean, visit_mean in segment_centers.values():
    spend = rng.normal(spend_mean, spend_mean * 0.12, 100)
    visits = rng.normal(visit_mean, visit_mean * 0.15, 100)
    rows.append(np.column_stack([spend, visits]))
X_customers = np.clip(np.vstack(rows), 0, None)

# Inertia: total squared distance from every point to its assigned centroid
inertias = []
k_range = range(1, 9)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_customers)
    inertias.append(km.inertia_)
    print(f"k={k}  inertia={km.inertia_:,.0f}")

fig, ax = plt.subplots()
ax.plot(list(k_range), inertias, marker="o")
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow method")
fig.savefig("elbow.png")
print("saved elbow.png")
```

```
k=1  inertia=194,896,902
k=2  inertia=25,822,853
k=3  inertia=8,109,444
k=4  inertia=3,837,148
k=5  inertia=2,472,750
k=6  inertia=1,769,872
k=7  inertia=1,298,934
k=8  inertia=996,045
saved elbow.png
```

The steepest drops happen from k=1 to k=3; after that the curve flattens
out noticeably — the elbow sits around k=3, exactly matching the three
segments this data was built from.

Fitting the final model at k=3 and describing each segment — this is what
"cluster customers and describe the segments" actually looks like:

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
segment_centers = {"budget": (300, 6), "regular": (900, 18), "vip": (2200, 30)}
rows = []
for spend_mean, visit_mean in segment_centers.values():
    spend = rng.normal(spend_mean, spend_mean * 0.12, 100)
    visits = rng.normal(visit_mean, visit_mean * 0.15, 100)
    rows.append(np.column_stack([spend, visits]))
X_customers = np.clip(np.vstack(rows), 0, None)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_customers)

df = pd.DataFrame(X_customers, columns=["annual_spend", "visit_frequency"])
df["segment"] = kmeans.labels_

summary = df.groupby("segment").agg(
    customers=("segment", "size"),
    avg_spend=("annual_spend", "mean"),
    avg_visits=("visit_frequency", "mean"),
).round(1).sort_values("avg_spend")
print(summary)
```

```
         customers  avg_spend  avg_visits
segment                                  
2              100      298.2         6.0
0              100      893.3        18.3
1              100     2188.2        29.6
```

k-means recovered all three built-in segments cleanly — a low-spend,
infrequent group; a mid-spend, regular group; and a high-spend, frequent
"VIP" group — with no labels ever given to it.

Hierarchical clustering, compared against k-means on the same data:

```python
import warnings
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
segment_centers = {"budget": (300, 6), "regular": (900, 18), "vip": (2200, 30)}
rows = []
for spend_mean, visit_mean in segment_centers.values():
    spend = rng.normal(spend_mean, spend_mean * 0.12, 100)
    visits = rng.normal(visit_mean, visit_mean * 0.15, 100)
    rows.append(np.column_stack([spend, visits]))
X_customers = np.clip(np.vstack(rows), 0, None)

kmeans_labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_customers)

# Start with every point as its own cluster, repeatedly merge the two closest
hierarchical_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X_customers)

# The cluster NUMBERS won't necessarily match between methods, so compare
# the grouping itself - 1.0 means the two methods agree perfectly
agreement = adjusted_rand_score(kmeans_labels, hierarchical_labels)
print("agreement between k-means and hierarchical clustering:", round(agreement, 4))
```

```
agreement between k-means and hierarchical clustering: 1.0
```

Two completely different algorithms, no labels given to either, and they
land on the exact same grouping — strong evidence these three segments are
a real structure in the data, not an artifact of one particular method.

PCA, compressing the Iris dataset's four features down to two:

```python
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

iris = load_iris()
X = iris.data   # 4 features: sepal length/width, petal length/width
print("original shape:", X.shape)

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(X_scaled)
X_2d = pca.transform(X_scaled)
print("compressed shape:", X_2d.shape)

print("explained variance ratio per component:", np.round(pca.explained_variance_ratio_, 4))
print("total variance kept with 2 components:", round(pca.explained_variance_ratio_.sum(), 4))
```

```
original shape: (150, 4)
compressed shape: (150, 2)
explained variance ratio per component: [0.7296 0.2285]
total variance kept with 2 components: 0.9581
```

Four numbers per flower became two, and 95.8% of the original spread in
the data survived the compression — the first component alone (72.96%)
already captures most of what distinguishes one flower from another.

## Build this

Build your own synthetic customer dataset following the pattern above, but
with **four** segments instead of three (pick your own spend/visit means
for each). Run the elbow method for `k` from 1 to 10, print the inertia
table, and identify by eye where the elbow sits. Then fit the final
`KMeans` at your chosen `k` and print a `groupby` summary describing each
segment's average spend and visits, the same way the lesson did.

**Stretch:** rerun `KMeans` on the same data with the *wrong* number of
clusters (try `k=2` and `k=8` for your 4-segment data) and print both
summaries. Write two or three sentences comparing what happens to the
segment descriptions when `k` is too small versus too large.

## Go deeper

- [scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html) — the official guide covering k-means, hierarchical clustering, and several other algorithms in depth.
- [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) and [PCA, step by step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — clear, visual walkthroughs of both algorithms in this lesson.
- [scikit-learn: Principal Component Analysis (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca) — the official reference for PCA, including how explained variance is computed.
- [scikit-learn: Selecting the number of clusters with silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) — a second, more quantitative technique for choosing `k`, beyond the elbow method.

**Next:** [Trees & Ensembles](16-trees-ensembles.md)
