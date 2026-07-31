---
title: "Recommender Systems"
stage: 3
order: 45
minutes: 45
difficulty: intermediate
prerequisites: ["clustering-pca"]
tags: ["recommender-systems", "collaborative-filtering"]
summary: "The engine behind every 'you might also like' — content-based vs collaborative filtering, similarity, and the cold-start problem."
---

# Recommender Systems

## Why this matters

Netflix has said roughly 80% of what people watch comes from a
recommendation, not a search. Amazon's product pages, Spotify's Discover
Weekly, YouTube's autoplay queue, TikTok's For You feed — all of it is the
same underlying problem solved at different scales: out of millions of
items, which handful should this specific person see next? The clustering
lesson showed you how to find structure in unlabelled data by grouping
similar points. Recommender systems take that exact same tool — "how
similar are these two things?" — and point it at a business problem worth
billions of dollars a year.

## The concept

**Two ways to answer "what should I show them?"** Every recommender system
answers that question by leaning on one of two signals: what the item *is*,
or what other people *did*.

**Content-based filtering** recommends items similar to ones a user already
liked, based on the items' own **features** — a movie's genre, director,
and cast; a song's tempo and key; an article's topic tags. If you liked one
sci-fi movie, a content-based system finds other movies whose feature
vectors sit nearby, the same nearest-neighbor idea from the embeddings
lesson, just applied to item attributes instead of text meaning. It needs
no data about other users at all — one user's history is enough.

**Collaborative filtering** ignores item features entirely and instead uses
the **user-item interaction matrix**: a table with one row per user, one
column per item, and a rating (or a click, purchase, or watch) in each
cell where an interaction happened. It works on the premise that people who
agreed in the past will probably agree again — no genre tags, no audio
features, just patterns in *who liked what*. This is the more powerful and
more common approach at scale, and it's what the code below builds.

**User-based vs. item-based collaborative filtering.** There are two ways
to walk that interaction matrix. **User-based CF** finds users whose rating
patterns are similar to yours, then recommends what they liked that you
haven't seen. **Item-based CF** flips the comparison: it finds items whose
rating patterns are similar to items you've already rated highly — two
movies are "similar" here if the same users tended to rate them the same
way, regardless of what the movies are actually about — then recommends
those. Item-based CF is usually more stable in production: the set of
items and how they relate to each other changes slowly day to day, while
individual users' tastes and the set of active users shift constantly, so
item-item similarities need far less frequent recomputation. It's also the
approach Amazon popularized at scale (see "Go deeper").

**The cold-start problem.** Collaborative filtering has one hard failure
mode: it needs interaction history to work at all. A **new user** with zero
ratings has no row to compare against anyone else's. A **new item** with
zero ratings has no column to compare either — it's mathematically
invisible to every similarity calculation until someone interacts with it
first. This is the **cold-start problem**, and it has no purely
collaborative fix. The usual answers are hybrids: fall back to
content-based recommendations for new items (they have features even with
zero interactions), fall back to popularity — "most rented this week" —
for new users, and ask a few onboarding questions to get a thin first row
or column started.

**Matrix factorization and embeddings: the modern approach.** Computing
similarity directly on a giant, mostly-empty interaction matrix (most users
rate almost nothing) gets slow and noisy at real scale. **Matrix
factorization** techniques instead *learn* a short embedding vector for
every user and every item — the same "meaning packed into a fixed-length
vector" idea from the embeddings lesson, but trained so that a user's
vector dotted with an item's vector approximates that user's rating for
that item, rather than capturing text meaning. Netflix's famous $1M Netflix
Prize was won using exactly this idea (see "Go deeper"), and it's the
backbone of most production recommenders today, usually layered under a
neural network. The item-item similarity computed directly in this lesson
is the simple, transparent version of the same underlying idea.

## In code

A small ratings matrix — six users, six movies across three genres, rated
1-5, with `0` meaning "never rated" (never a real low rating, to keep the
numbers unambiguous) — then item-item cosine similarity computed by hand,
and top-N recommendations for a user scored from that similarity matrix.
`pip install numpy`.

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)

users = ["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay"]
items = ["Space Odyssey", "Star Voyager", "Laugh Riot", "Comedy Night",
         "Love Actually", "Romance Isle"]

# rows = users, columns = items, 0 = "never rated" (not "rated low")
ratings = np.array([
    [5, 4, 0, 0, 0, 0],   # Ana:  sci-fi only
    [4, 5, 3, 0, 0, 0],   # Ben:  sci-fi, dabbled in comedy
    [0, 3, 5, 4, 0, 0],   # Cleo: comedy fan, dabbled in sci-fi
    [0, 0, 4, 5, 3, 0],   # Dev:  comedy fan, dabbled in romance
    [0, 0, 0, 3, 5, 4],   # Eli:  romance fan, dabbled in comedy
    [0, 0, 0, 0, 4, 5],   # Fay:  romance only
])
print("Ratings matrix (rows=users, cols=items, 0=unrated):")
print(ratings)

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

# Item-item similarity: compare items by their rating COLUMNS across all
# users -- no genre metadata used anywhere in this calculation.
n_items = ratings.shape[1]
item_sim = np.zeros((n_items, n_items))
for i in range(n_items):
    for j in range(n_items):
        item_sim[i, j] = cosine_similarity(ratings[:, i], ratings[:, j])

print("\nItem-item similarity matrix:")
print(item_sim)

def recommend_item_based(user_idx, top_n=2):
    user_ratings = ratings[user_idx]
    unseen = np.where(user_ratings == 0)[0]
    rated_mask = user_ratings > 0
    scores = {}
    for item in unseen:
        weights = item_sim[item][rated_mask]
        weight_sum = weights.sum()
        scores[item] = float(np.dot(weights, user_ratings[rated_mask]) / weight_sum) if weight_sum else 0.0
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]

target = users.index("Ana")
print(f"\n{users[target]} rated:")
for i, r in enumerate(ratings[target]):
    if r > 0:
        print(f"  {items[i]:<15} {r}")

print(f"\nTop recommendations for {users[target]} (item-based CF):")
for item_idx, score in recommend_item_based(target):
    print(f"  {items[item_idx]:<15} predicted rating {score:.2f}")
```

```
Ratings matrix (rows=users, cols=items, 0=unrated):
[[5 4 0 0 0 0]
 [4 5 3 0 0 0]
 [0 3 5 4 0 0]
 [0 0 4 5 3 0]
 [0 0 0 3 5 4]
 [0 0 0 0 4 5]]

Item-item similarity matrix:
[[1.    0.883 0.265 0.    0.    0.   ]
 [0.883 1.    0.6   0.24  0.    0.   ]
 [0.265 0.6   1.    0.8   0.24  0.   ]
 [0.    0.24  0.8   1.    0.6   0.265]
 [0.    0.    0.24  0.6   1.    0.883]
 [0.    0.    0.    0.265 0.883 1.   ]]

Ana rated:
  Space Odyssey   5
  Star Voyager    4

Top recommendations for Ana (item-based CF):
  Laugh Riot      predicted rating 4.31
  Comedy Night    predicted rating 4.00
```

Ana only ever rated the two sci-fi movies, and nobody ever told the system
"sci-fi" or "comedy" — it only ever saw numbers. But the similarity matrix
shows *Space Odyssey* and *Star Voyager* at 0.883 (near-identical rating
patterns), the two comedies at 0.8, and *Space Odyssey* against either
romance movie at exactly 0.0 (they never overlap in a single user's row).
Ana's recommendations land on the comedies, not the romances, purely
because Ben's row bridges sci-fi and comedy — Ben's taste is the only
evidence the system has connecting those two genres at all, and item-based
CF found it.

## Build this

Using the same `ratings` matrix, implement **user-based** collaborative
filtering and compare it against the item-based results above:

1. Compute a **user-user** similarity matrix with the same
   `cosine_similarity` function, but comparing rating *rows* instead of
   columns.
2. For a target user, find their most similar *other* user(s), then
   recommend that neighbor's highly-rated items the target hasn't seen —
   weight by similarity the same way `recommend_item_based` does.
3. Run it for Ana and Cleo. Do the recommendations roughly agree with the
   item-based version? Write two or three sentences on why item-based and
   user-based CF can (and sometimes can't) land on the same answer from the
   same matrix.

**Stretch — cold start:** add a seventh user, `Grace`, with an all-zero
rating row (`np.zeros(6)`). Show that both your user-based and the
item-based similarity calculations break down or return nothing useful for
Grace — a zero vector has no direction, so cosine similarity against it is
undefined or trivially zero everywhere. Then write a `recommend_popularity`
fallback that ignores Grace entirely and instead recommends the items with
the highest **average rating across all users who rated them**, and print
what it returns for Grace. That's the real fix production systems use for
a brand-new user: fall back to "popular with everyone" until they've rated
enough for personalization to kick in.

## Go deeper

- [Google Machine Learning: Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) — free course covering candidate generation, scoring, and matrix factorization in more depth than fits here.
- [Linden, Smith & York — "Amazon.com Recommendations: Item-to-Item Collaborative Filtering"](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf) — the original paper that popularized the item-based approach used in this lesson's code.
- [GroupLens MovieLens datasets](https://grouplens.org/datasets/movielens/) — free, real ratings datasets (from 100K up to 32M ratings) for practicing on data bigger than six movies.
- [Netflix Research: Recommendations](https://research.netflix.com/research-area/recommendations) — how the ideas in this lesson scale to a real production system.
- [Surprise: a Python scikit for recommender systems](https://surprise.readthedocs.io/en/stable/) — a free library implementing collaborative filtering and matrix factorization algorithms if you want to go past hand-written NumPy.

**Next:** [Object Detection & Segmentation](46-object-detection.md)
