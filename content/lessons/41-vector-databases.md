---
title: "Vector Databases"
stage: 5
order: 41
minutes: 45
difficulty: intermediate
prerequisites: ["embeddings"]
tags: ["embeddings", "vector-search", "databases"]
summary: "Where embeddings live at scale — similarity search, why brute force stops scaling, and how approximate nearest neighbors makes it fast."
---

# Vector Databases

## Why this matters

Lesson 26 turned text into vectors and searched them with plain NumPy —
which was the point at the time: at a few dozen documents, brute-force
cosine similarity *is* what a vector database reduces to. Real systems
don't stay at a few dozen documents. A support-ticket archive, a codebase,
a year of chat logs — these run to millions of vectors, and "compare the
query against every single one" stops being free. **Vector databases**
exist to answer the same question — "which stored vectors are closest to
this one?" — at that scale, in milliseconds instead of seconds. This
lesson opens up what changes between "a NumPy array of embeddings" and
"a system built to hold billions of them."

## The concept

**Recap: embeddings are vectors, similarity is geometry.** From lesson 26:
an embedding model maps text to a fixed-length vector, and cosine
similarity measures how close two vectors point — which, because of how
the model was trained, tracks how similar the two texts mean. Nothing in
this lesson changes that. What changes is *where those vectors live* and
*how the search over them is done* once there are a lot of them.

**Brute force is O(n), and that's the whole problem.** Comparing a query
vector against every stored vector is a **brute-force search**: cost grows
linearly with the number of stored vectors, `n`. Double the vectors,
double the work. That's fine at a thousand vectors — a modern CPU does a
million floating-point multiplications without blinking — but at ten
million vectors, every single query pays for ten million comparisons, and
if you're serving many queries per second, that cost is paid again and
again. "In code" below measures this directly: the growth is real, not
theoretical.

**Approximate nearest neighbor (ANN): trade a little accuracy for a lot
of speed.** The fix isn't a smarter way to compare two vectors — dot
products are already about as cheap as arithmetic gets. The fix is
avoiding comparing against *most* of the vectors at all. An **ANN index**
pre-organizes the stored vectors into a structure that lets a query skip
straight to the region of the vector space it's likely to match, checking
a small fraction of the total instead of everything. Two index families
show up constantly in practice:

- **HNSW** (Hierarchical Navigable Small World) builds a multi-layer graph
  where each vector is connected to a handful of its neighbors. A search
  starts at a sparse top layer and "zooms in" through denser layers,
  narrowing toward the right neighborhood in roughly logarithmic time
  instead of linear.
- **IVF** (Inverted File Index) clusters the vector space ahead of time
  into partitions (via something like k-means), and a query only searches
  the partitions closest to it, skipping the rest entirely.

The word "approximate" is doing real work: neither structure is
guaranteed to return the mathematically exact top-k nearest vectors the
way brute force does — a vector sitting just across a partition boundary
can occasionally get missed. In exchange, a search that would take seconds
over brute force takes milliseconds over an ANN index, and for the
overwhelming majority of applications (search, recommendations, RAG
retrieval), a result that's 99% as good and 100x faster is a trade worth
taking every time.

**What a vector database adds beyond "a NumPy array."** An ANN index is
the headline feature, but a real vector database is an operational system
built around it:

- **Persistence** — vectors survive a process restart; a NumPy array in
  memory doesn't.
- **Metadata filtering** — search *and* constrain by structured fields at
  the same time ("closest vectors where `category = "python"` and `year >=
  2023`"), which a raw index alone doesn't give you for free.
- **Updates and deletes** — add, change, or remove vectors after the index
  is built, without rebuilding the whole thing from scratch every time.
- **Scale beyond one machine's RAM** — sharding and distribution across
  multiple machines once the vector count outgrows a single process.

**The landscape, at a glance.** There's no single "the" vector database —
the right choice depends on scale and how it fits your existing stack.

| Tool | What it is |
|---|---|
| FAISS | A library (not a server) from Meta — you embed it directly in your own process. Extremely fast, no metadata filtering or persistence built in by default. |
| Chroma | An open-source vector database built for the RAG/LLM-app use case specifically; easy to run embedded or as a server. |
| pgvector | A PostgreSQL extension — adds vector search to a database you may already be running, so your vectors live next to the rest of your relational data. |
| Pinecone | A fully managed, hosted vector database — no infrastructure to run, scales to billions of vectors, paid beyond a free tier. |
| Weaviate | An open-source vector database with built-in hybrid (vector + keyword) search and its own query language. |

At the scale most projects start at — thousands to low millions of
vectors — the honest answer is that any of these work, and the ANN
concepts above are what actually transfer between them.

## In code

First, the same brute-force cosine search from lesson 26, but wrapped in a
small class that behaves like a real vector store: it holds vectors *and*
metadata together, and returns both. This is genuinely what a vector
database is, reduced to its essence, before persistence, ANN indexing, and
distribution get added on top.

```python
import numpy as np

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

DIM = 8

# Stand-ins for what a real embedding model (lesson 26) would produce --
# three cluster centers in 8-dimensional space, one per topic.
centers = {
    "python": np.random.randn(DIM),
    "baking": np.random.randn(DIM),
    "finance": np.random.randn(DIM),
}

def make_vector(topic, noise=0.3):
    return centers[topic] + np.random.randn(DIM) * noise

documents = [
    {"text": "Python's list comprehensions replace many for-loops.", "category": "python", "year": 2023},
    {"text": "Type hints catch bugs before you run the code.", "category": "python", "year": 2024},
    {"text": "Virtual environments keep dependencies isolated.", "category": "python", "year": 2022},
    {"text": "Decorators wrap a function with extra behavior.", "category": "python", "year": 2024},
    {"text": "Proof the dough until it doubles in size.", "category": "baking", "year": 2021},
    {"text": "A pinch of salt sharpens sweetness in desserts.", "category": "baking", "year": 2023},
    {"text": "Laminated dough makes croissants flaky.", "category": "baking", "year": 2024},
    {"text": "Sourdough starters need daily feeding.", "category": "baking", "year": 2020},
    {"text": "Diversifying a portfolio reduces risk.", "category": "finance", "year": 2022},
    {"text": "Compound interest rewards starting early.", "category": "finance", "year": 2023},
    {"text": "An emergency fund covers three to six months.", "category": "finance", "year": 2024},
    {"text": "Index funds track a market benchmark cheaply.", "category": "finance", "year": 2021},
]

class VectorStore:
    """A minimal vector database: store vectors + metadata, search by cosine similarity."""

    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.empty((0, dim))
        self.metadata = []

    def add(self, vector, metadata):
        vector = np.asarray(vector, dtype=float).reshape(1, self.dim)
        self.vectors = np.vstack([self.vectors, vector])
        self.metadata.append(metadata)

    def search(self, query, k=3):
        query = np.asarray(query, dtype=float)
        dots = self.vectors @ query
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query)
        scores = dots / norms
        top_idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.metadata[i]) for i in top_idx]

store = VectorStore(dim=DIM)
for doc in documents:
    store.add(make_vector(doc["category"]), {"text": doc["text"], "category": doc["category"], "year": doc["year"]})

print(f"Stored {len(store.metadata)} vectors of dimension {store.dim}")

query = make_vector("python", noise=0.1)
results = store.search(query, k=4)
print("\nQuery: something close to the 'python' cluster\n")
for score, meta in results:
    print(f"  {score:.3f}  [{meta['category']:7s}]  {meta['text']}")
```

```
Stored 12 vectors of dimension 8

Query: something close to the 'python' cluster

  0.977  [python ]  Decorators wrap a function with extra behavior.
  0.952  [python ]  Virtual environments keep dependencies isolated.
  0.932  [python ]  Python's list comprehensions replace many for-loops.
  0.921  [python ]  Type hints catch bugs before you run the code.
```

That's `add` and `search` — the entire public surface of a vector
database, at the conceptual core. Everything a real one adds on top
(persistence to disk, an ANN index instead of the brute-force `@`, filter
support, distribution) is engineering around this same idea, not a
different idea.

Now the O(n) claim from "The concept," measured for real: how does search
time change as the number of stored vectors grows, at a realistic
embedding size (384 dimensions, the same as lesson 26's model)?

```python
import time

np.random.seed(0)

def brute_force_search_time(n, dim=384):
    vectors = np.random.randn(n, dim)
    query = np.random.randn(dim)
    start = time.perf_counter()
    scores = np.einsum("ij,j->i", vectors, query) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    )
    top5 = np.argsort(-scores)[:5]
    return time.perf_counter() - start

for n in [1_000, 10_000, 100_000, 500_000]:
    elapsed = brute_force_search_time(n)
    print(f"N={n:>7,}  {elapsed * 1000:7.2f} ms")
```

```
N=  1,000     0.42 ms
N= 10,000     3.98 ms
N=100,000    48.08 ms
N=500,000   318.80 ms
```

The exact milliseconds depend on the machine this runs on (a laptop CPU is
not a data-center CPU), but the *shape* is the point and it will reproduce
anywhere: roughly a 10x jump in vectors produces roughly a 10x jump in
time, all the way up. That's `O(n)` made visible, not asserted — and it's
exactly the curve an ANN index (HNSW, IVF) is built to flatten.

Here's the same `add`/`search` idea against a real vector database
library, `chromadb`, so the shape of a production API is visible
side-by-side with the NumPy version above:

```python
# ILLUSTRATIVE -- requires `pip install chromadb`. Not executed here;
# the NumPy version above is the real, run, verified centerpiece.
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

collection.add(
    ids=["1", "2", "3"],
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.1, 0.9], [0.2, 0.8, 0.1]],
    metadatas=[{"category": "python"}, {"category": "baking"}, {"category": "finance"}],
    documents=[
        "Python's list comprehensions replace many for-loops.",
        "Proof the dough until it doubles in size.",
        "Diversifying a portfolio reduces risk.",
    ],
)

results = collection.query(
    query_embeddings=[[0.12, 0.19, 0.28]],
    n_results=2,
    where={"category": "python"},   # metadata filter, built in
)
# results["documents"] would hold the matching text -- not shown, not run.
```

Same shape as the hand-rolled version (`add` a vector plus metadata,
`query`/`search` for the closest matches) with two things the NumPy class
above doesn't have: an ANN index under the hood instead of brute force,
and a `where` filter baked into the query itself instead of something you
write yourself.

## Build this

Extend the `VectorStore` class above with metadata filtering: add a
`filter` parameter to `search` (a dict of exact-match key/value pairs, or
a predicate function — your choice) so a query can ask for "the closest
vectors where `category == "finance"`" and only ever rank vectors that
pass the filter, the same feature `chromadb`'s `where` gives you natively.
Test it with a query vector built near one cluster but a filter that
excludes that cluster, and confirm the results come back from a different
category entirely, correctly filtered.

**Stretch:** extend the timing experiment from "In code" — add larger
values of `N` (a few million, if your machine can hold it in memory) and
plot or tabulate the growth. Then look up FAISS's `IndexHNSWFlat` or
`IndexIVFFlat` (no need to install anything unless you want to), read what
parameters they expose (like the number of graph neighbors in HNSW, or the
number of clusters in IVF), and connect each parameter back to the
accuracy-versus-speed trade-off described in "The concept."

## Go deeper

- [Pinecone: What is a Vector Database?](https://www.pinecone.io/learn/vector-database/) — a clear, practical overview of what a vector database adds beyond a raw index.
- [Pinecone: Hierarchical Navigable Small Worlds (HNSW)](https://www.pinecone.io/learn/series/faiss/hnsw/) — the clearest visual explanation available of how HNSW actually navigates toward nearest neighbors.
- [Chroma documentation](https://docs.trychroma.com/) — the real API for the library shown as ILLUSTRATIVE above, including persistence and filtering.
- [pgvector on GitHub](https://github.com/pgvector/pgvector) — adding vector search to a Postgres database you may already run.
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) — the library most other vector databases build their ANN indexes on top of.

**Next:** [Structured Outputs & Function Calling](42-structured-outputs.md)
</content>
