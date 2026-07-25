---
title: "Embeddings"
stage: 5
order: 26
minutes: 45
difficulty: intermediate
prerequisites: ["how-llms-work"]
tags: ["embeddings", "semantic-search"]
summary: "Text as vectors, cosine similarity, embedding models, vector databases, and chunking strategy, with a real semantic search built in NumPy."
---

# Embeddings

## Why this matters

Transformers turn tokens into vectors as an internal implementation detail
on the way to predicting the next token. This lesson takes that same idea
— text as vectors — and repurposes it as a standalone tool: a model whose
entire job is to turn a piece of text into one vector that captures its
meaning, arranged so that "these two texts mean similar things" becomes
"these two vectors point in similar directions" — a question computers
answer with a handful of multiplications. That single move is the
foundation under semantic search, deduplication, recommendation, and,
directly, next lesson's topic: retrieval-augmented generation.

## The concept

**Text as vectors.** An **embedding model** maps any string — a word, a
sentence, a paragraph — to a fixed-length vector of floating-point numbers
(384 numbers for the model used below; other models use different fixed
sizes). The model is trained so that texts with similar meaning land near
each other in that vector space, and texts with unrelated meaning land far
apart. This differs from the per-token embeddings inside an LLM's first
layer (one vector per token, learned as a byproduct of next-token
prediction) — an embedding model here is trained specifically so a
*whole-text* vector is meaningful on its own, typically via contrastive
training: shown many pairs of texts labeled similar or dissimilar, and
pushed to place similar pairs' vectors close together and dissimilar
pairs' vectors far apart.

**Cosine similarity.** Given two embedding vectors, the standard way to
measure "how similar" is **cosine similarity**: the cosine of the angle
between them, computed as `dot(a, b) / (norm(a) * norm(b))`. It ranges from
-1 to 1 in general, and in practice lands mostly between 0 and 1 for the
kind of models used here — 1 means the vectors point in exactly the same
direction (same meaning), 0 means unrelated. The reason cosine similarity
rather than, say, plain Euclidean distance is the standard choice: these
models are trained so that *direction* carries the meaning signal, not
vector length — cosine similarity looks only at direction and ignores
magnitude entirely, which matches how the model was actually trained to
encode meaning.

**Embedding models.** There's a landscape here, not one model: hosted API
embedding models from LLM providers, and open, locally-runnable models —
the `sentence-transformers` family (used below) is the standard entry
point, with models ranging from tiny-and-fast (like the 384-dimension
model this lesson uses) to large-and-precise. Bigger models generally
capture meaning more precisely at the cost of more compute and slower
embedding calls; for a lot of real applications, a small model is more
than good enough, and the numbers below back that up directly.

**Vector databases.** Once you have thousands, or billions, of embeddings,
brute-force cosine similarity — compare the query vector against literally
every stored vector — gets slow. **Vector databases** (FAISS as a library
you embed in your own process; Pinecone, Weaviate, pgvector, and others as
services or database extensions) use **approximate nearest-neighbor
(ANN)** index structures to find the top-k most similar vectors far faster
than brute force, trading a small amount of recall for a large speed gain.
At the scale in this lesson — dozens of documents — brute-force NumPy
isn't a simplification of what a vector database does, it *is* what a
vector database reduces to under load-free conditions; the index
structures only start to matter once brute force actually becomes the
bottleneck.

**Chunking strategy.** You almost never embed an entire document as one
vector — cramming a whole document's content into a single fixed-length
vector averages away the specific details that make retrieval useful, the
same way summarizing a book in one sentence loses which page has the fact
you need. Instead, documents are split into smaller **chunks** — by
paragraph, by a fixed size in characters or tokens, or by more sophisticated
semantic boundaries — and each chunk gets its own vector, so a search can
point at the specific passage that's actually relevant. The trade-off:
chunks that are too small may lack enough surrounding context to be useful
on their own; chunks that are too large dilute the vector (and waste
downstream context budget) with content the query wasn't actually about.
Next lesson goes deeper on chunk sizing specifically, because it's the
single highest-leverage knob in a RAG pipeline.

## In code

Real embeddings, real cosine similarity, computed offline after a one-time
model download — `pip install sentence-transformers` (installs PyTorch as
a dependency; the model itself, `all-MiniLM-L6-v2`, is a small ~90MB
download from Hugging Face the first time you run this, then fully
offline). Eight short documents across three topics, one query, ranked by
cosine similarity computed by hand with NumPy — no library similarity
function used, since seeing the formula run for real is the point:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

np.set_printoptions(precision=3, suppress=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Python is a popular programming language.",
    "The stock market fell sharply today.",
    "Cats are independent pets.",
    "JavaScript runs in the browser.",
    "Interest rates rose again this quarter.",
    "A puppy is a young dog.",
]

query = "programming languages for beginners"

doc_embeddings = model.encode(docs)
query_embedding = model.encode([query])[0]

print("Embedding shape per document:", doc_embeddings[0].shape)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

scores = [cosine_similarity(query_embedding, d) for d in doc_embeddings]
ranked = sorted(zip(docs, scores), key=lambda x: -x[1])

print(f"\nQuery: {query!r}\n")
print("Ranked by cosine similarity:")
for doc, score in ranked:
    print(f"  {score:.3f}  {doc}")
```

```
Embedding shape per document: (384,)

Query: 'programming languages for beginners'

Ranked by cosine similarity:
  0.535  Python is a popular programming language.
  0.175  JavaScript runs in the browser.
  0.038  A puppy is a young dog.
  0.037  Cats are independent pets.
  0.034  Dogs are loyal companions.
  0.028  The stock market fell sharply today.
  0.026  Interest rates rose again this quarter.
  0.017  The cat sat on the mat.
```

Two documents about programming rank at the top, cleanly separated from
everything else, purely from vector direction — nobody told the model
"these two are about the same topic." A direct spot check makes the
"direction encodes meaning" claim from "The concept" completely concrete:

```python
print("\nSpot check -- two cat sentences vs. a finance sentence:")
a = model.encode("The cat sat on the mat.")
b = model.encode("Cats are independent pets.")
c = model.encode("The stock market fell sharply today.")
print("  cat/cat similarity:    ", round(cosine_similarity(a, b), 3))
print("  cat/finance similarity:", round(cosine_similarity(a, c), 3))
```

```
Spot check -- two cat sentences vs. a finance sentence:
  cat/cat similarity:     0.391
  cat/finance similarity: 0.075
```

Two sentences that never share a word beyond "cat"/"cats" land over five
times closer together than two sentences about unrelated topics — real
numbers, not an illustration.

## Build this

Build a semantic search over roughly 50 documents, computing cosine
similarity in NumPy yourself (no library similarity function — write the
same `cosine_similarity` function from "In code" from memory). Start from
this 50-document corpus, five topics of ten sentences each:

```python
topics = {
    "animals": ["Cats are independent pets.", "Dogs love to play fetch.", "Elephants have excellent memory.",
                "Sharks have existed for millions of years.", "Parrots can mimic human speech.",
                "Wolves hunt in coordinated packs.", "Owls can rotate their heads nearly 270 degrees.",
                "Penguins huddle together to stay warm.", "Bees communicate through dance.",
                "Octopuses can change color to camouflage."],
    "programming": ["Python is popular for data science.", "JavaScript runs in every web browser.",
                "Rust guarantees memory safety without a garbage collector.", "SQL is used to query relational databases.",
                "Git tracks changes to source code over time.", "APIs let programs talk to each other.",
                "Recursion is a function calling itself.", "Compilers translate code into machine instructions.",
                "Unit tests catch regressions early.", "Version control prevents lost work."],
    "finance": ["The stock market fell sharply today.", "Interest rates rose again this quarter.",
                "Inflation erodes purchasing power over time.", "Diversification reduces investment risk.",
                "A mortgage is a loan secured by real estate.", "Compound interest grows savings exponentially.",
                "Central banks set monetary policy.", "Bonds are generally less volatile than stocks.",
                "A budget tracks income against expenses.", "Credit scores affect loan interest rates."],
    "cooking": ["Searing meat locks in flavor.", "Basil pairs well with tomatoes.", "Kneading develops gluten in bread dough.",
                "A roux thickens sauces and soups.", "Fresh herbs are added at the end of cooking.",
                "Marinating tenderizes tougher cuts of meat.", "Caramelizing onions takes low heat and time.",
                "Salt enhances almost every savory dish.", "Resting meat after cooking keeps it juicy.",
                "Fermentation preserves food and adds flavor."],
    "sports": ["A marathon is 26.2 miles long.", "Soccer is the most popular sport worldwide.",
                "Basketball teams have five players on court.", "Swimming builds full-body endurance.",
                "Tennis matches are scored in sets and games.", "Cycling is popular for both sport and commuting.",
                "Baseball games have nine innings.", "Yoga improves flexibility and balance.",
                "Weightlifting builds muscular strength.", "Chess is considered a competitive mental sport."],
}
docs = [d for group in topics.values() for d in group]
```

Embed all 50 with `sentence-transformers`, then run at least three queries
of your own — including at least one that doesn't obviously name a topic
(something like "how do I get better at chess" should surface the sports
documents even though it never says "sports"). Print the top 5 results per
query with their scores. Then find one genuinely surprising result: a case
where the top match isn't from the category you'd have guessed, and write
one sentence hypothesizing why the embedding model likely put those two
pieces of text close together.

**Stretch:** for one query, rank and print similarity scores for all 50
documents, not just the top few, and look for the point where the score
drops off sharply between clearly-relevant and clearly-irrelevant results.
That gap is an intuitive, data-driven way to choose a similarity threshold
or a `k` for retrieval, instead of guessing a round number.

## Go deeper

- [`sentence-transformers/all-MiniLM-L6-v2` on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — the exact model used in this lesson's code, with its training data and benchmark numbers.
- [Pinecone: What are Vector Embeddings?](https://www.pinecone.io/learn/vector-embeddings/) — a clear conceptual introduction, with more on how embedding models are trained.
- [Jay Alammar: The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) — the classic visual explanation of how word vectors capture meaning, still the clearest starting point for the intuition.
- [sentence-transformers documentation](https://www.sbert.net/) — the library used in this lesson, including its full catalog of models and their size/accuracy trade-offs.

**Next:** [Retrieval-Augmented Generation (RAG)](27-rag.md)
