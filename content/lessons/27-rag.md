---
title: "Retrieval-Augmented Generation (RAG)"
stage: 5
order: 27
minutes: 60
difficulty: intermediate
prerequisites: ["embeddings", "prompt-engineering"]
tags: ["rag", "retrieval"]
summary: "Why retrieval beats stuffing the context window, the ingest-chunk-embed-retrieve-generate pipeline, chunk sizing, and common RAG failure modes."
---

# Retrieval-Augmented Generation (RAG)

## Why this matters

Two lessons come together here. Prompt engineering (lesson 25) is about
making the most of whatever's in the context window. Embeddings (lesson
26) let you find the most relevant text out of a large corpus by vector
similarity. **RAG** is the combination: instead of hand-writing everything
the model might need into a prompt — impossible once your knowledge base
is bigger than a context window, which is almost immediately — or baking
new knowledge into the model's weights (heavy, and lesson 28's territory),
you retrieve only the passages relevant to the current question, at query
time, and put just those into the prompt. It's the single most common way
real applications give an LLM access to information it wasn't trained on.

## The concept

**Why retrieval beats stuffing the context window.** Even with today's
very large context windows, more context isn't free in three separate
ways. It costs money and latency — every token you send is billed and
takes time to process, on every single call. It doesn't reliably get
used — models have a well-documented tendency to use information placed
in the middle of a long context less reliably than information near the
beginning or end (sometimes called "lost in the middle"), so padding the
prompt with mostly-irrelevant material can make the model *worse* at
finding the one relevant fact, not just slower. And most real knowledge
bases — a company wiki, a codebase, a year of support tickets, your own
notes — simply don't fit in any context window regardless of size.
Retrieval solves all three at once: pull in only what's actually relevant
to *this* question, keeping the prompt small, cheap, and — because
there's less irrelevant material competing for the model's attention —
more likely to be used correctly.

**The pipeline: ingest, chunk, embed, retrieve, generate.** Five stages,
in order. **Ingest** — gather your source material: notes, docs, PDFs,
code, web pages, whatever the knowledge base actually is. **Chunk** —
split each source document into smaller passages (lesson 26 covered why:
one vector per whole document loses precision). **Embed** — run every
chunk through an embedding model and store the resulting vector alongside
the original chunk text, in an index — a NumPy array in this lesson, a
real vector database in production. **Retrieve** — at query time, embed
the user's question with the *same* embedding model used to build the
index, compute similarity between the question's vector and every stored
chunk vector, and take the top-k most similar. **Generate** — build a
prompt containing the retrieved chunks as context plus the original
question, instruct the model to answer using that context, and send it to
an LLM. Ingest through embed happen once (or whenever the source material
changes); retrieve and generate happen on every single query.

**Chunk sizing.** The trade-off from last lesson, sharpened: chunks that
are too small (a single sentence) retrieve with high precision but may not
be self-contained enough to actually answer the question once retrieved —
the fact is split across two chunks and you only got one. Chunks that are
too large (a full document) dilute the embedding's specificity and burn
prompt budget on content the question wasn't about, re-creating the
context-stuffing problem this lesson opened with, just per-chunk instead
of per-document. A common practical starting point is a few hundred
tokens per chunk, sometimes with a small overlap between consecutive
chunks so information sitting right at a chunk boundary isn't orphaned in
neither chunk. And there's a more basic failure mode than "picking the
wrong number": chunking **naively** — slicing raw text into fixed-size
windows without respecting sentence or paragraph boundaries — can
literally cut a word or a sentence in half at the boundary, corrupting the
exact passage you needed. This isn't hypothetical; the "In code" demo
below hits this bug for real, unplanned, on the very first run.

**Common failure modes.** Bad chunking (boundary cuts, wrong size) returns
technically-similar but practically-useless passages. Retrieval can return
plausible-but-wrong chunks, because embedding similarity measures semantic
closeness, not "actually answers the question" — the two usually line up
but aren't the same thing. The model can ignore retrieved context entirely
and answer from its own training data instead, which may be wrong or
stale — a good RAG prompt has to explicitly instruct the model to rely on
the provided context, not just hope it will. The index can go stale if
source documents change but you don't re-ingest and re-embed them. Getting
`k` wrong in either direction hurts: too few retrieved chunks and the
answer's actual evidence might be split across chunks you didn't retrieve;
too many and you're back to the stuffing problem. And the one that matters
most to get right conceptually: **retrieval reduces hallucination by
grounding the answer in real, retrieved text — it does not eliminate it.**
The model can still misread a passage, over-generalize from it, or add a
plausible-sounding detail the passage never actually said, for exactly the
mechanism described in lesson 24 — it's still predicting a plausible
continuation, now conditioned on better evidence, not running a fact
-checker. Grounded generation is a real, measurable improvement over
unconstrained recall; it is not a guarantee.

## In code

A complete, real, runnable ingest → chunk → embed → retrieve pipeline over
a small set of personal notes (`pip install sentence-transformers`, same
model as last lesson). First, chunking by paragraph, with a max size as a
fallback for any paragraph that runs long:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

notes = """
Project Aurora kicked off in March. The goal is to migrate the billing
service from a monolith to a set of small services. The team agreed to
split the work into three phases: extraction, dual-write, and cutover.

During the extraction phase, we copy the billing code into its own
repository without changing behavior. This phase is expected to take
two sprints and has no user-facing risk.

The dual-write phase is the riskiest part of the project. Both the old
and new billing paths write to the database at the same time, and we
compare their outputs nightly. Any mismatch blocks the next phase.

Cutover happens once dual-write has run clean for two full weeks. At
that point we flip a feature flag and the new service becomes the
source of truth. The old billing code stays in place for one more
release as a rollback path.

The team decided against a big-bang rewrite after a postmortem on
Project Halcyon, an earlier migration that failed because it tried to
change the database schema and the service boundary at the same time.
"""

def chunk_text(text, max_chars=220):
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        p = " ".join(p.split())
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars])
    return chunks

chunks = chunk_text(notes)
print(f"Split notes into {len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}] ({len(c)} chars) {c[:70]}...")
```

```
Split notes into 6 chunks:

[0] (212 chars) Project Aurora kicked off in March. The goal is to migrate the billing...
[1] (176 chars) During the extraction phase, we copy the billing code into its own rep...
[2] (206 chars) The dual-write phase is the riskiest part of the project. Both the old...
[3] (220 chars) Cutover happens once dual-write has run clean for two full weeks. At t...
[4] (8 chars) ck path....
[5] (205 chars) The team decided against a big-bang rewrite after a postmortem on Proj...
```

Look at chunk `[4]`: eight characters, `ck path....`. The "rollback"
paragraph is 228 characters — 8 over the 220-character limit — so the
`max_chars` fallback sliced it mid-word, right through "rollback," leaving
`rolla...` — sorry, `rollba` — as the tail of chunk `[3]` and the orphaned
fragment `ck path....` as chunk `[4]`. This is the exact naive-chunking bug
described in "The concept," caught here for real, not staged.

Now embed the chunks and retrieve for two real questions:

```python
chunk_embeddings = model.encode(chunks)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(question, k=2):
    q_emb = model.encode(question)
    scores = [cosine_similarity(q_emb, c) for c in chunk_embeddings]
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
    return ranked[:k]

for question in ["Why did the team avoid a big-bang rewrite?", "When does cutover happen?"]:
    print(f"\n\nQuestion: {question!r}")
    top = retrieve(question, k=2)
    for chunk, score in top:
        print(f"  score={score:.3f}  {chunk}")
```

```
Question: 'Why did the team avoid a big-bang rewrite?'
  score=0.604  The team decided against a big-bang rewrite after a postmortem on Project Halcyon, an earlier migration that failed because it tried to change the database schema and the service boundary at the same time.
  score=0.200  Cutover happens once dual-write has run clean for two full weeks. At that point we flip a feature flag and the new service becomes the source of truth. The old billing code stays in place for one more release as a rollba

Question: 'When does cutover happen?'
  score=0.392  Cutover happens once dual-write has run clean for two full weeks. At that point we flip a feature flag and the new service becomes the source of truth. The old billing code stays in place for one more release as a rollba
  score=0.208  Project Aurora kicked off in March. The goal is to migrate the billing service from a monolith to a set of small services. The team agreed to split the work into three phases: extraction, dual-write, and cutover.
```

Retrieval works — both questions' top hits are genuinely the right chunk
— but look closely at chunk `[3]`, the cutover/rollback passage: it shows
up as a result for *both* questions, and both times it's cut off mid-word
at "rollba", because it's the chunk the boundary bug damaged. This system
is shipped with a real, live bug: it happens to still answer these two
questions correctly, but any question whose answer depends specifically on
the word "rollback," or on the orphaned eight-character chunk `[4]`, will
fail. It's left unfixed here on purpose — lesson 30 builds an eval suite
that catches exactly this, the way it would in a real project, instead of
a human noticing by eye.

Building the final grounded prompt is just string construction — no model
call needed to see the shape of it:

```python
def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(f"- {chunk}" for chunk, _score in retrieved_chunks)
    return f"""Answer the question using ONLY the context below. If the context
doesn't contain the answer, say you don't have enough information --
do not guess.

Context:
{context}

Question: {question}
Answer:"""

top = retrieve("When does cutover happen?", k=2)
print(build_prompt("When does cutover happen?", top))
```

```
Answer the question using ONLY the context below. If the context
doesn't contain the answer, say you don't have enough information --
do not guess.

Context:
- Cutover happens once dual-write has run clean for two full weeks. At that point we flip a feature flag and the new service becomes the source of truth. The old billing code stays in place for one more release as a rollba

- Project Aurora kicked off in March. The goal is to migrate the billing service from a monolith to a set of small services. The team agreed to split the work into three phases: extraction, dual-write, and cutover.

Question: When does cutover happen?
Answer:
```

That's the real, complete prompt a generation call would receive — including,
honestly, the "rollba" truncation artifact flowing straight through from
retrieval into the context the model would see. The instruction to answer
*only* from the provided context, and to say so plainly when the context
doesn't cover the question, is the RAG-level mitigation for lesson 24's
hallucination mechanism — it doesn't eliminate the risk, but it gives the
model an explicit, honest alternative to guessing.

Sending that prompt to a real model needs your own API key — shown here as
idiomatic code, **not run**, no response text shown as if captured. Using
the [Anthropic Python SDK](https://pypi.org/project/anthropic/):

```python
# ILLUSTRATIVE -- requires your own ANTHROPIC_API_KEY. Not executed here.
import anthropic

client = anthropic.Anthropic()

prompt = build_prompt("When does cutover happen?", retrieve("When does cutover happen?", k=2))

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)
# response.content[0].text would hold the grounded answer -- not shown,
# since we have no key here to actually produce one.
```

## Build this

Build a working RAG system over your own notes: a folder of real text
files, or a handful of paragraphs pasted into a Python string like the
demo above. Chunk them, embed with `sentence-transformers`, write your own
`retrieve(question, k)` function, and build the final grounded prompt
string — reuse the pieces above, but deliberately check whether your
chunker has the same mid-word boundary bug this lesson's demo does, and
decide whether to fix it or leave it (either is a valid choice to make
consciously). Then, with your own Anthropic (or other provider's) API key,
actually send the prompt and read the real answer. Compare it against what
you know from your own notes to be true, and write down any place the
model added a detail your notes didn't actually contain — that's
hallucination on top of correct retrieval, happening for real.

**Stretch:** vary `k` and observe how the retrieved context changes as you
retrieve more or fewer chunks — and, if you're running it against a real
model, whether the final answer changes too. Then ask a question your
notes genuinely don't answer, and check whether your prompt (and the
model) correctly says "I don't have enough information" instead of
guessing.

## Go deeper

- [Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) — the original 2020 paper that named and formalized this pattern.
- [Pinecone: Retrieval Augmented Generation (RAG)](https://www.pinecone.io/learn/retrieval-augmented-generation/) — a practical overview of the pipeline and common production patterns.
- [Anthropic: Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — a real technique for meaningfully improving retrieval accuracy over the naive pipeline built in this lesson.
- [LangChain: RAG tutorial](https://docs.langchain.com/oss/python/langchain/rag) — the same five-stage pipeline built with a popular framework, useful once you're ready to move past hand-rolled NumPy.

**Next:** [Fine-tuning](28-fine-tuning.md)
