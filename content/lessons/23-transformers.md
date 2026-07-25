---
title: "Transformers"
stage: 4
order: 23
minutes: 70
difficulty: advanced
prerequisites: ["sequence-models"]
tags: ["deep-learning", "transformers", "attention"]
summary: "Attention as a learned lookup over query, key and value, self-attention and multi-head attention, positional encoding, and the encoder/decoder split."
---

# Transformers

## Why this matters

Last lesson ended on a wish list: a mechanism that lets a model look
directly at any earlier position in a sequence, on demand, without forcing
information through a long chain of sequential recurrent steps. **Attention**
is that mechanism, and the **Transformer** architecture built entirely
around it — no recurrence at all — is the architecture behind every large
language model this roadmap will cover from here on. This lesson works
through attention with an actual worked numerical example, small enough to
follow by hand, and then implements it in PyTorch. Everything in Stage 5 —
how LLMs work, prompting, embeddings, fine-tuning — assumes you understand
what's computed in this lesson.

## The concept

**Attention is a learned, differentiable lookup.** A regular dictionary
lookup takes a query, finds the matching key exactly, and returns its
value. Attention softens every step of that: instead of an exact match, it
scores the query against *every* key by similarity; instead of returning
one value, it returns a weighted blend of *all* the values, weighted by
those similarity scores. "Learned" because the query, key, and value
vectors themselves come from weight matrices trained by backpropagation,
same as every other weight in this roadmap.

**Query, key, and value.** For each token's embedding vector `x`, three
learned weight matrices project it into three new vectors: `q = x @ Wq`
(the **query**: what this token is looking for), `k = x @ Wk` (the
**key**: what this token offers to be matched against, by any token's
query), and `v = x @ Wv` (the **value**: what this token actually
contributes to the output, once it's been matched). All three come from the
*same* input embedding, just projected differently — `Wq`, `Wk`, and `Wv`
start random and are learned like any other weights.

**Self-attention: every token attends to every token, including
itself.** For a sequence of tokens, stack every token's query into a matrix
`Q`, every key into `K`, every value into `V`. The similarity between token
`i`'s query and token `j`'s key is their dot product, `Q @ K^T` — computing
every pairwise similarity in the sequence in one matrix multiplication.
Divide by `sqrt(d_k)` (the dimension of the key vectors) — without this
scaling, dot products grow larger as `d_k` grows, pushing the next step's
softmax into a region where its gradient is nearly flat, which would slow
learning. Apply **softmax** across each row, turning that row's raw
similarity scores into a probability distribution — nonnegative, summing to
1 — over "how much attention this token pays to every other token,
including itself." Multiply those attention weights by `V`: each token's
output is a weighted sum of every token's value vector, weighted by how
much it attended to each one. In one line: `Attention(Q, K, V) =
softmax(Q @ K^T / sqrt(d_k)) @ V`.

**A worked example, three tokens, by hand.** Take three token embeddings
(`X`, 3×4) and three small weight matrices (`Wq`, `Wk`, `Wv`, each 4×4).
Project: `Q = X @ Wq`, `K = X @ Wk`, `V = X @ Wv` — each now 3×4, one
query/key/value vector per token. Compute raw scores `Q @ K^T` (3×3 — one
score per pair of tokens), scale by `1/sqrt(4) = 0.5`, and softmax each row.
The "In code" section runs this exact computation and prints every
intermediate matrix; here are the final attention weights it produces:

```
                 attends to tok1   attends to tok2   attends to tok3
token 1 output:       0.509             0.196             0.295
token 2 output:       0.092             0.422             0.486
token 3 output:       0.233             0.104             0.663
```

Each row sums to 1 (a probability distribution) and each row is different —
token 1 attends mostly to itself (0.509) with some attention to token 3
(0.295); token 2 attends most to token 3 (0.486); token 3 attends
overwhelmingly to itself (0.663). Multiplying these weights by `V` produces
token 1's output as `0.509 * v1 + 0.196 * v2 + 0.295 * v3` — a genuinely
different blend of the same three value vectors for every token, computed
from nothing but dot-product similarity and a softmax.

**Multi-head attention runs several attention computations in parallel.**
One set of `Wq`, `Wk`, `Wv` gives the model one *way* of relating tokens to
each other. **Multi-head attention** splits the embedding dimension into
several smaller chunks (say, 8 heads of dimension `d_model / 8` each),
runs the exact same scaled dot-product attention independently in each
chunk with its *own* learned `Wq`, `Wk`, `Wv`, and concatenates the results
back together. In practice, different heads often learn to specialise — one
might track nearby words, another long-range syntactic relationships,
another something not easily nameable — the same way different filters in
a CNN specialise in different visual patterns.

**Positional encoding: why attention needs to be told about order.** Look
back at the attention formula: nothing in `softmax(Q @ K^T / sqrt(d_k)) @
V` refers to a token's *position* in the sequence — only to its content,
through the query/key/value projections. Permute the input tokens and
self-attention produces the exact same set of output vectors, just
reordered to match — it is, on its own, completely order-agnostic (the
"In code" section verifies this directly, permuting a real input and
showing the outputs match once un-permuted). That's a problem, since order
plainly carries meaning ("the dog bit the man" again). The fix used in the
original Transformer: compute a fixed vector per position using sine and
cosine at different frequencies (no learned weights involved), and *add*
it to each token's embedding before attention ever runs. Now two identical
words in different positions have different input vectors, and attention
has something position-dependent to work with. (Some later Transformer
variants use *learned* positional embeddings instead of fixed sine/cosine
ones — the mechanism differs, but the reason one is needed at all is
exactly this lesson's point.)

**The encoder/decoder split.** The original Transformer paper used two
stacks. The **encoder** processes an entire input sequence with
self-attention, where every token can attend to every other token in both
directions — good for building a rich representation of a complete input,
as in translation's source sentence. The **decoder** generates output one
token at a time and uses **masked** self-attention — each position is only
allowed to attend to positions at or before it, never ahead — because at
generation time, the tokens after the current one don't exist yet; masking
enforces during training the same constraint that's unavoidable at
inference time. The decoder also attends to the encoder's output through a
separate **cross-attention** step (queries from the decoder, keys and
values from the encoder), letting each generated token look back at the
entire input. Modern large language models mostly use a decoder-only
Transformer — just the masked self-attention stack, trained to predict the
next token — which is what Stage 5's "How LLMs Work" builds on directly.

## In code

The worked numerical example above, computed and printed step by step —
every matrix shown here is real, executed output, not hand-calculated:

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# Three tokens, each already embedded as a 4-dim vector
X = np.array([
    [1.0, 0.0, 1.0, 0.0],   # token 1: "the"
    [0.0, 1.0, 0.0, 1.0],   # token 2: "cat"
    [1.0, 1.0, 0.0, 0.0],   # token 3: "sat"
])
print("X (3 tokens x 4-dim embedding):\n", X)

# Learned weight matrices that project embeddings into query/key/value space
# (fixed seed so these numbers are reproducible -- in a real model these start
# random like this and are then LEARNED by gradient descent, same as every
# other weight matrix in this roadmap)
rng = np.random.default_rng(0)
Wq = np.round(rng.normal(0, 1, size=(4, 4)), 1)
Wk = np.round(rng.normal(0, 1, size=(4, 4)), 1)
Wv = np.round(rng.normal(0, 1, size=(4, 4)), 1)

Q = X @ Wq   # what each token is "looking for"
K = X @ Wk   # what each token "offers" to be matched against
V = X @ Wv   # what each token actually "sends" once matched

print("\nQ = X @ Wq (queries):\n", Q)
print("\nK = X @ Wk (keys):\n", K)
print("\nV = X @ Wv (values):\n", V)

d_k = K.shape[1]
raw_scores = Q @ K.T
print(f"\nraw scores Q @ K^T (d_k={d_k}):\n", raw_scores)

scaled_scores = raw_scores / np.sqrt(d_k)  # keeps the softmax from saturating as d_k grows
print("\nscaled scores (divided by sqrt(d_k)):\n", scaled_scores)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

weights = softmax(scaled_scores, axis=-1)  # each ROW sums to 1
print("\nattention weights (softmax over each row):\n", weights)
print("\nrow sums (should be 1.0):", weights.sum(axis=1))

output = weights @ V   # weighted sum of values, weighted by attention
print("\noutput = weights @ V (3 tokens x 4-dim, one output vector per token):\n", output)
```

```
X (3 tokens x 4-dim embedding):
 [[1. 0. 1. 0.]
 [0. 1. 0. 1.]
 [1. 1. 0. 0.]]

Q = X @ Wq (queries):
 [[-0.6 -1.4  0.   0.1]
 [-2.8  0.2  0.1  0.2]
 [-0.4  0.3  1.9  1. ]]

K = X @ Wk (keys):
 [[ 0.4 -0.2 -0.3  0.1]
 [-0.6  1.6 -1.7  0.2]
 [-0.6  1.1 -0.3  1.4]]

V = X @ Wv (values):
 [[-1.5  2.   1.5  1.2]
 [-0.4 -0.4  2.3  3.5]
 [-0.9  0.4  1.   1.9]]

raw scores Q @ K^T (d_k=4):
 [[ 0.05 -1.86 -1.04]
 [-1.17  1.87  2.15]
 [-0.69 -2.31  1.4 ]]

scaled scores (divided by sqrt(d_k)):
 [[ 0.025 -0.93  -0.52 ]
 [-0.585  0.935  1.075]
 [-0.345 -1.155  0.7  ]]

attention weights (softmax over each row):
 [[0.509 0.196 0.295]
 [0.092 0.422 0.486]
 [0.233 0.104 0.663]]

row sums (should be 1.0): [1. 1. 1.]

output = weights @ V (3 tokens x 4-dim, one output vector per token):
 [[-1.107  1.058  1.509  1.857]
 [-0.744  0.21   1.595  2.511]
 [-0.988  0.69   1.251  1.903]]
```

Scaled dot-product attention as a general PyTorch function — this is
approximately the exercise below, cross-checked against PyTorch's own
built-in implementation to confirm it's correct:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)   # (..., seq_len_q, seq_len_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)                 # attention weights, rows sum to 1
    return weights @ V, weights                          # (..., seq_len_q, d_v)

batch, seq_len, d_k, d_v = 2, 3, 8, 8
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Q shape:     ", Q.shape)
print("output shape:", output.shape)
print("weights shape:", weights.shape)
print("weights row sums (should all be 1.0):", weights.sum(dim=-1))

ref = F.scaled_dot_product_attention(Q, K, V)
print("\nmatches torch's built-in F.scaled_dot_product_attention:",
      torch.allclose(output, ref, atol=1e-6))
```

```
Q shape:      torch.Size([2, 3, 8])
output shape: torch.Size([2, 3, 8])
weights shape: torch.Size([2, 3, 3])
weights row sums (should all be 1.0): tensor([[1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000]])

matches torch's built-in F.scaled_dot_product_attention: True
```

Multi-head self-attention with `nn.MultiheadAttention`, plus a direct check
that attention alone really is order-agnostic — permute the input tokens,
undo the permutation on the output, and compare to the un-permuted run:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

d_model, n_heads, seq_len, batch = 16, 4, 3, 1
mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

x = torch.randn(batch, seq_len, d_model)
out, attn_weights = mha(x, x, x)   # self-attention: query, key, value all come from x
print("input shape: ", x.shape)
print("output shape:", out.shape, " (same shape as input -- attention preserves shape)")

perm = torch.tensor([2, 0, 1])
x_permuted = x[:, perm, :]
out_permuted, _ = mha(x_permuted, x_permuted, x_permuted)

out_unpermuted = torch.empty_like(out_permuted)
out_unpermuted[:, perm, :] = out_permuted

print("\nmax difference between original output and un-permuted permuted-output:",
      (out - out_unpermuted).abs().max().item())
```

```
input shape:  torch.Size([1, 3, 16])
output shape: torch.Size([1, 3, 16])  (same shape as input -- attention preserves shape)

max difference between original output and un-permuted permuted-output: 5.960464477539063e-08
```

That difference is floating-point noise, not signal — reordering the input
and then reordering the output back produces the *same* result. Attention
by itself genuinely cannot tell position 0 from position 2; positional
encoding exists to fix exactly this:

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])   # even dims: sine
    pe[:, 1::2] = np.cos(angles[:, 1::2])   # odd dims: cosine
    return pe

pe = positional_encoding(seq_len=5, d_model=8)
print("positional encoding, 5 positions x 8 dims:\n", pe)
```

```
positional encoding, 5 positions x 8 dims:
 [[ 0.     1.     0.     1.     0.     1.     0.     1.   ]
 [ 0.841  0.54   0.1    0.995  0.01   1.     0.001  1.   ]
 [ 0.909 -0.416  0.199  0.98   0.02   1.     0.002  1.   ]
 [ 0.141 -0.99   0.296  0.955  0.03   1.     0.003  1.   ]
 [-0.757 -0.654  0.389  0.921  0.04   0.999  0.004  1.   ]]
```

Every position gets its own distinct, fixed vector — no two rows are
alike — added to that position's token embedding before attention runs, so
the model has *something* order-dependent to key off, even though attention
itself remains permutation-agnostic.

## Build this

Implement `scaled_dot_product_attention(Q, K, V)` in PyTorch yourself, in
roughly 20 lines, from memory rather than copying the version above:
compute scores, scale by `1/sqrt(d_k)`, softmax, multiply by `V`. Test it
on random `Q`, `K`, `V` tensors of shape `(batch=2, seq_len=5, d_k=16)` and
verify the output shape is `(2, 5, 16)` and that each row of your attention
weights sums to 1 (print `weights.sum(dim=-1)` and confirm it's all ones).
Cross-check your result against `torch.nn.functional.scaled_dot_product_attention`
with `torch.allclose` — if it returns `False`, the bug is almost always the
scaling factor or which dimension the softmax runs over.

**Stretch:** run this lesson's worked-example weights (`X`, `Wq`, `Wk`,
`Wv` from the "In code" section above) through your PyTorch function
instead of the NumPy version, and confirm you get the same `output` matrix
back (convert with `torch.from_numpy`, cast to `float32`). Then add a
causal mask — a lower-triangular matrix of 1s — so token `i` can only
attend to tokens `0..i`, matching the decoder's masked self-attention
described above, and print the resulting attention weight matrix to confirm
every entry above the diagonal is now exactly zero.

## Go deeper

- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the canonical visual walkthrough of the full architecture, from embeddings through the decoder's output layer.
- [d2l.ai: Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — a from-scratch treatment of attention, multi-head attention, positional encoding and the full Transformer.
- [PyTorch: Language Modeling with nn.Transformer](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html) — the official tutorial building a Transformer language model with PyTorch's built-in layers.
- [Vaswani et al., "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the original 2017 paper that introduced this entire architecture; worth reading once the ideas above are solid.
- [Karpathy: Let's build GPT, from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — implements a decoder-only Transformer, exactly the architecture behind modern LLMs, line by line.

**Next:** [How LLMs Work](24-how-llms-work.md)
