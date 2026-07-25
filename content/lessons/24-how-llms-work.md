---
title: "How LLMs Work"
stage: 5
order: 24
minutes: 55
difficulty: intermediate
prerequisites: ["transformers"]
tags: ["llm", "genai", "tokenization"]
summary: "Tokenization, next-token prediction, pretraining vs post-training, context windows, temperature and sampling, and why models hallucinate."
---

# How LLMs Work

## Why this matters

Last stage ended with the mechanism: attention, self-attention, multi-head
attention, the decoder-only Transformer. This stage is about what happens
when you take that mechanism, stack it dozens of layers deep, train it on a
meaningful fraction of the public internet, and give the result a chat
interface. Everything from here — prompting, embeddings, RAG, fine-tuning,
agents, evals — treats "the LLM" as a black box with specific, learnable
behavior. This lesson opens the box just enough to replace guesses with
facts: what the model actually sees (tokens, not words), what it's actually
doing on every single forward pass (predicting one next token), how it went
from "predicts plausible internet text" to "answers your question
helpfully" (pretraining vs post-training), what it can and can't see at
once (the context window), how a fixed set of next-token probabilities
becomes actual generated text (temperature and sampling), and — the one
piece of folk wisdom worth replacing with a mechanism — why it sometimes
states false things with total confidence. Get this lesson wrong and every
later lesson inherits a wrong mental model.

## The concept

**Tokenization: the model reads pieces of words, not words.** Before any
text reaches the model, a **tokenizer** breaks it into **tokens** — chunks
that are usually smaller than a word (a common word might be one token; an
uncommon or made-up word gets split into several subword pieces) and
occasionally larger (a common multi-character sequence like a space plus a
short word). The tokenizer's vocabulary — typically tens of thousands of
possible tokens — is built once, ahead of time, by running a
frequency-based merging algorithm (a family of techniques called **byte
pair encoding**, or BPE, and its relatives) over a huge corpus: start from
individual characters, repeatedly merge the most frequent adjacent pair
into a new token, until the vocabulary reaches its target size. The result
is a fixed set of subword pieces where common words end up as single
tokens (they were frequent enough to fully merge) and rare words end up
split into multiple pieces (they weren't). This is why tokenization is
subword-based rather than word-based: a fixed vocabulary of whole words
could never cover every word, typo, and made-up string a user might type,
but a subword vocabulary can represent *any* string as some sequence of
its pieces, right down to individual bytes if nothing bigger matches. The
"In code" section runs a real tokenizer on a real sentence and a real
invented word, so you can see exactly where the splits land.

**Next-token prediction: the entire pretraining objective, in one
sentence.** Given a sequence of tokens, the model — the decoder-only
Transformer from last lesson, trained with masked self-attention so
position *i* only ever attends to positions `0..i` — outputs, for the
position after the last token, a probability distribution over every token
in the vocabulary: "given everything so far, what's the next token?" That
single objective, repeated over trillions of tokens of text, is
**pretraining**. Nobody hand-labels "the correct next word" — the label is
just *whatever token actually came next in the training text*, which makes
this a form of self-supervised learning: the data labels itself. Every
capability an LLM appears to have — answering questions, writing code,
holding a conversation, "reasoning" — is downstream of getting very good at
this one task at a scale where next-token prediction requires implicitly
learning grammar, facts, and patterns of reasoning well enough to predict
what a person who knew those things would write next.

**Pretraining produces a completion engine, not an assistant — post-training
turns it into one.** A raw pretrained ("base") model, asked "What's the
capital of France?", is just as likely to continue with another trivia
question as it is to answer the one you asked — it learned to continue
*internet text*, and internet text containing that exact question is
sometimes a quiz, sometimes a forum thread, sometimes a list. Turning a
base model into something that reliably behaves like a helpful assistant
is **post-training**, itself usually two further stages. **Supervised
fine-tuning (SFT)** continues training the model, now on a much smaller,
curated dataset of (prompt, ideal response) pairs written or selected by
humans — this teaches the *shape* of a good response: answer the question
that was actually asked, in a helpful tone, in a reasonable format.
**Reinforcement learning from human feedback (RLHF)**, and related
techniques like Constitutional AI, go a step further: humans (or, in some
variants, another model acting as a judge) rank multiple candidate
responses to the same prompt from best to worst, that ranking data trains
a separate reward model to predict "how good is this response," and the
LLM is then further trained to produce responses the reward model scores
highly — nudging it toward being more helpful, honest, and harmless in
ways too subtle to write down as explicit training examples. The takeaway
that matters for everything downstream: prompting, chat formatting, and
"assistant behavior" are a post-training layer on top of a next-token
predictor, not a different kind of model underneath.

**The context window is the model's entire field of view.** Every LLM has
a maximum number of tokens it can process in one request — the **context
window** — fixed by the architecture and how it was trained (self-attention
computes a similarity score between every pair of positions, so cost grows
with sequence length; positional encoding schemes are also usually trained
or tuned for a specific maximum length). Anything outside that window,
the model literally does not see: it isn't summarized, it isn't
remembered, it's simply absent from the computation. An API call is also
**stateless** — the model has no memory of a previous conversation unless
the entire prior conversation is resent as part of the current prompt,
which is exactly what chat interfaces do behind the scenes. Everything
that will later look like "memory" in this stage — a system prompt, RAG,
an agent's running transcript — is really just "what got put into this
particular context window."

**Temperature and sampling: turning probabilities into an actual token.**
The model's raw output for the next position is **logits** — one
unnormalized score per vocabulary token — turned into a probability
distribution by softmax (exactly the operation from last lesson's
attention weights, applied here over the whole vocabulary instead of over
a handful of tokens). Given that distribution, something has to pick the
actual next token. **Greedy decoding** always takes the single
highest-probability token — deterministic, but often repetitive and dull,
since it never takes a slightly-lower-probability but more interesting
path. **Sampling** instead draws a token at random, weighted by the
distribution, and **temperature** controls how sharp or flat that
distribution is before you sample from it: divide the logits by the
temperature before softmax. A temperature below 1 sharpens the
distribution (the already-likely tokens get relatively more likely,
pushing sampling toward greedy-like, more deterministic behavior); a
temperature above 1 flattens it (probabilities even out, increasing the
chance of a low-probability, more surprising token); temperature exactly 1
leaves the distribution as computed. (Two related knobs you'll see in
every API: **top-k** sampling restricts sampling to only the k
highest-probability tokens before applying temperature; **top-p**, or
nucleus, sampling restricts to the smallest set of tokens whose cumulative
probability reaches p — both exist to stop temperature alone from
occasionally sampling a wildly implausible token from the distribution's
long tail.) The "In code" section runs the actual softmax-with-temperature
math over a toy vocabulary so you can see the distribution reshape itself
in real numbers.

**Hallucination: the model optimizes for plausible, not true.** This is
the single most important sentence in this lesson: an LLM was trained to
predict the statistically likely continuation of its input, given
everything it learned during pretraining and post-training — it was never
given a separate mechanism for "is this claim actually true" that runs
before it emits a token. Most of the time this works out fine, because
true statements about the world tend to be more consistent and more
frequent across a huge training corpus than any one specific false
statement, so "plausible" and "true" line up often enough that the model
looks reliable. **Hallucination is what happens when they come apart**:
ask about a rare fact, a specific citation, an exact statistic, a
person or event the model has weak or conflicting evidence about, or
anything genuinely after its training cutoff, and the model still does
the only thing it knows how to do — produce fluent, confident-sounding
text that continues the pattern of "an answer to this kind of question" —
because fluency, not epistemic humility, is what next-token prediction
plus post-training actually optimizes for. Post-training partially helps
(some RLHF signal rewards saying "I don't know" over confidently
fabricating), but it doesn't fix the underlying mechanism, and can even
make it worse if the reward signal implicitly favors confident-sounding
answers over hedged-but-honest ones — a **reward-model bias toward fluent
overconfidence**. (This is distinct from *sycophancy*, a related RLHF
failure mode where the model shapes its answer toward what it thinks the
*user* wants to hear — agreeing when challenged, mirroring your stated
opinion — rather than toward what is true.) Two lessons from now, RAG (lesson 27) is the standard
mitigation — give the model real, retrieved text to condition on instead
of asking it to recall from weights alone — and it genuinely reduces
hallucination by grounding the answer in something checkable. It does not
eliminate it: a model can still misread, over-generalize from, or add
unsupported detail on top of a passage it was correctly given, for the
exact same reason described here. Evals (lesson 30) exist because "the
output sounds right" is not evidence that it is right.

## In code

Real subword tokenization, using `tiktoken` (`pip install tiktoken`) — this
runs entirely offline once installed, no API key, no download beyond the
small vocabulary file. `tiktoken` implements OpenAI's tokenizer family, not
Claude's — the two use different trained vocabularies, so exact token
boundaries would differ if you ran the same sentence through Claude's
tokenizer via the API's `count_tokens` endpoint — but the *mechanism*
(byte-pair-encoding merges into a subword vocabulary) and the *shape of the
result* (common words survive as one token, rare or invented words get
chopped into several) generalize to essentially every modern LLM,
including Claude:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

sentence = "Tokenization splits text into subword pieces, not whole words."
tokens = enc.encode(sentence)

print(f"Sentence: {sentence!r}")
print(f"Characters: {len(sentence)}")
print(f"Words (naive split): {len(sentence.split())}")
print(f"Tokens: {len(tokens)}")
print(f"Token IDs: {tokens}")
print("\nToken pieces:")
for t in tokens:
    piece = enc.decode([t])
    print(f"  id={t:>6}  {piece!r}")
```

```
Sentence: 'Tokenization splits text into subword pieces, not whole words.'
Characters: 62
Words (naive split): 9
Tokens: 13
Token IDs: [3404, 2065, 41567, 1495, 1139, 1207, 1178, 9863, 11, 539, 4459, 4339, 13]

Token pieces:
  id=  3404  'Token'
  id=  2065  'ization'
  id= 41567  ' splits'
  id=  1495  ' text'
  id=  1139  ' into'
  id=  1207  ' sub'
  id=  1178  'word'
  id=  9863  ' pieces'
  id=    11  ','
  id=   539  ' not'
  id=  4459  ' whole'
  id=  4339  ' words'
  id=    13  '.'
```

Notice `Tokenization` itself splits into `Token` + `ization`, and `subword`
splits into `sub` + `word` — the tokenizer never saw either exact word
often enough during vocabulary-building to merge it into one token, even
though both are ordinary English. Push this further with two genuinely
rare, long words:

```python
rare = "The unbelievably antidisestablishmentarianistic supercalifragilisticexpialidocious tokenizer strikes again."
rare_tokens = enc.encode(rare)
print(f"Sentence: {rare!r}")
print(f"Words (naive split): {len(rare.split())}   Tokens: {len(rare_tokens)}")
print("Pieces for the long words:")
for word in ["antidisestablishmentarianistic", "supercalifragilisticexpialidocious"]:
    wt = enc.encode(word)
    pieces = [enc.decode([t]) for t in wt]
    print(f"  {word!r} -> {len(wt)} tokens -> {pieces}")
```

```
Sentence: 'The unbelievably antidisestablishmentarianistic supercalifragilisticexpialidocious tokenizer strikes again.'
Words (naive split): 7   Tokens: 23
Pieces for the long words:
  'antidisestablishmentarianistic' -> 6 tokens -> ['ant', 'idis', 'establish', 'ment', 'arian', 'istic']
  'supercalifragilisticexpialidocious' -> 11 tokens -> ['sup', 'erc', 'al', 'if', 'rag', 'il', 'istic', 'exp', 'ial', 'id', 'ocious']
```

Seven words became 23 tokens — over 3 tokens per word on average, because
two of those seven words alone cost 17 tokens between them. This is exactly
why API pricing and context windows are measured in tokens, not words or
characters: token count depends on *how common the text is*, not how long
it looks.

Now the other real, runnable piece: softmax with temperature, over a tiny
five-word toy vocabulary standing in for "next-token logits" (numpy only,
no model needed — this is literally the last arithmetic step of every LLM
forward pass, isolated so you can watch it happen):

```python
import numpy as np

np.set_printoptions(precision=3, suppress=True)

vocab = ["cat", "dog", "car", "carpet", "banana"]
logits = np.array([2.0, 1.8, 0.5, 0.3, -1.0])

def softmax_with_temperature(logits, temperature):
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled))
    return exp / exp.sum()

for temp in [0.1, 0.7, 1.0, 2.0]:
    probs = softmax_with_temperature(logits, temp)
    print(f"temperature={temp}")
    for word, p in zip(vocab, probs):
        print(f"  {word:8s} {p:.3f}")
    print(f"  (sum = {probs.sum():.3f})")
    print()
```

```
temperature=0.1
  cat      0.881
  dog      0.119
  car      0.000
  carpet   0.000
  banana   0.000
  (sum = 1.000)

temperature=0.7
  cat      0.507
  dog      0.381
  car      0.060
  carpet   0.045
  banana   0.007
  (sum = 1.000)

temperature=1.0
  cat      0.440
  dog      0.360
  car      0.098
  carpet   0.080
  banana   0.022
  (sum = 1.000)

temperature=2.0
  cat      0.330
  dog      0.299
  car      0.156
  carpet   0.141
  banana   0.074
  (sum = 1.000)
```

Same five logits, four completely different distributions. At
`temperature=0.1` the model would sample `cat` almost every time (88.1%) —
close to greedy decoding. At `temperature=2.0` the four alternatives all
become live possibilities, `banana` included, even though it started with
a *negative* logit. Nothing here called an LLM — this is the exact
arithmetic every LLM API runs on its final layer's output before it hands
you back a token, isolated so the mechanism is visible.

## Build this

Tokenize five sentences of your own choosing with the code above — include
at least one with your own name, one with a technical term or piece of
jargon from your field, and one with a deliberate typo. Before running
each, write down your guess for the token count; then run it and check.
For any sentence where your guess was off by more than one or two tokens,
decode the individual token pieces (as in the examples above) and look at
exactly where the tokenizer split — is it splitting on a word boundary
you'd expect, or somewhere that surprises you?

**Stretch:** implement top-p (nucleus) sampling on top of the temperature
demo. Given a probability array from `softmax_with_temperature`, sort
tokens by probability descending, keep adding tokens to a candidate set
until their cumulative probability reaches or exceeds `p`, renormalize
just that subset's probabilities to sum to 1, and sample from it (use
`np.random.choice` with the renormalized weights). Verify two boundary
cases with real runs: a very small `p` (like 0.1) should usually collapse
to just the single most likely token, behaving like greedy decoding; `p =
1.0` should include the whole vocabulary and behave identically to plain
temperature sampling.

## Go deeper

- [Jay Alammar: The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — the visual walkthrough of exactly the next-token-prediction loop this lesson describes in prose.
- [Hugging Face NLP Course: Tokenizers](https://huggingface.co/learn/nlp-course/chapter2/4) — a deeper, from-scratch treatment of how subword tokenizers are built and used.
- [OpenAI's tiktoken](https://github.com/openai/tiktoken) — the library this lesson's code runs; its README covers the encodings and how BPE merges work.
- [Anthropic: Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting) — how to count tokens against Claude's actual tokenizer once you have an API key, instead of tiktoken's OpenAI-family approximation.

**Next:** [Prompt Engineering](25-prompt-engineering.md)
