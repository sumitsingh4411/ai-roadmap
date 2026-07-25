---
title: "Sequence Models"
stage: 4
order: 22
minutes: 50
difficulty: advanced
prerequisites: ["pytorch"]
tags: ["deep-learning", "rnn", "sequence-models"]
summary: "Why order matters, how RNNs process sequences step by step, the vanishing gradient problem, LSTM/GRU, and why attention replaced them."
---

# Sequence Models

## Why this matters

Every network so far has taken one fixed-size input and produced one
output: an image in, ten class scores out. Text, audio, sensor readings and
time series don't fit that mould — they're **sequences**, where the number
of items varies and, critically, *the order carries meaning*. "The dog bit
the man" and "the man bit the dog" contain the exact same words. A CNN's
convolution and a plain `nn.Linear` layer both have no concept of "before"
and "after" built in. This lesson covers the architecture built specifically
to fix that — the recurrent neural network — what breaks when you try to
train it on long sequences, the fixes that patched it for two decades, and
why, despite the fixes, the field eventually moved on to something else
entirely (next lesson's topic).

## The concept

**Why order matters, concretely.** A model that just averages or sums a
sequence's elements — the "bag of words" approach — produces the exact same
representation for "the dog bit the man" and "the man bit the dog," because
addition doesn't care about order. Any model that needs to tell these apart
needs some mechanism that treats "which position" as information, not just
"which values."

**A recurrent neural network (RNN) processes a sequence one step at a
time, carrying a memory forward.** At each timestep `t`, it takes the
current input `x_t` and the previous **hidden state** `h_{t-1}` (its
"memory" of everything seen so far), and computes a new hidden state:
`h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)`. Notice this is exactly a
neuron's weighted-sum-plus-nonlinearity from Lesson 18, just with two
inputs — the current data and the network's own previous output — instead
of one. The *same* weights (`W_xh`, `W_hh`, `b`) are reused at every
timestep, which is what lets an RNN handle sequences of any length: a
100-step sequence and a 5-step sequence use the identical weight matrices,
just applied more or fewer times. `h_t` is then typically fed to an output
layer, or passed to the next timestep, or both.

**Vanishing gradients: why plain RNNs struggle with long sequences.**
Training an RNN uses **backpropagation through time**: unroll the recurrence
across all `T` timesteps and apply the same layer-by-layer chain rule from
Lesson 19, except now "layers" are timesteps sharing one set of weights. The
gradient of the loss with respect to an *early* timestep's input has to flow
backward through *every* timestep in between, and at each one it gets
multiplied by roughly the same recurrent weight matrix and the same
activation function's derivative (`tanh'`, which is at most `1` and usually
much less). Multiply a bunch of numbers each less than 1 together, many
times, and the product shrinks toward zero exponentially fast — this is the
**vanishing gradient problem**. In practice it means: the loss at the end of
a long sequence has almost no gradient signal reaching the network's
handling of the sequence's early elements, so a plain RNN effectively can't
learn long-range dependencies — it "forgets" what happened many steps ago,
not because of a design choice but because training simply can't reach that
far back. (The reverse failure, gradients *growing* exponentially instead of
shrinking, is called the **exploding gradient problem** — rarer, but the
reason you'll sometimes see **gradient clipping**, capping the gradient's
size before each update, in real training code.) The "In code" section
measures this directly: how much the final loss gradient shrinks by the
time it reaches the first timestep of a real, untrained RNN.

**LSTM and GRU: architectures designed to fight vanishing gradients.** A
**Long Short-Term Memory (LSTM)** cell adds a second recurrent path, the
**cell state**, that flows forward with only simple additive, elementwise
updates rather than a repeated matrix multiply — a more direct route for
gradients to flow backward through many timesteps without repeatedly
shrinking. It controls what flows through that path with three learned
**gates** (values between 0 and 1, from a sigmoid, acting like a per-value
dial): a **forget gate** decides what to drop from the cell state, an
**input gate** decides what new information to add, and an **output gate**
decides what to expose to the rest of the network as the hidden state. A
**Gated Recurrent Unit (GRU)** is a simplified variant with two gates
instead of three and no separate cell state, cheaper to compute, and
competitive with LSTMs on many tasks. Neither architecture eliminates the
vanishing gradient problem entirely, but both make it far less severe than
a plain RNN, which is why LSTMs and GRUs, not plain RNNs, were the default
choice for sequence data for roughly a decade.

**The limits that motivated attention.** Even with LSTM/GRU gating, every
RNN variant shares two structural constraints. First, they're inherently
**sequential**: computing `h_t` requires `h_{t-1}`, which requires
`h_{t-2}`, and so on — you cannot compute timestep 50 before timestep 49,
which makes RNNs hard to parallelise on modern hardware built for doing many
independent computations at once. Second, even with the cell-state
improvements, a single fixed-size hidden state has to compress *everything*
relevant from an arbitrarily long history into one vector, and information
from far back still tends to get diluted or overwritten well before very
long sequences end. Both limits point toward the same wish: a mechanism
that lets a model look directly back at *any* earlier position in the
sequence, on demand, without forcing information through a long chain of
sequential steps first. That mechanism is attention — next lesson's entire
subject.

## In code

The same three numbers, fed to an untrained RNN in two different orders,
produce two different final hidden states — direct evidence that an RNN's
output depends on order, not just on which values were present:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

rnn = nn.RNN(input_size=1, hidden_size=4, batch_first=True)

seq_a = torch.tensor([[[1.0], [2.0], [3.0]]])   # order: 1, 2, 3
seq_b = torch.tensor([[[3.0], [2.0], [1.0]]])   # same values, reversed order

out_a, h_a = rnn(seq_a)
out_b, h_b = rnn(seq_b)

print("same three numbers, different order:")
print("final hidden state, order [1,2,3]:", h_a.detach().numpy().round(4))
print("final hidden state, order [3,2,1]:", h_b.detach().numpy().round(4))
print("\nmean of inputs is identical either way:", seq_a.mean().item(), seq_b.mean().item())
```

```
same three numbers, different order:
final hidden state, order [1,2,3]: [[[ 0.0223  0.9087 -0.7739 -0.5655]]]
final hidden state, order [3,2,1]: [[[ 0.1902  0.743  -0.1652  0.0013]]]

mean of inputs is identical either way: 2.0 2.0
```

The vanishing gradient, measured directly: how strongly does the loss at
the *final* timestep's output depend on the input at each earlier
timestep, in a real (untrained, randomly initialised) 30-step RNN?

```python
import torch
import torch.nn as nn

torch.manual_seed(5)

seq_len = 30
plain_rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
x = torch.randn(1, seq_len, 1, requires_grad=True)

out, _ = plain_rnn(x)
loss = out[0, -1].sum()   # loss depends only on the FINAL timestep's output
loss.backward()
grads = x.grad[0, :, 0].abs()   # how much does the final loss "care about" each input timestep?

print("gradient magnitude of the final-step loss w.r.t. each input timestep:")
print(f"{'timestep':>10} {'|gradient|':>14}")
for t in [0, 5, 10, 15, 20, 25, 29]:
    print(f"{t:>10} {grads[t].item():>14.2e}")

print(f"\ngrad at t=29 (last) / grad at t=0 (first): {(grads[29] / grads[0]).item():,.0f}x larger")
```

```
gradient magnitude of the final-step loss w.r.t. each input timestep:
  timestep     |gradient|
         0       9.73e-06
         5       3.81e-05
        10       1.94e-04
        15       1.26e-03
        20       8.89e-03
        25       5.28e-02
        29       5.21e-01

grad at t=29 (last) / grad at t=0 (first): 53,583x larger
```

The gradient shrinks by close to five orders of magnitude between the last
timestep and the first, in a network that hasn't even started training yet
— exactly the vanishing gradient problem the concept section described,
made numeric instead of abstract. (LSTM and GRU cells reduce, but do not
eliminate, this effect; the size of the improvement depends heavily on
sequence length and initialisation, which is why this lesson measures the
plain-RNN case, the cleanest version of the problem, rather than a
head-to-head comparison that would vary run to run.)

Character-level text generation, the exercise this lesson builds toward,
trained for real on a tiny corpus (small enough to run in about a second on
CPU — a real corpus and a real training run need far more data and far more
time, which is why this one is deliberately toy-sized):

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

text = "the cat sat on the mat. the cat ran to the door. "
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
print("corpus length:", len(text), " unique characters:", vocab_size)

data = torch.tensor([stoi[ch] for ch in text])
inputs = data[:-1]    # every character except the last
targets = data[1:]    # every character shifted by one -- "predict the next character"

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        x = self.embed(x)
        out, state = self.lstm(x, state)
        return self.fc(out), state

model = CharRNN(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

x = inputs.unsqueeze(0)
y = targets.unsqueeze(0)

for epoch in range(300):
    optimizer.zero_grad()
    logits, _ = model(x)
    loss = loss_fn(logits.squeeze(0), y.squeeze(0))
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 or epoch == 299:
        print(f"epoch={epoch:3d}  loss={loss.item():.4f}")

# Generate by feeding the model's own output back in as the next input
model.eval()
with torch.no_grad():
    seed = "the "
    generated = seed
    state = None
    inp = torch.tensor([[stoi[ch] for ch in seed]])
    for _ in range(40):
        logits, state = model(inp, state)
        next_id = torch.argmax(logits[0, -1]).item()
        generated += itos[next_id]
        inp = torch.tensor([[next_id]])

print(f"\nseed: {seed!r}")
print(f"generated: {generated!r}")
```

```
corpus length: 49  unique characters: 13
epoch=  0  loss=2.5778
epoch= 50  loss=0.1421
epoch=100  loss=0.0150
epoch=150  loss=0.0055
epoch=200  loss=0.0031
epoch=250  loss=0.0021
epoch=299  loss=0.0015

seed: 'the '
generated: 'the cat sat on the mat. the cat ran to the d'
```

Loss starts near `ln(13) ≈ 2.56` — the loss of guessing uniformly among 13
possible characters, exactly as expected before any training — and drops to
nearly zero. But be honest with yourself about what happened: with only 49
characters of training data, the model has enough capacity to simply
**memorise** the entire corpus rather than learn any general pattern about
English — the generated text above is (almost) exactly the training text
repeated. That's expected and correct behaviour for a model this size on
data this small, not a sign of "understanding language"; a real char-RNN
trained on megabytes of text, for much longer, is what produces genuinely
novel-sounding output, at the cost of far more compute than fits in a
lesson.

## Build this

Run the character-level RNN above with your own short piece of text (a
favourite quote, a nursery rhyme, anything 50–200 characters long) in place
of `text`. Try a few different seed strings for generation, including ones
that appear inside your corpus and ones that don't. Write one sentence
about what happens when you seed with characters the model never saw
together during training.

**Stretch:** replace `nn.LSTM` with plain `nn.RNN` (same constructor
signature) and retrain on the same tiny corpus. Does it still reach a loss
near zero in 300 epochs? Now extend the corpus to something 5-10x longer by
repeating or concatenating text, and compare how many epochs each
architecture needs to reach the same loss threshold — this is your first
direct, hands-on look at the difference gating makes, on a scale small
enough to actually run.

## Go deeper

- [Christopher Olah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — the canonical visual explanation of the LSTM's gates and cell state, still the best introduction a decade later.
- [Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) — the blog post that made char-RNNs famous, with real (large-scale, long-trained) generated text to compare against this lesson's toy version.
- [PyTorch: NLP From Scratch — Generating Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) — the official tutorial this lesson's exercise is a miniature version of.
- [d2l.ai: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html) — a from-scratch treatment of RNNs, backpropagation through time, and language modelling.

**Next:** [Transformers](23-transformers.md)
