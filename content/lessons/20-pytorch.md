---
title: "PyTorch"
stage: 4
order: 20
minutes: 60
difficulty: intermediate
prerequisites: ["backprop-training"]
tags: ["deep-learning", "pytorch", "tooling"]
summary: "Tensors, autograd, nn.Module, optimisers, and the canonical training loop — the framework that automates the backprop you just wrote by hand."
---

# PyTorch

## Why this matters

You just wrote every line of a working backward pass by hand — eight lines
of chain-rule derivatives, four lines of weight updates, correct, but
tedious and easy to get subtly wrong the moment the network gets deeper than
two layers. That's the problem PyTorch exists to solve: it keeps last
lesson's exact ideas — weighted sums, activations, the chain rule, gradient
descent — and automates the bookkeeping. You still need to understand what
`loss.backward()` is doing, which is why the last two lessons made you do it
by hand first. This lesson is where you stop doing it by hand. (If you
haven't already, install it with `pip install torch`; every example below
runs on CPU, no GPU required.)

## The concept

**Tensors are NumPy arrays that remember how they were computed.** A
`torch.Tensor` supports the same shape, dtype, indexing, broadcasting, and
matrix multiplication as a NumPy array — `torch.from_numpy` and
`.numpy()` convert between them directly, and for plain array math they're
interchangeable. The difference that matters for this lesson is
`requires_grad=True`: when set, PyTorch builds a graph of every operation
applied to that tensor as it happens, so it can later work backward through
that graph automatically.

**Autograd is backpropagation, generalised and automated.** Call
`.backward()` on any scalar tensor built from `requires_grad=True` inputs,
and PyTorch walks the computation graph it recorded — in reverse, applying
the chain rule at every recorded operation — filling in `.grad` on every
tensor that fed into the computation. This is precisely last lesson's
backward pass, except PyTorch derived each local derivative for you instead
of you writing `sigmoid_deriv` by hand, and it works for any computation
graph, not just the specific two-layer network you wrote it for.

**`nn.Module` is how you define a network's structure.** Subclass it, define
the layers you need as attributes in `__init__` (`nn.Linear(in_features,
out_features)` is a weighted-sum-plus-bias layer — the exact `W @ x + b`
from every earlier lesson, with `W` and `b` created and tracked for you),
and write a `forward` method that chains them together with whatever
activations you choose. Calling `model(x)` runs `forward` and returns the
output, with the full computation graph recorded behind it automatically.

**Optimisers apply the gradient descent update rule for you.**
`torch.optim.SGD(model.parameters(), lr=...)` is handed every learnable
tensor in the model; calling `optimizer.step()` applies exactly
`W := W - learning_rate * W.grad` to each one — the same line you wrote by
hand last lesson, now looped over every parameter in the network in one
call. `optimizer.zero_grad()` matters because `.backward()` *accumulates*
gradients into `.grad` rather than overwriting them (useful for advanced
cases, a footgun if forgotten here) — call it before every backward pass, or
gradients from the previous step silently add into the current one.

**The canonical training loop.** Every PyTorch training script, from a toy
XOR network to a billion-parameter language model, is a variation on the
same five lines, repeated once per batch: zero the gradients, run the
forward pass, compute the loss, call `.backward()`, call `.step()`. You will
type some version of this loop dozens of times across the rest of this
roadmap — it is worth memorising now, on a network small enough to verify
by hand against last lesson's manual version.

**Moving to GPU.** Every tensor and every model lives on a **device** — CPU
by default. `.to(device)` moves a tensor or model's data there; matching
devices is required for every operation (you can't multiply a CPU tensor by
a GPU tensor). `device = "cuda" if torch.cuda.is_available() else "cpu"` is
the standard pattern: write code once, and it uses a GPU if one is present
and falls back to CPU otherwise. The examples in this lesson are small
enough that CPU is completely fine and, in fact, often faster than a GPU
once you include the cost of moving tiny amounts of data across to it — GPUs
earn their keep on the larger models later in this roadmap.

## In code

Tensors: creation, shape, arithmetic, matrix multiplication, and converting
to and from NumPy:

```python
import torch
import numpy as np

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("x:\n", x)
print("shape:", x.shape, " dtype:", x.dtype)

y = torch.ones(2, 2) * 2
print("\nx + y:\n", x + y)
print("x @ y (matrix multiply):\n", x @ y)

n = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(n)
print("\nnumpy -> torch:", t, t.dtype)
print("torch -> numpy:", t.numpy())
```

```
x:
 tensor([[1., 2.],
        [3., 4.]])
shape: torch.Size([2, 2])  dtype: torch.float32

x + y:
 tensor([[3., 4.],
        [5., 6.]])
x @ y (matrix multiply):
 tensor([[ 6.,  6.],
        [14., 14.]])

numpy -> torch: tensor([1., 2., 3.], dtype=torch.float64) torch.float64
torch -> numpy: [1. 2. 3.]
```

Autograd, checked against a derivative you can verify by hand — for
`f(x) = x^2 + 3x`, `df/dx = 2x + 3`:

```python
import torch

x = torch.tensor(4.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()   # walks the computation graph backward, applying the chain rule at every step

print(f"x = {x.item()}")
print(f"y = x^2 + 3x = {y.item()}")
print(f"x.grad (dy/dx, computed by autograd) = {x.grad.item()}")
print(f"2x + 3 by hand = {2 * x.item() + 3}")

# A two-step computation to see the chain rule explicitly: c = (3a)^2 = 9a^2, dc/da = 18a
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
c = b ** 2
c.backward()
print(f"\na={a.item()}, c=(3a)^2={c.item()}, a.grad (dc/da)={a.grad.item()}, 18a by hand={18 * a.item()}")
```

```
x = 4.0
y = x^2 + 3x = 28.0
x.grad (dy/dx, computed by autograd) = 11.0
2x + 3 by hand = 11.0

a=2.0, c=(3a)^2=36.0, a.grad (dc/da)=36.0, 18a by hand=36.0
```

Now the exercise this lesson is building toward: last lesson's XOR network,
rewritten with `nn.Module`, an optimiser, and the canonical training loop:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

model = XORNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(10000):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()      # autograd computes every gradient
    optimizer.step()     # the optimizer applies the update rule

    if epoch % 2000 == 0 or epoch == 9999:
        print(f"epoch={epoch:5d}  loss={loss.item():.4f}")

print("\nfinal predictions vs targets:")
with torch.no_grad():
    preds = model(X)
    for inputs, pred, target in zip(X, preds, y):
        print(f"  input={inputs.tolist()}  predicted={pred.item():.3f}  target={target.item():.0f}")
```

```
epoch=    0  loss=0.2544
epoch= 2000  loss=0.0296
epoch= 4000  loss=0.0029
epoch= 6000  loss=0.0014
epoch= 8000  loss=0.0009
epoch= 9999  loss=0.0006

final predictions vs targets:
  input=[0.0, 0.0]  predicted=0.024  target=0
  input=[0.0, 1.0]  predicted=0.975  target=1
  input=[1.0, 0.0]  predicted=0.975  target=1
  input=[1.0, 1.0]  predicted=0.025  target=0
```

Same problem, same architecture, same convergence — the `d_a2`, `d_z2`,
`d_W2`, `d_b2`, `d_a1`, `d_z1`, `d_W1`, `d_b1` block from last lesson (eight
lines of chain-rule derivatives) plus the four-line update loop are gone,
replaced by `loss.backward()` and `optimizer.step()`.

Checking which device this code is running on:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
```

```
device: cpu
```

## Build this

Rewrite last lesson's manual-backprop XOR network as a `nn.Module`, from
memory rather than copy-pasting the block above. Once it trains
successfully, count lines: how many lines does your PyTorch version take
from "define the network" through "print final predictions," versus your
NumPy version from last lesson's exercise? Write down the two counts and one
sentence about which lines specifically disappeared.

**Stretch:** swap `torch.optim.SGD(model.parameters(), lr=0.5)` for
`torch.optim.Adam(model.parameters(), lr=0.05)` — a different, more adaptive
optimiser you'll see used far more often in later lessons — and compare how
many epochs each needs to reach a loss below `0.01`. Nothing else about the
model or the loop changes; this is the entire point of separating the
optimiser from the model.

## Go deeper

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — the official beginner tutorial series, covering tensors through a full training loop on a real dataset.
- [PyTorch: A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — the official autograd walkthrough, going one level deeper than this lesson's two examples.
- [PyTorch: What is torch.nn *really*?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html) — builds `nn.Module` and an optimiser up from raw tensors, showing exactly what the framework is automating.
- [Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — having built `micrograd` by hand, the same series shows how that maps onto real PyTorch code.

**Next:** [CNNs & Vision](21-cnns-vision.md)
