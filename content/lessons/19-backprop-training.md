---
title: "Backprop & Training"
stage: 4
order: 19
minutes: 60
difficulty: intermediate
prerequisites: ["neural-networks"]
tags: ["deep-learning", "backpropagation", "gradient-descent"]
summary: "How the chain rule turns one output error into a gradient for every weight in a network, and how learning rate, epochs and batches shape training."
---

# Backprop & Training

## Why this matters

Last lesson ended with a network that fits a curve a single neuron
structurally cannot — but the code that trained it just said "backward
pass" and moved on. That backward pass, **backpropagation**, is the one
algorithm that makes training any network with more than one layer
possible, and it is nothing more than the chain rule from the calculus
lesson, applied once per layer, moving from the output back to the input.
Understand it once, by hand, in NumPy, and every framework's `.backward()`
call for the rest of this roadmap stops being a black box.

## The concept

**The loss function measures how wrong the network is, as a single
number.** Exactly as in Stage 3: for regression, mean squared error; for
classification, something like cross-entropy that scores predicted
probabilities against true labels. Everything about training is about
adjusting weights to make this one number smaller.

**A network's prediction is a chain of functions, so its gradient is a
chain of derivatives.** A two-layer network computes `a1 = f1(W1 @ x + b1)`,
then `a2 = f2(W2 @ a1 + b2)`, then a loss `L` from `a2`. This is precisely
the "function built out of smaller functions" the calculus lesson's chain
rule was written for — just with more functions chained together, and
matrices and vectors in place of single numbers. **Backpropagation is the
chain rule applied once per layer, starting at the loss and working
backward toward the input**, reusing each layer's result for the layer
before it instead of recomputing it from scratch. That reuse is the entire
reason it's efficient enough to train networks with millions of weights.

**Working backward through one layer.** Say you already know
`dL/da2` — how much the loss changes per unit change in the output layer's
activation (for MSE this is just `2 * (a2 - y)`, the same derivative the
calculus lesson derived). To get the gradient for `W2`, the chain rule
splits the journey from `L` to `W2` into two hops: `L` depends on `a2`,
`a2` depends on `z2` through the activation function `f2`, and `z2 = W2 @
a1 + b2` depends on `W2` directly. So:
`dL/dz2 = dL/da2 * f2'(z2)`, then `dL/dW2 = a1^T @ dL/dz2`. Critically,
`dL/da1 = dL/dz2 @ W2^T` — the gradient *with respect to the previous
layer's output* — which is exactly the quantity the previous layer needs to
run the same two-hop calculation for `W1`. Each layer only ever needs one
local derivative (its own activation function's slope) and the gradient
signal handed to it by the layer after it. Chain these hops from the output
back to the input and every weight in the network gets a gradient, without
ever re-deriving the whole expression from `L` down to that weight.

**Gradient descent updates every weight, all at once, against the
gradient.** Same rule as before: `W := W - learning_rate * dL/dW` for every
weight matrix and bias in the network, simultaneously, after the backward
pass has computed all of them. The gradient points toward *steeper loss*;
subtracting it is what makes the loss go down.

**Learning rate, epochs, and batches.** **Learning rate** is unchanged from
before: too small and training crawls, too large and it overshoots or
diverges — this lesson's code section makes the "too large" failure mode
concrete, in a way that's easy to miss reading about it in the abstract. One
**epoch** is one full pass through the entire training set. Networks are
usually trained for many epochs, because a single pass rarely adjusts the
weights enough. A **batch** is a subset of the training set used for one
gradient update — instead of computing the gradient from every example
before updating (**batch** gradient descent, what this lesson's code does,
practical only because XOR has four examples total) or from one example at
a time (noisy, rarely used alone), real training almost always uses
**mini-batches**: a few dozen to a few thousand examples per update. Smaller
batches mean noisier but more frequent updates and lower memory use; larger
batches mean smoother but slower and more memory-hungry updates. This is a
practical engineering knob, not a change to the math above — the gradient
for a mini-batch is just the same average-over-examples formula, computed
over fewer examples.

## In code

XOR is the standard toy problem for this lesson because it's the smallest
dataset that a single linear neuron *cannot* solve (no straight line
separates the two classes), forcing a real hidden layer. Full forward pass,
full backward pass by hand, full gradient descent update — every line
labelled with which piece of the chain rule it is:

```python
import numpy as np

rng = np.random.default_rng(0)

# XOR: not linearly separable -- a single neuron/linear model cannot learn this
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

n_in, n_hidden, n_out = 2, 4, 1
W1 = rng.normal(0, 1, size=(n_in, n_hidden))
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(0, 1, size=(n_hidden, n_out))
b2 = np.zeros((1, n_out))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)   # derivative of sigmoid, written in terms of its OWN output

lr = 0.5
for epoch in range(10000):
    # ---- forward pass ----
    z1 = X @ W1 + b1        # (4, 4) weighted sum + bias, layer 1
    a1 = sigmoid(z1)        # (4, 4) nonlinearity, layer 1
    z2 = a1 @ W2 + b2       # (4, 1) weighted sum + bias, layer 2
    a2 = sigmoid(z2)        # (4, 1) nonlinearity, layer 2 (the prediction)

    loss = np.mean((a2 - y) ** 2)

    # ---- backward pass: chain rule, one layer at a time, output to input ----
    d_a2 = 2 * (a2 - y) / len(y)      # dL/d(a2)
    d_z2 = d_a2 * sigmoid_deriv(a2)   # dL/d(z2) = dL/d(a2) * d(a2)/d(z2)
    d_W2 = a1.T @ d_z2                # dL/d(W2) = a1^T . dL/d(z2)
    d_b2 = d_z2.sum(axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T                # dL/d(a1), pushed back through W2
    d_z1 = d_a1 * sigmoid_deriv(a1)   # dL/d(z1) = dL/d(a1) * d(a1)/d(z1)
    d_W1 = X.T @ d_z1
    d_b1 = d_z1.sum(axis=0, keepdims=True)

    # ---- gradient descent update: step AGAINST the gradient ----
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 2000 == 0 or epoch == 9999:
        print(f"epoch={epoch:5d}  loss={loss:.4f}")

print("\nfinal predictions vs targets:")
z1 = X @ W1 + b1
a1 = sigmoid(z1)
a2 = sigmoid(a1 @ W2 + b2)
for inputs, pred, target in zip(X, a2, y):
    print(f"  input={inputs}  predicted={pred[0]:.3f}  target={target[0]:.0f}")
```

```
epoch=    0  loss=0.3430
epoch= 2000  loss=0.0233
epoch= 4000  loss=0.0033
epoch= 6000  loss=0.0016
epoch= 8000  loss=0.0011
epoch= 9999  loss=0.0008

final predictions vs targets:
  input=[0. 0.]  predicted=0.011  target=0
  input=[0. 1.]  predicted=0.971  target=1
  input=[1. 0.]  predicted=0.971  target=1
  input=[1. 1.]  predicted=0.037  target=0
```

Learning rate, made concrete on this exact network — too small barely
moves in 3000 epochs, a good value converges, and too large breaks training
so badly the network settles for predicting `0.5` for everything (pure
guessing, since XOR's targets average to 0.5):

```python
import numpy as np

def train_xor(lr, epochs=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    W1 = rng.normal(0, 1, size=(2, 4)); b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, size=(4, 1)); b2 = np.zeros((1, 1))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for epoch in range(epochs):
        a1 = sigmoid(X @ W1 + b1)
        a2 = sigmoid(a1 @ W2 + b2)

        d_a2 = 2 * (a2 - y) / len(y)
        d_z2 = d_a2 * a2 * (1 - a2)
        d_W2 = a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0, keepdims=True)
        d_a1 = d_z2 @ W2.T
        d_z1 = d_a1 * a1 * (1 - a1)
        d_W1 = X.T @ d_z1
        d_b1 = d_z1.sum(axis=0, keepdims=True)

        W2 -= lr * d_W2; b2 -= lr * d_b2
        W1 -= lr * d_W1; b1 -= lr * d_b1

    return np.mean((a2 - y) ** 2)

for lr in [0.01, 0.5, 5.0, 200.0]:
    loss = train_xor(lr)
    print(f"learning_rate={lr:6.2f}  final loss after 3000 epochs = {loss:.4f}")
```

```
learning_rate=  0.01  final loss after 3000 epochs = 0.2504
learning_rate=  0.50  final loss after 3000 epochs = 0.0061
learning_rate=  5.00  final loss after 3000 epochs = 0.0002
learning_rate=200.00  final loss after 3000 epochs = 0.5000
```

`0.01` hasn't had time to get anywhere in 3000 epochs. `0.5` and `5.0` both
converge — a wider "good enough" range than the single-neuron case, because
sigmoid activations keep every intermediate value bounded. `200.0` is so
large that each update overshoots wildly and the network never settles
anywhere useful, landing on `0.5` loss — the score you'd get by predicting
`0.5` (maximum uncertainty) for every example, meaning training has learned
literally nothing.

## Build this

From memory (or by re-deriving each line, not copy-pasting), write the full
XOR forward pass, backward pass, and gradient descent loop yourself. Once it
converges, change the target `y` from XOR's `[0, 1, 1, 0]` to AND's
`[0, 0, 0, 1]` and retrain with the *same* network. Then try it with a
single neuron and no hidden layer at all (skip straight from `X` to a
sigmoid output). AND is linearly separable, so the single neuron should
solve it fine — XOR never will, however long you train it, unless it has a
hidden layer. Write one sentence explaining why, in terms of what a hidden
layer buys you that last lesson didn't.

**Stretch:** shrink `n_hidden` from 4 down to 2 and retrain XOR from a few
different random seeds (change `rng = np.random.default_rng(seed)` for
`seed` in `0..4`). Two hidden units is the theoretical minimum that can
represent XOR — does every seed still converge to a low loss, or do some
get stuck? This is your first hands-on encounter with a **local minimum**:
gradient descent on a non-convex loss (which any network with a hidden
layer has) isn't guaranteed to reach the best possible solution, only *a*
point where the gradient is zero.

## Go deeper

- [3Blue1Brown: Neural Networks series](https://www.3blue1brown.com/topics/neural-networks) — the third and fourth videos in this series work through backpropagation and its calculus visually, building directly on the first two.
- [CS231n: Backpropagation notes](http://cs231n.github.io/optimization-2/) — Stanford's course notes deriving backprop through computational graphs, a more general framing than this lesson's two-layer case.
- [Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — starts by building `micrograd`, a working autograd engine, from scratch in Python; the best way to see this lesson's chain-rule bookkeeping generalised.
- [Karpathy: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) — practical, hard-won advice on what actually goes wrong when training networks, and how to debug it.

**Next:** [PyTorch](20-pytorch.md)
