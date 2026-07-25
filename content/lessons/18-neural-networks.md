---
title: "Neural Networks"
stage: 4
order: 18
minutes: 55
difficulty: intermediate
prerequisites: ["first-ml-project", "calculus"]
tags: ["deep-learning", "neural-networks", "gradient-descent"]
summary: "What one neuron computes, why nonlinear activations are non-negotiable, how depth builds representations, and the universal approximation intuition."
---

# Neural Networks

## Why this matters

Every model in Stage 3 had a fixed shape: linear regression draws a straight
line (or hyperplane), a decision tree draws axis-aligned splits. To fit
something more complicated you had to hand-engineer features — add `x**2`,
bucket a column, multiply two features together. A neural network removes
that ceiling: it's built from the same weighted-sum-plus-bias you already
know from linear regression and the same gradient descent you implemented by
hand in the calculus lesson, but stacked into layers that *learn their own
features* instead of waiting for you to invent them. This lesson is about
exactly one building block — the neuron — and exactly one question: why
does stacking many of them, with one specific ingredient added, turn a
collection of straight lines into something that can approximate almost any
function. Everything else in Stage 4 is this idea, scaled up.

## The concept

**A single neuron computes a weighted sum, adds a bias, and applies a
nonlinearity.** In symbols: `z = w · x + b`, then `a = f(z)`. The first half
is exactly linear regression's `prediction = w · x + b` — nothing new. `f`
is the **activation function**, and `a` (the neuron's **activation**, its
output) is what gets passed to whatever uses this neuron's result next. If
`f` is the identity function (`f(z) = z`), a neuron *is* linear regression.
If `f` is the sigmoid you met in classification, a neuron *is* logistic
regression. A neural network is not a different kind of model from what
you've already built — it's many of these, connected together.

**Common activation functions.** `sigmoid(z) = 1 / (1 + e^-z)` squashes any
real number into `(0, 1)`. `tanh(z)` is sigmoid's cousin, squashing into
`(-1, 1)`, centred on zero. `ReLU(z) = max(0, z)` — "rectified linear
unit" — is the default choice in modern hidden layers: it's just as
nonlinear as the others (it has a sharp bend at zero) but is far cheaper to
compute and, as Stage 4 will show, suffers less from a specific training
problem called vanishing gradients. Different activations behave
differently, but they share the one property that actually matters here:
none of them is a straight line.

**Why the nonlinearity is not optional.** Suppose you skip it and stack
"linear layers" with no activation between them. Layer one computes
`h = W1 @ x + b1`. Layer two computes `output = W2 @ h + b2`. Substitute the
first into the second: `output = W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) @ x +
(W2 @ b1 + b2)`. That's a matrix (`W2 @ W1`) times `x` plus a vector — in
other words, exactly the same *shape* of computation as a single linear
layer, just with different numbers inside it. Stack ten such layers and you
still get one linear layer's worth of representational power: a straight
line (or flat hyperplane) no matter how many parameters you throw at it. The
"In code" section below multiplies this out and shows two stacked linear
layers producing bit-for-bit the same output as one combined layer. The
nonlinearity — sigmoid, tanh, ReLU, anything with a bend in it — is what
stops that collapse. It's the entire reason "depth" means something.

**Layers and depth.** The **input layer** is just your feature vector — no
computation happens there. Each **hidden layer** takes the previous layer's
activations, computes a new weighted sum plus bias per neuron, and applies a
nonlinearity, producing a new vector of activations that the next layer
consumes. The **output layer** does the same but its activation (or lack of
one) is chosen to match the task — identity for regression, sigmoid for
binary classification, softmax for multi-class. **Depth** is the number of
layers; **width** is how many neurons sit in a given layer. The lesson 00
intuition — early layers finding edges, later layers finding faces — is
exactly this: each hidden layer builds a new representation out of the
previous layer's representation, and it can only build something genuinely
new, rather than a repackaged straight line, because of the nonlinearity
sitting between them.

**The universal approximation intuition.** A landmark result (Cybenko 1989;
Hornik 1991) says a feedforward network with even a *single* hidden layer,
given enough hidden units and *any* reasonable nonlinear activation, can
approximate any continuous function on a bounded input range to arbitrary
precision. It's worth being precise about what this does and doesn't
promise: it's an **existence** proof, not a recipe. It doesn't say gradient
descent will *find* the right weights, or that "enough hidden units" is a
practical number, or that a shallow network is a good idea in practice (deep
narrow networks are, empirically, far more efficient learners than shallow
wide ones — the rest of Stage 4 is largely about architectures that exploit
that). But the intuition behind it is worth keeping: each hidden unit
contributes one bump- or step-shaped nonlinear function of the input;
summing enough of them, each scaled and shifted differently by its weights,
can sculpt an arbitrarily wiggly curve — the same way enough small, simple
building blocks can approximate any complex shape. The "In code" section
makes this concrete by fitting a curve that a single linear neuron
mathematically cannot represent.

## In code

What one neuron computes — weighted sum, bias, sigmoid:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Three inputs, one weight per input, one shared bias
x = np.array([0.5, -0.2, 0.9])
w = np.array([0.4, 0.8, -0.5])
b = 0.1

z = np.dot(w, x) + b       # weighted sum + bias
a = sigmoid(z)              # nonlinearity

print(f"z (weighted sum + bias) = {z:.4f}")
print(f"a (after sigmoid)       = {a:.4f}")
```

```
z (weighted sum + bias) = -0.3100
a (after sigmoid)       = 0.4231
```

Depth without nonlinearity collapses to one linear layer — proved by
multiplying it out, not just asserted:

```python
import numpy as np

rng = np.random.default_rng(0)

W1 = rng.normal(size=(4, 3))
b1 = rng.normal(size=4)
W2 = rng.normal(size=(2, 4))
b2 = rng.normal(size=2)

def two_stacked_linear_layers(x):
    h = W1 @ x + b1     # layer 1: linear, NO activation
    return W2 @ h + b2  # layer 2: linear, NO activation

# Multiply the layers together algebraically into one combined layer
W_combined = W2 @ W1
b_combined = W2 @ b1 + b2

def one_combined_linear_layer(x):
    return W_combined @ x + b_combined

x = rng.normal(size=3)
print("two stacked linear layers:", np.round(two_stacked_linear_layers(x), 4))
print("one combined linear layer:", np.round(one_combined_linear_layer(x), 4))
```

```
two stacked linear layers: [1.961  2.5786]
one combined linear layer: [1.961  2.5786]
```

Identical outputs. Two "layers" of pure linear algebra bought nothing that
one layer couldn't already do — this is why every real network puts a
nonlinearity between layers.

A single neuron with an identity activation, trained with gradient descent
(exactly last lesson's algorithm) to fit a line:

```python
import numpy as np

rng = np.random.default_rng(1)

# A single neuron with an identity activation: prediction = w*x + b
x = np.linspace(0, 10, 40)
y = 3.0 * x - 4.0 + rng.normal(0, 1.0, size=x.shape[0])  # true line: y = 3x - 4

w, b = 0.0, 0.0
learning_rate = 0.01

for step in range(500):
    pred = w * x + b
    error = pred - y
    grad_w = np.mean(2 * error * x)
    grad_b = np.mean(2 * error)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    if step % 100 == 0 or step == 499:
        loss = np.mean(error ** 2)
        print(f"step={step:3d}  w={w:6.3f}  b={b:6.3f}  loss={loss:7.3f}")

print(f"\nlearned: y = {w:.2f}x {b:+.2f}   (true rule: y = 3.00x - 4.00)")
```

```
step=  0  w= 1.622  b= 0.220  loss=199.722
step=100  w= 2.601  b=-1.357  loss=  2.553
step=200  w= 2.752  b=-2.371  loss=  1.472
step=300  w= 2.843  b=-2.980  loss=  1.082
step=400  w= 2.898  b=-3.346  loss=  0.941
step=499  w= 2.930  b=-3.564  loss=  0.890

learned: y = 2.93x -3.56   (true rule: y = 3.00x - 4.00)
```

Now the payoff: `y = x**2` is a curve no single linear neuron can represent
at all. A tiny network — one hidden layer of 8 units with a `tanh`
nonlinearity — fits it, using the exact same chain-rule machinery as the
last lesson's `gradient_descent`, applied one layer at a time (full detail
of that "one layer at a time" step is next lesson's topic):

```python
import numpy as np

rng = np.random.default_rng(2)

# Target: a curve a single linear neuron cannot represent
x = np.linspace(-3, 3, 60).reshape(-1, 1)
y = (x ** 2).flatten()

n_hidden = 8
W1 = rng.normal(0, 0.5, size=(1, n_hidden))
b1 = np.zeros(n_hidden)
W2 = rng.normal(0, 0.5, size=(n_hidden, 1))
b2 = np.zeros(1)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1 - a ** 2

lr = 0.01
for epoch in range(3000):
    z1 = x @ W1 + b1
    a1 = tanh(z1)              # the nonlinearity, once per hidden unit
    z2 = a1 @ W2 + b2
    pred = z2.flatten()

    loss = np.mean((pred - y) ** 2)

    d_pred = 2 * (pred - y) / len(y)
    d_W2 = a1.T @ d_pred.reshape(-1, 1)
    d_b2 = d_pred.sum(keepdims=True)
    d_a1 = d_pred.reshape(-1, 1) @ W2.T
    d_z1 = d_a1 * tanh_deriv(a1)
    d_W1 = x.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    W2 -= lr * d_W2; b2 -= lr * d_b2
    W1 -= lr * d_W1; b1 -= lr * d_b1

    if epoch % 500 == 0 or epoch == 2999:
        print(f"epoch={epoch:4d}  loss={loss:7.4f}")

# A single linear neuron (no hidden layer) fit to the same data, for comparison
w_lin, b_lin = 0.0, 0.0
xf = x.flatten()
for _ in range(2000):
    pred_lin = w_lin * xf + b_lin
    err = pred_lin - y
    w_lin -= 0.01 * np.mean(2 * err * xf)
    b_lin -= 0.01 * np.mean(2 * err)
linear_loss = np.mean((w_lin * xf + b_lin - y) ** 2)
print(f"\nbest a single LINEAR neuron can do on y=x^2: loss={linear_loss:7.4f}")
print(f"tiny nonlinear network (1 hidden layer, 8 units) final loss: {loss:7.4f}")
```

```
epoch=   0  loss=17.4594
epoch= 500  loss= 0.6797
epoch=1000  loss= 0.1209
epoch=1500  loss= 0.0657
epoch=2000  loss= 0.0469
epoch=2500  loss= 0.0358
epoch=2999  loss= 0.0280

best a single LINEAR neuron can do on y=x^2: loss= 7.6900
tiny nonlinear network (1 hidden layer, 8 units) final loss:  0.0280
```

A loss of `7.69` versus `0.028` on the same data is the universal
approximation intuition made numeric: eight nonlinear hidden units can do
what one linear neuron structurally cannot, at any amount of training.

## Build this

Write your own version of the "single neuron fits a line" code above from
scratch: generate `x` with `np.linspace`, define `y` as your own line
(`y = m*x + c` for coefficients you choose) plus `rng.normal` noise, then
implement the neuron (`w`, `b`, forward pass, gradient descent loop) and
confirm it recovers your `m` and `c`. Print the loss every 100 steps.

**Stretch:** change the neuron's activation from identity to `sigmoid` (keep
everything else the same, including your line-shaped data) and retrain.
Sigmoid output is trapped in `(0, 1)`, so if your `y` values range outside
that — which any real line's will, eventually — the neuron structurally
cannot fit it, no matter how long you train. Print the final loss for both
versions side by side. This is the concrete answer to "why does the output
layer of a regression network usually use an identity activation, while a
binary classifier's output layer uses sigmoid?"

## Go deeper

- [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — the clearest visual walkthrough of a neuron and a layered network that exists.
- [3Blue1Brown: Neural Networks series](https://www.3blue1brown.com/topics/neural-networks) — the full playlist this video belongs to; later videos in it cover backpropagation, next lesson's topic.
- [CS231n: Neural Networks Part 1](http://cs231n.github.io/neural-networks-1/) — Stanford's course notes on neurons, activation functions, and network architecture.
- [Michael Nielsen: Neural Networks and Deep Learning, Ch. 4](http://neuralnetworksanddeeplearning.com/chap4.html) — "A visual proof that neural nets can compute any function," a hands-on treatment of universal approximation.

**Next:** [Backprop & Training](19-backprop-training.md)
