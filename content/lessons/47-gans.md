---
title: "Generative Adversarial Networks (GANs)"
stage: 4
order: 47
minutes: 50
difficulty: advanced
prerequisites: ["neural-networks"]
tags: ["gans", "generative-ai"]
summary: "Two networks in a duel — a generator faking data and a discriminator catching fakes — and how that adversarial game learns to create."
---

# Generative Adversarial Networks (GANs)

## Why this matters

Every network so far in this roadmap learns by being told the right answer
directly: a loss function compares its output to a known target and
gradient descent closes the gap. Generative modeling breaks that setup —
there is no "right answer" for what a new, realistic face should look
like, no target vector to subtract a prediction from. **Generative
Adversarial Networks**, introduced by Ian Goodfellow in 2014, solve this
with a strange and genuinely clever trick: instead of one network learning
from a fixed loss, *two* networks learn by competing with each other, and
the competition itself is what defines "realistic." One network's entire
job is to be the other network's loss function. This lesson builds that
idea from the ground up and is honest about the well-known reason GANs are
notoriously hard to train.

## The concept

**A generator and a discriminator play an adversarial game.** The
**generator** takes random noise as input — a vector of random numbers with
no meaning on its own — and outputs a fake sample: an image, in the classic
case, but the toy example below uses a single number. The **discriminator**
is a binary classifier (the same kind of network from earlier lessons, sigmoid
output and all) that takes a sample, real or fake, and outputs the
probability it's real. The two train against each other: the discriminator
is trained to correctly separate real samples from the generator's fakes,
while the generator is trained to produce fakes that fool the
discriminator into predicting "real." Neither network is ever told directly
what a "good" image looks like — the generator's only signal is whether it
fooled the other network.

**This is a minimax game, not ordinary supervised training.** The
discriminator wants to *maximize* its accuracy at telling real from fake;
the generator wants to *minimize* that same accuracy, working against it.
Training alternates: hold the generator fixed, take one or more gradient
steps improving the discriminator on a batch of real and fake samples; then
hold the discriminator fixed, take one or more gradient steps improving the
generator (using the discriminator's judgment as the generator's loss,
even though the discriminator itself isn't being updated on this step).
Neither network trains toward a fixed target — the "target" a generator's
gradient points toward changes every time the discriminator updates, since
it's chasing a moving opponent, not a fixed loss surface the way regression
or classification networks in earlier lessons did.

**The equilibrium is the discriminator being unable to tell them apart.**
If training goes well, the generator gets good enough that its fakes are
statistically indistinguishable from real data, and the discriminator's
accuracy converges toward 50% — pure guessing, the same as flipping a coin,
because there is genuinely no signal left to separate real from fake. That
convergence point (both networks reaching a draw where neither can improve
further against the other) is the theoretical goal, called a **Nash
equilibrium** in game theory, though real training rarely reaches it this
cleanly. The "In code" section trains a tiny GAN and prints the
discriminator's accuracy over training, so you can watch this dynamic (or
its breakdown) happen in numbers.

**How this differs from other generative approaches.** A **diffusion
model** (an earlier lesson) also generates new samples from noise, but
trains with an ordinary, direct loss: at each step, it's shown exactly what
noise it should be predicting and told to get closer to it — no adversary,
no minimax game, just standard gradient descent toward a fixed target. A
GAN has no such direct target for the generator; realism is defined purely
relative to whatever the discriminator currently believes. This makes GANs
conceptually elegant and, historically, capable of very sharp, realistic
outputs — but also harder to train stably, which is the reason diffusion
models have displaced GANs as the default choice for high-quality image
generation in most recent systems (more in "where GANs sit today," below).

**Mode collapse: the generator finds one shortcut and stops exploring.**
If the real data has several distinct clusters or varieties (say, photos of
several different dog breeds) a poorly-trained generator can learn to
produce only *one* variety extremely convincingly — one that reliably fools
the discriminator — and stop producing anything else, because there's no
term in its loss explicitly rewarding *diversity*, only rewarding "fooled
the discriminator right now." The result is a generator that produces
highly realistic but nearly identical outputs regardless of the random
noise fed in. It's called mode collapse because the generator has
collapsed onto one "mode" (cluster) of the real data distribution and
abandoned the rest.

**Training instability is the norm, not the exception.** Because the
generator's loss landscape shifts every time the discriminator updates (and
vice versa), GAN training can oscillate instead of converging: the
discriminator gets too strong too fast and gives the generator a vanishing,
uninformative gradient (the discriminator is so confident every fake is
fake that there's barely any signal for the generator to climb); or the
generator overshoots and the two networks chase each other back and forth
without settling down. This is a real, actively-researched problem, not a
sign of a badly-written implementation — much of the GAN research
literature (Wasserstein GANs, spectral normalization, and others) exists
specifically to make this dynamic more stable.

**Where GANs sit today.** GANs were the dominant approach to high-quality
image generation from roughly 2014 through the early 2020s (StyleGAN's
faces were the benchmark for photorealism for years). Diffusion models
(covered in an earlier lesson) have since become the default for most
new image- and video-generation systems, largely because their ordinary,
stable training loss makes them far easier to scale up reliably than a
minimax game between two competing networks. GANs haven't disappeared —
they're still used where fast, single-pass generation matters (a trained
GAN generates in one forward pass; diffusion models need many denoising
steps) — but the adversarial-training idea itself, two networks defining
each other's objective, has also spread well beyond image generation, into
areas like data augmentation and domain adaptation.

## In code

`pip install numpy` if you haven't already for a previous lesson.

A 1-D GAN, small enough to read every line: real data comes from a Gaussian
centred at 4.0, the generator is two numbers (a shift and a scale applied
to noise), and the discriminator is a single logistic-regression neuron —
the exact neuron from the neural networks lesson, sigmoid and all:

```python
import numpy as np

rng = np.random.default_rng(0)

# Real data: samples from a target distribution -- a Gaussian centred at 4.0
TARGET_MEAN, TARGET_STD = 4.0, 0.5

def sample_real(n):
    return rng.normal(TARGET_MEAN, TARGET_STD, size=n)

# Generator: turns noise z into a fake sample via a learned shift and scale.
# fake = z * scale + shift -- two numbers are the entire "network."
gen_shift, gen_scale = 0.0, 1.0

def generate(n, shift, scale):
    z = rng.normal(0, 1, size=n)          # noise input
    return z * scale + shift

# Discriminator: logistic regression on a single scalar input.
# p(real) = sigmoid(w * x + b)
disc_w, disc_b = 0.0, 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def disc_predict(x, w, b):
    return sigmoid(w * x + b)

lr = 0.03
batch = 64

def train_discriminator_step(w, b, shift, scale):
    real = sample_real(batch)
    fake = generate(batch, shift, scale)

    # Discriminator wants p(real)->1 for real, p(real)->0 for fake --
    # gradient of binary cross-entropy loss w.r.t. w and b.
    pred_real = disc_predict(real, w, b)
    grad_w_real = np.mean((pred_real - 1) * real)
    grad_b_real = np.mean(pred_real - 1)

    pred_fake = disc_predict(fake, w, b)   # target label 0 for fakes
    grad_w_fake = np.mean(pred_fake * fake)
    grad_b_fake = np.mean(pred_fake)

    w -= lr * (grad_w_real + grad_w_fake)
    b -= lr * (grad_b_real + grad_b_fake)
    return w, b

def train_generator_step(shift, scale, w, b):
    # Generator wants the discriminator to say p(real) -> 1 for its fakes --
    # it never sees real data, only the discriminator's judgment of its fakes.
    z = rng.normal(0, 1, size=batch)
    fake = z * scale + shift
    pred_fake = disc_predict(fake, w, b)

    d_fake = (pred_fake - 1) * w   # d(loss)/d(fake), loss = -log(pred_fake)
    grad_shift = np.mean(d_fake)
    grad_scale = np.mean(d_fake * z)

    shift -= lr * grad_shift
    scale -= lr * grad_scale
    return shift, scale

print(f"target distribution: mean={TARGET_MEAN}, std={TARGET_STD}")
print(f"{'step':>4}  {'gen_mean':>9}  {'gen_std':>8}  {'disc_acc':>9}")

for step in range(450):
    disc_w, disc_b = train_discriminator_step(disc_w, disc_b, gen_shift, gen_scale)
    gen_shift, gen_scale = train_generator_step(gen_shift, gen_scale, disc_w, disc_b)

    if step % 75 == 0 or step == 449:
        real_eval = sample_real(200)
        fake_eval = generate(200, gen_shift, gen_scale)
        real_pred = disc_predict(real_eval, disc_w, disc_b) > 0.5
        fake_pred = disc_predict(fake_eval, disc_w, disc_b) <= 0.5
        disc_acc = (real_pred.mean() + fake_pred.mean()) / 2
        print(f"{step:4d}  {gen_shift:9.3f}  {abs(gen_scale):8.3f}  {disc_acc:9.3f}")

print(f"\nfinal generator: fake ~ N(mean={gen_shift:.3f}, std={abs(gen_scale):.3f})")
print(f"target was:      real ~ N(mean={TARGET_MEAN}, std={TARGET_STD})")
```

```
target distribution: mean=4.0, std=0.5
step   gen_mean   gen_std   disc_acc
   0      0.001     1.000      0.750
  75      0.698     0.852      0.840
 150      1.594     0.663      0.873
 225      2.454     0.510      0.840
 300      3.236     0.393      0.770
 375      3.922     0.310      0.532
 449      4.502     0.262      0.323

final generator: fake ~ N(mean=4.502, std=0.262)
target was:      real ~ N(mean=4.0, std=0.5)
```

Watch both columns together. `gen_mean` climbs steadily from `0.0` toward
the target `4.0` as the generator learns where the real data lives — by
step 375 it's essentially arrived (`3.922`). But look at `disc_acc`
alongside it: it *rises* first, to `0.873` at step 150 (the discriminator
easily spotting fakes that are still far from the target), then *falls* as
the generator closes in, dropping to `0.323` by the end — worse than random
guessing. That's the adversarial game visible in numbers: as the generator
gets better, the discriminator's job gets harder, not easier, and by step
449 the generator has actually overshot the target mean (`4.502` vs. `4.0`)
while the discriminator is confidently wrong — a small, honest example of
the oscillating instability described above, not a bug in the code.

Real image GANs replace the two-number generator and one-neuron
discriminator with full convolutional networks and train for tens of
thousands of steps — far too much compute for this lesson, so shown here as
**illustrative rather than executed**:

```python
# ILLUSTRATIVE -- a DCGAN generator, in the style of the original DCGAN paper
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),   # output pixels scaled to [-1, 1]
        )

    def forward(self, z):
        return self.net(z)   # noise vector -> a full RGB image
```

What to expect if you run a full DCGAN training loop yourself: a
discriminator loss and generator loss printed every batch, both usually
bouncing around rather than smoothly decreasing (exactly the instability
described above, just with real images instead of one scalar), and image
samples saved periodically that start as static and gradually sharpen into
recognizable, if imperfect, images over thousands of steps.

## Build this

Run the 1-D GAN code above, then change `TARGET_MEAN` to a different value
(try `-2.0`, or `10.0`) and re-run. Confirm `gen_shift` moves to chase
whatever new target you pick, and note whether `disc_acc` follows the same
rise-then-fall pattern.

**Stretch:** induce mode collapse directly. Change `sample_real` to draw
from *two* separated clusters instead of one — for example, with 50%
probability return a sample from `N(-4, 0.3)` and with 50% probability from
`N(4, 0.3)` — but leave the generator exactly as it is (`z * scale +
shift`, one shift and one scale, unable to represent two separate clusters
no matter how it's trained). Run training and watch `gen_shift` settle
near one cluster or land somewhere in between; either way, print 10 samples
from `generate()` at the end and compare their spread to the true two-
cluster data. This is mode collapse forced by the generator's own limited
capacity — a real GAN generator (many parameters, not two) can also mode-
collapse even when it technically *has* enough capacity to cover every
cluster.

## Go deeper

- [Goodfellow et al., "Generative Adversarial Networks"](https://arxiv.org/abs/1406.2661) — the 2014 paper that introduced GANs and the minimax formulation this lesson builds on.
- [Google Machine Learning: GAN course](https://developers.google.com/machine-learning/gan) — a free interactive course covering the generator/discriminator loop, loss functions, and common failure modes including mode collapse.
- [PyTorch: DCGAN Tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) — the official tutorial training a real convolutional GAN on face images, the basis for this lesson's illustrative generator.
- [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) — a rigorous but readable walkthrough of why GAN training is unstable and the fixes later research introduced.

**Next:** [Speech & Audio AI](48-speech-audio.md)
