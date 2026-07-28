---
title: "Diffusion Models & Image Generation"
stage: 5
order: 40
minutes: 55
difficulty: advanced
prerequisites: ["transformers"]
tags: ["diffusion", "generative-ai", "image-generation"]
summary: "How Stable-Diffusion-style models turn noise into images — the forward noising process, learning to denoise, and why it works."
---

# Diffusion Models & Image Generation

## Why this matters

Every generative model needs a way to turn "nothing" into "something that
looks like the training data." A language model does it one token at a
time (Stage 5's earlier lessons). Image models that produce Stable
Diffusion or Midjourney-quality output do it a completely different way:
they start from pure random noise — static, no structure at all — and
repeatedly refine it, subtracting a little noise at a time, until a
coherent image emerges. That refining model isn't hand-designed; it's
*trained* by first doing the easy, deterministic part backwards — taking
real images and slowly destroying them with noise — and teaching a network
to reverse each tiny step. This lesson builds that idea from the ground
up: the forward process that adds noise, why training the model to predict
the noise is enough to reverse it, and where text prompts and latent space
fit into the picture actually used by real image generators.

## The concept

**The forward process: destroy an image, one small step at a time.** Take
a real image `x0` (or, in this lesson's runnable toy, a real 2-D signal).
Define `T` steps. At each step `t`, mix in a small amount of Gaussian
noise, controlled by a per-step **noise schedule** `beta_t` — a small
number, typically increasing slightly from step 1 to step `T`, that says
"how much noise to add right now." After enough steps, the original signal
is completely gone and `x_T` is indistinguishable from pure random noise.
Critically, this forward process is **fixed** — no learning happens here,
it's just repeated noise-mixing according to a formula you choose in
advance.

**The one-step-to-anywhere shortcut.** Because each step just mixes in
Gaussian noise, there's a closed-form way to jump directly from `x0` to
`x_t` at *any* step `t`, without simulating every step in between. Define
`alpha_t = 1 - beta_t` and `alpha_bar_t` as the running product
`alpha_1 * alpha_2 * ... * alpha_t`. Then:

```
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
```

where `noise` is drawn fresh from a standard normal distribution. Read
this as a weighted blend: `sqrt(alpha_bar_t)` is how much of the *original
signal* survives at step `t`, and `sqrt(1 - alpha_bar_t)` is how much
*noise* has been mixed in. As `t` grows, `alpha_bar_t` shrinks toward 0 —
signal fades out, noise takes over — which is exactly what the "In code"
section below measures directly, as a signal-to-noise ratio that decays
toward zero.

**Training target: predict the noise, not the image.** Here is the trick
that makes reversal learnable. At training time, take a real image `x0`,
pick a random step `t`, sample a random `noise`, and compute `x_t` with
the formula above — you now have a noisy image *and* you know exactly
which noise produced it, because you added it yourself. Train a network
`eps_theta(x_t, t)` to predict that `noise` from the noisy image and the
step number, using ordinary MSE loss between the predicted and actual
noise — the same loss-and-gradient-descent loop from the PyTorch lesson,
nothing exotic about the optimisation itself. Predicting the noise turns
out to be equivalent to predicting a direction back toward the data
manifold: subtract a scaled copy of the predicted noise, and what remains
looks a little more like a real image than `x_t` did.

**Sampling: reverse the process, one predicted step at a time.** To
generate a *new* image, start from `x_T` — literally sample pure Gaussian
noise, no real image involved at all. Feed it, and the step number `T`,
into the trained network to get a predicted noise. Use that prediction to
compute a slightly-less-noisy `x_{T-1}` (the exact DDPM update is
`x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) *
eps_theta(x_t, t)) + sigma_t * z`, where `z` is fresh random noise added
back in at every step except the last — this extra noise is what makes
sampling stochastic, so the same starting noise doesn't always produce the
exact same image). Repeat for `t = T-1, T-2, ..., 1`. After `T` predicted
denoising steps, `x_0` is a brand new sample that looks like it came from
the training distribution — built entirely out of a network that only
ever learned to predict noise.

**Where text prompts, CLIP, and latent space fit in.** Three practical
additions turn this into Stable Diffusion:

- **Latent diffusion.** Running the forward/reverse process on full-size
  pixel images is expensive. A separate autoencoder first compresses an
  image into a much smaller **latent** representation; diffusion happens
  entirely in that compressed space, and a decoder expands the final
  denoised latent back into a pixel image at the very end. This is the
  "Latent" in "Latent Diffusion Model," the architecture behind Stable
  Diffusion, and it's the main reason it's practical to run on consumer
  hardware at all.
- **Text conditioning.** The denoising network doesn't just see `x_t` and
  `t` — it also sees an embedding of the text prompt, produced by a text
  encoder. That embedding is fed in via **cross-attention** (the exact
  mechanism from the Transformers lesson: queries from the image's
  intermediate representation, keys and values from the text embedding),
  at every denoising step, steering *which* noise gets predicted so the
  final image matches the prompt instead of being an unconditioned sample.
- **CLIP.** CLIP is a model trained separately to place matching images
  and text captions near each other in a shared embedding space. Stable
  Diffusion uses CLIP's text encoder specifically because its embeddings
  already carry rich, image-relevant meaning for a prompt — a head start
  over training a text encoder from scratch alongside the diffusion model.

## In code

The full training loop needs real images and a real neural network and is
not something to fabricate output for — it's marked `# ILLUSTRATIVE`
below. What *is* fully runnable, seeded, and checked against real printed
numbers is the forward process itself: take a 2-D signal, run the exact
closed-form noising formula from above across `T` steps, and confirm the
signal-to-noise ratio decays the way the maths predicts.

```python
import numpy as np

rng = np.random.default_rng(0)

# "Data": 200 points on a 2-D spiral -- a simple 2-D signal to noise
n_points = 200
t = np.linspace(0, 4 * np.pi, n_points)
x0 = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
x0 = x0 / np.abs(x0).max()  # scale roughly into [-1, 1]

print("x0 shape:", x0.shape)
print("x0 signal power (mean squared value):", np.mean(x0 ** 2))

# Linear noise schedule: beta_t controls how much noise is mixed in at each step
T = 300
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)  # alpha_bar_t = product of alphas up to step t

# Closed-form forward process: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
# This jumps directly to any step t without simulating every step in between.
def forward_noise(x0, t_index, rng):
    ab = alpha_bars[t_index]
    noise = rng.normal(size=x0.shape)
    x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise
    return x_t, noise

print("\nstep | alpha_bar_t | signal power | noise power | SNR (signal/noise)")
for t_index in [0, 49, 99, 149, 199, 249, 299]:
    x_t, noise = forward_noise(x0, t_index, rng)
    signal_power = np.mean((np.sqrt(alpha_bars[t_index]) * x0) ** 2)
    noise_power = np.mean(((np.sqrt(1 - alpha_bars[t_index])) * noise) ** 2)
    snr = signal_power / noise_power
    print(f"{t_index:4d} | {alpha_bars[t_index]:.5f}     | {signal_power:.5f}      | {noise_power:.5f}     | {snr:.4f}")

x_final, _ = forward_noise(x0, T - 1, rng)
print("\noriginal x0[0]:      ", x0[0])
print("fully noised x_T[0]: ", x_final[0])
print("x_T mean:", x_final.mean(axis=0), " x_T std:", x_final.std(axis=0),
      "(should look close to standard-normal noise: mean~0, std~1)")
```

```
x0 shape: (200, 2)
x0 signal power (mean squared value): 0.16708542713567837

step | alpha_bar_t | signal power | noise power | SNR (signal/noise)
   0 | 0.99990     | 0.16707      | 0.00010     | 1684.8899
  49 | 0.91702     | 0.15322      | 0.08390     | 1.8263
  99 | 0.71162     | 0.11890      | 0.25056     | 0.4745
 149 | 0.46705     | 0.07804      | 0.54501     | 0.1432
 199 | 0.25911     | 0.04329      | 0.82373     | 0.0526
 249 | 0.12144     | 0.02029      | 0.88761     | 0.0229
 299 | 0.04806     | 0.00803      | 0.88352     | 0.0091

original x0[0]:       [0. 0.]
fully noised x_T[0]:  [-1.25015363  0.88045363]
x_T mean: [-0.04581853  0.00261366]  x_T std: [1.04580424 1.01583909] (should look close to standard-normal noise: mean~0, std~1)
```

Two things to notice in this real output. First, the SNR column falls
from `1684.9` at step 0 (almost pure signal — `alpha_bar_0` is `0.9999`)
to `0.0091` at step 299 (almost pure noise), exactly the decay the forward
formula predicts, with no step where it jumps around unpredictably — it's
monotonic, because `alpha_bar_t` is a running product of numbers all
slightly less than 1. Second, `x_T`'s mean and standard deviation
(`~0, ~1`) confirm the fully-noised signal really has become indistinguishable
from standard Gaussian noise — precisely the distribution sampling starts
from at generation time.

Idiomatic use of a real pretrained diffusion pipeline — this needs a GPU
and multi-gigabyte model weights, so it's illustrative rather than run
here, but it's exactly what production code looks like:

```python
# ILLUSTRATIVE -- requires `pip install diffusers torch` and a real model
# download; not run in this lesson.
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

image = pipe(
    "a watercolor painting of a lighthouse at sunset",
    num_inference_steps=30,   # roughly T steps of the reverse process, from noise to image
).images[0]
image.save("lighthouse.png")
```

Every argument here maps onto a concept above: `num_inference_steps` is
how many reverse steps to take (fewer steps is faster but lower quality —
modern samplers get good results in far fewer than the `T` used during
training), and the text prompt is encoded and fed into the denoising
network's cross-attention at every one of those steps.

## Build this

Run this lesson's forward-process code yourself, then change two things
and observe the printed SNR table each time: (1) widen the noise schedule
to `betas = np.linspace(1e-4, 0.05, T)` and see how much faster the SNR
collapses toward zero; (2) swap the spiral for a different 2-D shape (a
circle: `np.stack([np.cos(t), np.sin(t)], axis=1)`) and confirm the SNR
values at each step are unchanged — the noise schedule controls the decay
rate, the *data* does not.

**Stretch:** implement one reverse step by hand on the toy data. Pick a
late step `t` (e.g. `t=299`), get `x_t` and the true `noise` from
`forward_noise`, and — pretending you have a perfect denoiser that returns
the exact `noise` used — apply the DDPM reverse-step formula from "The
concept" to compute `x_{t-1}`. Compare it against calling `forward_noise`
directly at `t-1`: since your "denoiser" is cheating with the real noise,
the two should land close to each other, which confirms the reverse
formula genuinely undoes one step of the forward formula.

## Go deeper

- [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — the most-cited from-scratch derivation of the forward and reverse processes, with the full maths this lesson simplified.
- [Hugging Face: The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — builds a working DDPM in PyTorch line by line, training and sampling included.
- [Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) — the 2020 paper that introduced the exact forward/reverse formulation used in this lesson.
- [Jay Alammar: The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — a visual walkthrough of latent diffusion, text conditioning, and CLIP working together.
- [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/index) — the library used in this lesson's illustrative pipeline example; a natural next step for running real models.

**Next:** [Vector Databases](41-vector-databases.md)
