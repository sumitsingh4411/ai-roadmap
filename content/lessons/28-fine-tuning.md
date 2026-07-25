---
title: "Fine-tuning"
stage: 5
order: 28
minutes: 60
difficulty: advanced
prerequisites: ["how-llms-work", "pytorch"]
tags: ["fine-tuning", "lora"]
summary: "When fine-tuning beats RAG or prompting, full fine-tuning vs LoRA/PEFT, dataset preparation, and evaluating the result, with a real LoRA parameter-count demo."
---

# Fine-tuning

## Why this matters

There are now two ways, from earlier in this stage, to specialize a
general-purpose LLM without touching its weights: prompting (lesson 25)
and retrieval (lesson 27). **Fine-tuning** is the third, heavier option —
actually updating the model's parameters on your own data. It's reached
for more often than it should be, and it's by far the most expensive of
the three to get wrong, so this lesson leads with when *not* to use it,
before getting into the technique — LoRA — that made it practical outside
a handful of large labs.

## The concept

**When fine-tuning is the right answer, and when it isn't.** If the model
needs to know new *facts* — your company's current documentation, this
week's data, anything that changes over time — reach for RAG, not
fine-tuning. Fine-tuning teaches a model to imitate a pattern demonstrated
by examples; it's a notoriously unreliable way to inject specific facts,
because the model tends to blend a newly-trained-on fact with everything
else it already learned rather than storing it like a lookup table, and
fine-tuned-in facts go stale exactly the way retrieved facts would — except
now fixing them means retraining instead of just re-indexing. If the model
needs to follow a specific format, tone, or behavior more reliably, try
prompting first — it's free, instant, and fully reversible — then add RAG
if the gap is really about missing task-specific knowledge, and only reach
for fine-tuning if real iteration on prompting genuinely can't get you
reliable enough behavior, or the volume is high enough that baking the
pattern into the weights saves meaningful prompt tokens and latency at
scale. Fine-tuning earns its cost when the model needs a new *skill* that's
easy to demonstrate but hard to describe in words — a very particular code
style, a structured-extraction format across thousands of real edge cases,
mimicking a specific voice — and you have enough quality labeled examples
to teach it. The cheap-to-expensive order to try, in general: prompting,
then RAG, then fine-tuning — and they're not mutually exclusive; a
fine-tuned model still needs good prompts and can still use RAG.

**Full fine-tuning vs LoRA/PEFT.** **Full fine-tuning** updates every one
of the model's parameters, the same optimization loop as pretraining
(lesson 24), just run on your much smaller dataset. It's correct in
principle but expensive in practice: training requires storing gradients
and optimizer state for every parameter (Adam, the standard optimizer,
keeps two extra numbers *per parameter* on top of the parameter itself),
which can multiply memory requirements several times over the model's own
size — often infeasible on a single consumer GPU for anything but small
models — and it produces an entirely new, full-size copy of the model per
fine-tune. **PEFT** (Parameter-Efficient Fine-Tuning) is a family of
techniques that instead fine-tune a small number of *new* parameters while
keeping the pretrained weights frozen, capturing most of fine-tuning's
benefit at a fraction of the memory, compute, and storage cost. **LoRA**
(Low-Rank Adaptation) is the most widely used PEFT method, and it's worth
understanding exactly, since it's easy to describe imprecisely.

**LoRA, precisely.** For a frozen pretrained weight matrix `W`, LoRA does
*not* modify `W`. It adds a parallel, trainable low-rank update: two small
matrices, `A` (shape `r x in_features`) and `B` (shape `out_features x
r`), where the **rank** `r` is a small number — 4, 8, 16 — far smaller than
`in_features` or `out_features`. Only `A` and `B` are trained;
`W.requires_grad` stays `False` throughout. The effective weight used at
inference is `W + (alpha/r) * (B @ A)` — so the *update* to the weight,
`ΔW = B @ A`, is constrained to be **low-rank**: its rank is at most `r`,
even though the original `W` has full rank. That's a real, strong
assumption — it claims the adaptation a task needs lives in a small
subspace of everything `W` could possibly do — and empirically it holds up
well across a huge range of fine-tuning tasks. Because `A` and `B`
together have `r * (in_features + out_features)` parameters instead of
`in_features * out_features`, the number of *trainable* parameters — and
the optimizer state you need for them — drops by orders of magnitude,
which the "In code" demo measures directly on a real layer. `B` is
initialized to all zeros (`A` to small random values), so `ΔW = B @ A =
0` exactly at the start of training: the adapted model behaves *identically*
to the frozen base model before a single training step happens, a safe,
predictable starting point that the demo below also verifies with a real
number. LoRA is typically applied only to specific weight matrices inside
each layer — commonly the attention projection matrices `Wq`, `Wk`, `Wv`,
`Wo` from lesson 23 — not to every parameter in the model; which matrices
to adapt is itself a hyperparameter. After training, the adapter (`A` and
`B` for every adapted layer) is small — megabytes, not gigabytes — and can
ship separately from the frozen base weights, or be merged into `W` once
(`W' = W + B @ A`) for zero extra inference cost. (**QLoRA** combines LoRA
with a quantized, lower-precision — often 4-bit — frozen base model,
shrinking memory further; it's the common recipe for fine-tuning large
open models on a single consumer GPU.)

**Dataset preparation.** For fine-tuning, quality and format consistency
matter more than raw quantity: you need clean (prompt, ideal-completion)
pairs that genuinely represent what you want the model to do at inference
time, deduplicated, checked for label noise or contradictory examples, and
split into train and validation sets. A model trained on inconsistent or
noisy examples will confidently learn the noise right along with the
signal — it has no way to tell the difference, the same way it has no way
to tell a plausible-but-false completion from a true one (lesson 24,
again).

**Evaluating the result.** You cannot eyeball a handful of outputs and
ship it — the same argument lesson 30 makes about prompts applies here,
to fine-tuning results. Hold out a validation set the model never trained
on. Measure it with the metric you actually care about in production, not
just training loss trending down — loss is a proxy, and a model can
overfit: lower training loss while held-out behavior gets *worse*.
Critically, compare the fine-tuned model's validation performance against
the baseline — the un-fine-tuned model, well-prompted, possibly with RAG —
on the exact same validation set, to confirm fine-tuning actually bought
something over the cheaper alternatives you were supposed to try first.

## In code

A real, runnable PyTorch demo of a LoRA layer's shape and parameter-count
math — no pretrained model download needed, since the point is the
adapter's arithmetic, not what it learns:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class LoRALinear(nn.Module):
    """A frozen base linear layer plus a trainable low-rank adapter.
    Computes W @ x + (alpha/r) * (B @ A) @ x -- the base weight W never
    changes; only the small A and B matrices are trained.
    """
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)  # freeze the pretrained weights

        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))  # B starts at zero
        self.scale = alpha / r

    def forward(self, x):
        base_out = self.base(x)
        delta = (x @ self.A.T) @ self.B.T   # this is x @ (BA)^T, i.e. delta_W = B @ A
        return base_out + self.scale * delta

in_features, out_features, r = 4096, 4096, 8
layer = LoRALinear(in_features, out_features, r=r)

base_params = layer.base.weight.numel()
a_params = layer.A.numel()
b_params = layer.B.numel()
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())

print(f"Base weight W: {in_features} x {out_features} = {base_params:,} params (frozen, requires_grad=False)")
print(f"LoRA A: {r} x {in_features} = {a_params:,} params (trainable)")
print(f"LoRA B: {out_features} x {r} = {b_params:,} params (trainable)")
print(f"Trainable params: {trainable:,}")
print(f"Total params (base + adapter): {total:,}")
print(f"Trainable fraction: {100 * trainable / total:.3f}%")

x = torch.randn(2, in_features)
out = layer(x)
print("\ninput shape:", x.shape, " output shape:", out.shape)

print("\nAt initialization, B is all zeros, so the adapter starts as a no-op:")
base_only = layer.base(x)
print("max |output - base_output| at init:", (out - base_only).abs().max().item())
```

```
Base weight W: 4096 x 4096 = 16,777,216 params (frozen, requires_grad=False)
LoRA A: 8 x 4096 = 32,768 params (trainable)
LoRA B: 4096 x 8 = 32,768 params (trainable)
Trainable params: 65,536
Total params (base + adapter): 16,842,752
Trainable fraction: 0.389%

input shape: torch.Size([2, 4096])  output shape: torch.Size([2, 4096])

At initialization, B is all zeros, so the adapter starts as a no-op:
max |output - base_output| at init: 0.0
```

A single 4096x4096 layer has 16,777,216 weights. Its rank-8 LoRA adapter
has 65,536 trainable parameters — **0.389% of the base layer** — and the
`max |output - base_output|` line confirms, with a real measured `0.0`,
that before any training happens the adapted layer computes exactly the
same output as the frozen base layer alone, exactly as "The concept"
claimed from the `B = 0` initialization.

Actually fine-tuning a real pretrained LLM — full or LoRA — needs a real
pretrained model's weights (a multi-gigabyte download) and real training
compute, which this lesson's lightweight, no-key environment doesn't have.
The shape of a full fine-tuning loop is exactly the canonical PyTorch
training loop from lesson 20 — zero gradients, forward pass, compute loss,
backward pass, optimizer step — run over your own (prompt, completion)
dataset instead of a toy problem:

```python
# ILLUSTRATIVE -- needs a real pretrained checkpoint and real compute
# (GPU, meaningful CPU time for anything but a tiny toy model). No numbers
# below are real; this is the training loop's SHAPE, not a captured run.
for epoch in range(num_epochs):
    for batch in dataloader:                 # your (prompt, completion) pairs
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss                  # next-token prediction loss, lesson 24
        loss.backward()
        optimizer.step()
```

## Build this

Fine-tune a small open model with LoRA on a toy dataset, for real, on your
own machine. Install `transformers` and `peft` (`pip install transformers
peft`), load a small model from Hugging Face that will actually train on
CPU in reasonable time for a genuinely tiny dataset (a small GPT-2 variant
is a reasonable starting point), wrap its linear layers with
`peft.LoraConfig` and `get_peft_model`, and call
`model.print_trainable_parameters()` — confirm the fraction it reports is
small, echoing this lesson's math with a real model instead of the toy
`LoRALinear` above. Write 8–10 of your own (prompt, completion) pairs
around a single narrow pattern (a consistent format, a specific style),
fine-tune for a few epochs, and compare the model's output on a held-out
prompt before and after training. This lesson didn't run this step for
you, on purpose, to keep the environment lightweight — expect a real, if
modest, behavior shift, not a dramatic one from a handful of examples.

**Stretch:** change the LoRA rank `r` in your real run and re-print
`model.print_trainable_parameters()`. Confirm the trainable-parameter
count scales the way `r * (in_features + out_features)` predicts, using
your own real numbers instead of this lesson's.

## Go deeper

- [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) — the original paper; short, readable, and worth reading once the mechanism above is solid.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/en/index) — the library referenced in "Build this," covering LoRA and the rest of the PEFT family.
- [Sebastian Raschka: Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — hard-won, empirical answers to the hyperparameter questions this lesson raises but doesn't fully resolve (which layers, what rank, what learning rate).
- [Hugging Face NLP Course: Fine-tuning a pretrained model](https://huggingface.co/learn/nlp-course/chapter3/1) — a hands-on introduction to the full fine-tuning workflow this lesson's LoRA layer is a lighter-weight alternative to.

**Next:** [AI Agents](29-ai-agents.md)
