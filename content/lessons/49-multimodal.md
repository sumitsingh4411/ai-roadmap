---
title: "Multimodal AI (CLIP & Vision-Language Models)"
stage: 5
order: 49
minutes: 50
difficulty: advanced
prerequisites: ["transformers"]
tags: ["multimodal", "clip", "vision-language"]
summary: "One model, many senses — how CLIP puts images and text in the same space, enabling zero-shot classification and image search."
---

# Multimodal AI (CLIP & Vision-Language Models)

## Why this matters

The embeddings lesson showed that a model can turn *text* into a vector
where distance means "how similar in meaning." The previous lesson showed
that audio can be turned into a frequency representation a Transformer can
read. **Multimodal AI** asks: what if a text vector and an image vector
lived in the *same* space? Once they do, a huge amount of new behaviour
falls out almost for free — classifying images by typing a label instead of
training a classifier, searching photos with a sentence, and the
vision-capable chat models you've probably already used. This lesson builds
the core idea, CLIP's shared embedding space, from scratch in NumPy.

## The concept

**What "multimodal" means.** A **modality** is a kind of input or output —
text, images, audio, video. Every model so far in this roadmap has been
unimodal: text in, text out (an LLM), or images in, a label out (a CNN
classifier). A **multimodal** model handles more than one modality, either
by accepting more than one as input, producing more than one as output, or
both. The interesting engineering question is always the same: modalities
start as completely different kinds of data — pixels, audio samples,
token IDs — so how do you get a single model to relate them to each other
at all?

**CLIP's answer: a shared embedding space.** CLIP (Contrastive
Language-Image Pretraining, from OpenAI) trains two separate encoders at
once: an **image encoder** (a CNN or Vision Transformer) that maps an image
to a vector, and a **text encoder** (a Transformer, the same kind of
architecture as two lessons ago) that maps a caption to a vector, both
producing vectors of the *same* fixed length. Critically, the two encoders
are trained *together*, with one objective: for a real (image, caption)
pair — a photo and its actual caption, scraped from the web at huge scale —
push the image vector and the text vector to point in nearly the same
direction. For a *mismatched* pair — this photo with somebody else's
caption — push their vectors apart. This is **contrastive learning**: the
model doesn't learn "what a cat is" from a labelled dataset the way Stage 4's
CNNs did; it learns "this image's vector and this caption's vector should
be close" from hundreds of millions of naturally-occurring (image, text)
pairs, and a rich notion of visual concepts emerges as a side effect of
getting that matching task right at scale.

**Why this enables zero-shot classification.** Once an image and the
*words describing it* land near each other by direction, you can classify
an image without ever training a classifier for your specific categories:
embed the image once, embed each candidate label as a short piece of text
(often a template like "a photo of a {label}"), and compare with cosine
similarity — same formula as the embeddings lesson. Whichever label's
text vector is closest to the image vector is the prediction. This is
**zero-shot** because the model was never trained on "cat vs. dog vs. car"
as a labelled task; it was trained on the general image-caption matching
objective, and classification into *any* set of labels you can name in
words falls out of that at inference time, with no retraining.

**The same idea powers image search.** Flip the comparison around: embed a
text query ("a dog running on a beach") and compare it against the
embeddings of every image in a collection, computed once and stored ahead
of time. The closest image vectors are the best matches — text-to-image
search, using nothing but the shared space and cosine similarity, the exact
same brute-force approach the embeddings lesson used for text-to-text
search.

**Vision-language models, briefly.** CLIP relates images and text but
doesn't generate language about an image. A **vision-language model**
(VLM) — the kind of model behind GPT-4V-style "upload an image and ask
about it" — goes further: it feeds image information (often via a CLIP-like
encoder, or something trained similarly) into an LLM's input, alongside
text tokens, so the LLM can attend over both and generate a full text
response — describe the image, answer a question about it, read text in a
photo. The image encoder's output vectors get treated as if they were
extra "tokens" the decoder can attend to via the same self-attention
mechanism from the Transformers lesson, projected into the LLM's embedding
space so shapes line up. The shared-embedding idea from CLIP and the
"treat non-text as tokens the Transformer attends to" idea are related but
distinct: CLIP's contribution is the aligned space itself; VLMs are one way
of putting that alignment to use inside a generative model.

## In code

Real CLIP needs a pretrained model and actual images, so that part is
illustrative below. What's fully runnable is the core idea it's built on —
a shared space where matching image and text vectors land close together —
using hand-placed toy vectors instead of a trained encoder's real output:
`pip install numpy`

```python
import numpy as np

rng = np.random.default_rng(0)

# Pretend "shared embedding space" -- in real CLIP these come from trained
# image and text encoders; here we hand-place them so the geometry is clear.
# Each image vector sits near its true label's direction, plus noise --
# standing in for "a real image encoder's imperfect but roughly correct output."
labels = ["cat", "dog", "car"]
label_directions = {
    "cat": np.array([1.0, 0.0, 0.0]),
    "dog": np.array([0.0, 1.0, 0.0]),
    "car": np.array([0.0, 0.0, 1.0]),
}
text_embeddings = dict(label_directions)

image_true_labels = ["cat", "dog", "car", "cat", "car"]
image_embeddings = np.array([
    label_directions[label] + rng.normal(0, 0.2, size=3)
    for label in image_true_labels
])


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("Zero-shot classification by nearest text label:\n")
correct = 0
for i, (img_vec, true_label) in enumerate(zip(image_embeddings, image_true_labels)):
    scores = {name: cosine_similarity(img_vec, text_vec) for name, text_vec in text_embeddings.items()}
    predicted = max(scores, key=scores.get)
    correct += predicted == true_label
    score_str = ", ".join(f"{name}={score:.3f}" for name, score in scores.items())
    print(f"image {i} (true={true_label:>3}): predicted={predicted:>3}  [{score_str}]")

print(f"\nAccuracy: {correct}/{len(image_true_labels)}")
```

```
Zero-shot classification by nearest text label:

image 0 (true=cat): predicted=cat  [cat=0.992, dog=-0.026, car=0.124]
image 1 (true=dog): predicted=dog  [cat=0.023, dog=0.996, car=0.081]
image 2 (true=car): predicted=car  [cat=0.284, dog=0.206, car=0.936]
image 3 (true=cat): predicted=cat  [cat=0.986, dog=-0.165, car=0.011]
image 4 (true=car): predicted=car  [cat=-0.526, dog=-0.049, car=0.849]

Accuracy: 5/5
```

No classifier was trained on "cat vs. dog vs. car" — every prediction here
is just "which text vector is this image vector closest to." That's the
entire mechanism behind zero-shot classification: the hard work is getting
image and text vectors to share a space in the first place; comparison at
inference time is just cosine similarity.

Real CLIP, doing the same comparison with actual trained encoders (needs
`pip install open_clip_torch pillow` and an actual image file):

```python
# ILLUSTRATIVE -- requires open_clip_torch, torch, an image file, and downloading pretrained weights
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("photo.jpg")).unsqueeze(0)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
text = tokenizer(labels)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze(0)

print(dict(zip(labels, similarity.tolist())))
```

## Build this

Add a fourth label — say `"tree": np.array([1.0, 1.0, 1.0]) / np.sqrt(3)`
(normalized so it's comparable to the others) — to `label_directions`, add
at least two images whose true label is `"tree"` to `image_true_labels`,
and re-run the classification loop. Confirm the new label gets picked up
correctly and check whether accuracy holds at 100% or whether the added
category makes any existing image's classification less confident (look at
how close the runner-up score gets, not just which label wins).

**Stretch:** build a tiny text-to-image search. Reuse `image_embeddings`
and `text_embeddings` from the code above, but this time go the other
direction: for a text query embedding (any of the label vectors, or a new
one of your own), compute cosine similarity against every row of
`image_embeddings`, sort descending, and print the ranked results
image-by-image with their scores — the same brute-force nearest-neighbor
search from the embeddings lesson, just querying with text and searching
over images instead of text-to-text.

## Go deeper

- [OpenAI: CLIP — Connecting Text and Images](https://openai.com/index/clip/) — the original announcement, with the contrastive training diagram this lesson describes in words.
- [Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020) — the original CLIP paper.
- [Hugging Face: CLIP documentation](https://huggingface.co/docs/transformers/model_doc/clip) — how to load and run real CLIP with the `transformers` library.
- [OpenAI: GPT-4V(ision) System Card](https://openai.com/index/gpt-4v-system-card/) — what a production vision-language model can do and its documented limitations.
- [`open_clip` GitHub repository](https://github.com/mlfoundations/open_clip) — the open-source library used in this lesson's illustrative snippet, with multiple pretrained CLIP variants.

**Next:** [MLOps Basics](31-mlops-basics.md)
