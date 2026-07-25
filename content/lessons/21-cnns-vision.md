---
title: "CNNs & Vision"
stage: 4
order: 21
minutes: 55
difficulty: intermediate
prerequisites: ["pytorch"]
tags: ["deep-learning", "computer-vision", "cnn"]
summary: "Convolution as a learned filter, stride, padding and pooling, how a CNN's shapes flow layer to layer, and transfer learning with a pretrained backbone."
---

# CNNs & Vision

## Why this matters

Feed a 224x224 RGB image into the `nn.Linear` layers from the last two
lessons and the first layer alone needs a weight for every one of its
224*224*3 = 150,528 input values, times however many hidden units you want
— tens of millions of weights before the network has learned anything, and
none of them know that a pixel's neighbours matter more than a pixel on the
other side of the image. **Convolutional neural networks (CNNs)** fix this
by building in an assumption that's true for almost every image: local
patterns — edges, textures, corners — matter regardless of where in the
image they appear, so the same small filter can be reused at every
position instead of learning a separate weight per pixel. This lesson is
about that one idea, the layer types built around it, and how to stand on
the shoulders of a network someone else already trained.

## The concept

**Convolution is a small filter, applied at every position in the image.**
A **kernel** (or filter) is a small grid of learnable numbers — 3x3 is
typical. Convolving it with an image means: slide it over every position,
and at each position, multiply the kernel elementwise with the patch of
image underneath it and sum the result into a single number. That produces
a **feature map** — one number per position, recording how strongly that
kernel's pattern matched the image there. The "In code" section below hand
-codes exactly this with a fixed, hand-designed edge-detecting kernel; in a
real CNN, the numbers *inside* the kernel are weights, learned by
backpropagation, the same way every other weight in this roadmap is
learned. Early layers tend to learn kernels that respond to edges and
colour gradients; deeper layers combine those into detectors for textures,
parts, and eventually whole objects — the lesson 00 intuition about layers
building up abstraction, now with a concrete mechanism.

**A convolutional layer has many kernels, not one.** Each kernel produces
its own feature map, so a layer with 16 kernels turns one input image into
16 output feature maps (called **channels**), each having learned to
respond to a different pattern. Because the same kernel is reused at every
position, a convolutional layer has vastly fewer weights than a fully
connected layer would for the same input size — a 3x3 kernel over 3 input
channels has `3*3*3 = 27` weights, whatever the image's height and width.

**Stride and padding control the output size.** **Stride** is how many
pixels the kernel moves between positions — stride 1 checks every position;
stride 2 skips every other one, halving the output's height and width.
**Padding** adds a border of zeros around the input before convolving, most
often specifically so the output feature map comes out the *same* height
and width as the input (`padding = (kernel_size - 1) / 2` for stride 1) —
without it, every convolution shrinks the image slightly, since a kernel
can't center on a pixel too close to the edge. The "In code" section prints
the exact output shape for several stride/padding combinations so you can
see this arithmetic rather than derive it abstractly.

**Pooling shrinks the feature map on purpose.** A **max pooling** layer
slides a window (2x2 is typical) over the feature map and keeps only the
largest value in each window, discarding the rest — no learnable weights at
all. This does two things: it shrinks the spatial size (a 2x2 max pool
halves both height and width, quartering the total values), and it makes
the network slightly less sensitive to *exactly* where a feature appeared,
only that it appeared somewhere in that window. Stacking
`conv -> activation -> pool` blocks is the classic CNN pattern: each block
shrinks the spatial dimensions while typically *growing* the channel count,
trading "where exactly" for "what, more abstractly."

**Typical architectures chain these blocks, then flatten.** A CNN for
classification is usually several `conv -> ReLU -> pool` blocks, shrinking
a large image down to a small grid of many channels, followed by a
`flatten` (turning that 3D grid into one long vector) and one or more
`nn.Linear` layers — the fully-connected layers from the last two lessons —
to turn those learned features into class scores. Everything before the
flatten is feature extraction; everything after is the same classifier
you've already built twice.

**Transfer learning reuses a network someone else already trained.** A
model trained on a huge dataset like ImageNet (1.2 million photos, 1000
categories) has already learned early- and mid-layer kernels — edges,
textures, common shapes — that are useful for almost any image task, not
just the one it was trained on. Transfer learning takes that trained
network (the **backbone**), **freezes** its existing weights (so training
doesn't disturb what it already learned), replaces the final classification
layer with a new one sized for your problem, and trains only that new
layer. With a good backbone, this frequently beats training a CNN from
scratch on a small dataset, because it starts from millions of images' worth
of already-learned structure instead of random weights.

## In code

Convolution, by hand, with a fixed kernel designed to detect vertical
edges — no training involved, just the elementwise-multiply-and-sum
operation itself:

```python
import numpy as np

# A tiny 6x6 "image": a bright vertical edge down the middle
image = np.array([
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
], dtype=float)

# A hand-designed vertical-edge-detecting kernel (this is what a conv layer LEARNS)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=float)

def convolve2d(img, k):
    kh, kw = k.shape
    out_h, out_w = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * k)   # elementwise multiply + sum: one "neuron" per position
    return out

result = convolve2d(image, kernel)
print("input image:\n", image)
print("\nkernel (vertical edge detector):\n", kernel)
print("\nconvolution output (feature map):\n", result)
```

```
input image:
 [[0. 0. 0. 1. 1. 1.]
 [0. 0. 0. 1. 1. 1.]
 [0. 0. 0. 1. 1. 1.]
 [0. 0. 0. 1. 1. 1.]
 [0. 0. 0. 1. 1. 1.]
 [0. 0. 0. 1. 1. 1.]]

kernel (vertical edge detector):
 [[-1.  0.  1.]
 [-1.  0.  1.]
 [-1.  0.  1.]]

convolution output (feature map):
 [[0. 3. 3. 0.]
 [0. 3. 3. 0.]
 [0. 3. 3. 0.]
 [0. 3. 3. 0.]]
```

The feature map lights up (value `3`) exactly where the kernel crosses the
vertical edge, and is `0` everywhere the image is flat — this kernel really
does detect vertical edges, with no learning involved yet.

Stride and padding, read directly off `nn.Conv2d`'s output shape:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

x = torch.randn(1, 3, 32, 32)   # (batch, channels, height, width) -- one 32x32 RGB image
print("input:                          ", x.shape)

conv_default = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
print("Conv2d(k=3, stride=1, pad=0):   ", conv_default(x).shape)

conv_padded = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
print("Conv2d(k=3, stride=1, pad=1):   ", conv_padded(x).shape)

conv_strided = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
print("Conv2d(k=3, stride=2, pad=1):   ", conv_strided(x).shape)

pool = nn.MaxPool2d(kernel_size=2)
print("MaxPool2d(k=2) on padded output:", pool(conv_padded(x)).shape)
```

```
input:                           torch.Size([1, 3, 32, 32])
Conv2d(k=3, stride=1, pad=0):    torch.Size([1, 8, 30, 30])
Conv2d(k=3, stride=1, pad=1):    torch.Size([1, 8, 32, 32])
Conv2d(k=3, stride=2, pad=1):    torch.Size([1, 8, 16, 16])
MaxPool2d(k=2) on padded output: torch.Size([1, 8, 16, 16])
```

`pad=0` shrinks 32 to 30 (the kernel can't center on the outer pixel);
`pad=1` exactly cancels that shrinkage; `stride=2` halves it on top of
padding; and pooling halves it again, independently.

A small CNN, with shapes traced through every layer via a forward pass —
this is the architecture the exercise below trains for real:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 32x32 -> 32x32, 16 filters
        self.pool = nn.MaxPool2d(2)                                 # halves height and width
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16x16 -> 16x16, 32 filters
        self.fc = nn.Linear(32 * 8 * 8, n_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # -> (B, 16, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))   # -> (B, 32, 8, 8)
        x = x.flatten(start_dim=1)                  # -> (B, 32*8*8)
        return self.fc(x)                            # -> (B, n_classes)

model = SmallCNN()
batch = torch.randn(4, 3, 32, 32)   # 4 CIFAR-10-sized images
out = model(batch)
print("input batch shape: ", batch.shape)
print("output shape:      ", out.shape, "  (4 images, 10 class scores each)")
print("total trainable parameters:", sum(p.numel() for p in model.parameters()))
```

```
input batch shape:  torch.Size([4, 3, 32, 32])
output shape:       torch.Size([4, 10])   (4 images, 10 class scores each)
total trainable parameters: 25578
```

Training `SmallCNN` on the real CIFAR-10 dataset (60,000 32x32 images, 10
classes) looks exactly like last lesson's training loop, just with a
`DataLoader` feeding in batches of images instead of the whole dataset at
once:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = SmallCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} done")
```

This is real, runnable code, shown here as **illustrative rather than
executed** — training on the full 50,000-image training set genuinely takes
minutes to hours depending on hardware, too long to run inside this lesson,
and printing a fabricated loss curve would be worse than printing none.
What to expect if you run it yourself: `CrossEntropyLoss` on 10 balanced
classes starts near `ln(10) ≈ 2.3` (the loss of guessing uniformly at
random) and, over several epochs, a small CNN like this one typically works
its way down to a test accuracy somewhere in the 60–70% range — respectable
for a from-scratch model this small, well short of the 90%+ that deeper,
purpose-built architectures reach on the same dataset.

Transfer learning: loading a real backbone pretrained on ImageNet, freezing
it, and swapping in a new head sized for a 10-class problem — no training
here either, but every other line, including the download and the forward
pass, actually ran:

```python
import torch
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(0)

# A backbone pretrained on ImageNet (1.2M photos, 1000 classes) -- its early
# layers already recognise edges, textures and shapes, which transfer to
# almost any other image task.
backbone = models.resnet18(weights="IMAGENET1K_V1")

# Freeze every existing weight: we don't want to destroy what it already learned
for param in backbone.parameters():
    param.requires_grad = False

before = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

# Replace the final classification layer with a fresh, trainable one sized
# for our problem (CIFAR-10 has 10 classes; ImageNet's had 1000)
n_features = backbone.fc.in_features
backbone.fc = nn.Linear(n_features, 10)

after = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
total = sum(p.numel() for p in backbone.parameters())

print("trainable parameters before replacing the head:", before)
print("trainable parameters after replacing the head: ", after)
print("total parameters in the model:                 ", total)

backbone.eval()
with torch.no_grad():
    dummy_batch = torch.randn(4, 3, 224, 224)   # resnet18's expected input size
    logits = backbone(dummy_batch)
print("\ninput shape: ", dummy_batch.shape)
print("output shape:", logits.shape, " (4 images, 10 class scores each)")
```

```
trainable parameters before replacing the head: 0
trainable parameters after replacing the head:  5130
total parameters in the model:                  11181642
```
```
input shape:  torch.Size([4, 3, 224, 224])
output shape: torch.Size([4, 10])  (4 images, 10 class scores each)
```

Freezing left `0` trainable parameters; swapping in a fresh `nn.Linear` for
10 classes added exactly `5130` trainable ones back (`512 * 10` weights plus
10 biases) out of `11,181,642` total — you'd be training well under 0.05% of
the network, standing entirely on the other 99.95%'s already-learned
features.

## Build this

Run the CIFAR-10 training code above yourself (adjust `epoch` to something
small, like 2, for a first pass) and record: the loss printed at the start
of training versus the end, and roughly how long one epoch took on your
hardware. Then load a pretrained `resnet18` as shown above, freeze it,
replace the head, and fine-tune *only* the new head on CIFAR-10 for the same
number of epochs (you'll need to resize CIFAR-10's 32x32 images up to 224x224
with `T.Resize(224)` in your `transform`, since that's the input size
`resnet18` expects). Compare final training loss and, if you have time,
accuracy on a held-out test split, between the from-scratch `SmallCNN` and
the frozen-backbone model.

**Stretch:** unfreeze just the *last* convolutional block of the backbone
(leave everything earlier frozen) and fine-tune it along with the head, at a
smaller learning rate than the head uses. This is called **fine-tuning**
rather than pure feature extraction, and it's the technique Stage 5 uses
again, at a much larger scale, to adapt pretrained language models.

## Go deeper

- [CS231n: Convolutional Neural Networks notes](http://cs231n.github.io/convolutional-networks/) — Stanford's course notes on convolution, pooling, and classic architectures, the standard reference for this material.
- [PyTorch: Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — the official CIFAR-10 CNN tutorial this lesson's training code is based on.
- [PyTorch: Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — the official tutorial on freezing a backbone and fine-tuning, with a full training run included.
- [d2l.ai: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html) — a from-scratch treatment of convolution, padding, stride and pooling, plus classic architectures like LeNet.

**Next:** [Sequence Models](22-sequence-models.md)
