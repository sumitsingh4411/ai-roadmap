---
title: "Object Detection & Segmentation"
stage: 4
order: 46
minutes: 50
difficulty: advanced
prerequisites: ["cnns-vision"]
tags: ["computer-vision", "object-detection", "yolo"]
summary: "Beyond 'what's in this image?' to 'what's where?' — bounding boxes, IoU, non-max suppression, and how YOLO detects in real time."
---

# Object Detection & Segmentation

## Why this matters

The CNN from the last lesson answers one question per image: "what is
this?" It takes a photo of a street and outputs `car` — one label, full
stop. That's **classification**, and it quietly assumes the image contains
exactly one thing worth naming. Real photos don't cooperate: a street scene
has a car, three pedestrians, two traffic lights and a dog, at different
positions and different sizes, and a self-driving system that only knows
"car" without knowing *where* the car is has learned almost nothing useful.
**Object detection** answers a harder question — "what is here, and where,
for every object in the image" — by predicting a box around each thing it
finds along with its label. This lesson covers the machinery every detector
needs regardless of its exact architecture: how to score whether a
predicted box is any good, how to clean up the flood of overlapping guesses
a detector produces, and the two families of detector design — accurate-but-
slow and fast-but-good-enough — that this tradeoff has produced.

## The concept

**Detection adds "where" to "what."** A classifier's output is a vector of
class scores, one number per class. A detector's output is a *list* of
`(box, class, confidence)` triples, one per object it thinks it found — the
list can be empty (no objects), or have a dozen entries (a crowd). A
**bounding box** is usually written as four numbers: `(x1, y1, x2, y2)`, the
pixel coordinates of its top-left and bottom-right corners. Everything else
in this lesson exists to answer two questions a detector's training and
evaluation both depend on: how do you score a predicted box against the
truth, and what do you do when a detector predicts many overlapping boxes
for the same object.

**Intersection-over-Union (IoU) scores how well two boxes overlap.** Given a
predicted box and a ground-truth box, IoU is the area where they overlap
divided by the area they cover *between* them: `IoU = intersection_area /
union_area`. Two identical boxes score `1.0`; two boxes that don't touch at
all score `0.0`; typical "pretty good, not perfect" predictions land
somewhere around `0.5`–`0.8`. IoU is the single metric nearly everything
else in detection is built on: it decides whether a prediction counts as a
correct detection during evaluation (a common rule: IoU ≥ 0.5 against the
true box counts as a match), and it's the tool used to clean up duplicate
predictions, below. The "In code" section implements it directly from the
box-corner coordinates.

**A detector predicts far more boxes than there are objects.** Internally,
most detectors don't predict "here are your 3 objects" directly — they
score *many* candidate boxes (sometimes thousands, tiled across the image
at different positions and sizes) and most get a low confidence score for
"contains an object." But near a real object, several overlapping
candidates all score highly, because a box shifted five pixels left or
scaled 10% larger still mostly covers the same car. Left alone, a detector
would report ten overlapping boxes for one car instead of one.

**Non-max suppression (NMS) collapses duplicates into one box per object.**
The algorithm is a straightforward greedy loop: sort all candidate boxes by
confidence score, descending. Take the highest-scoring box, keep it, and
throw out every remaining box whose IoU with it exceeds a threshold (a
common default is `0.5`) — those are judged to be "the same object,
detected again." Repeat with whatever's left until no candidates remain.
The **IoU threshold** is the one knob that controls how aggressive this
cleanup is: a low threshold (e.g. `0.3`) suppresses anything with even
moderate overlap, risking merging two genuinely separate, nearby objects
into one; a high threshold (e.g. `0.7`) only suppresses near-duplicates,
risking leaving true duplicates behind. The "In code" section runs this
exact loop on a small set of candidate boxes at two different thresholds so
you can see the tradeoff directly, not just read about it.

**Two-stage detectors propose, then classify.** The R-CNN family (R-CNN,
Fast R-CNN, Faster R-CNN) splits detection into two steps: a first stage
proposes a set of candidate regions likely to contain *something* (without
yet saying what), and a second stage runs a classifier over each proposed
region to decide what's there and refine its box. This two-step design is
accurate — each region gets dedicated classification attention — but slow,
because the second stage reruns per proposal, and a busy image can generate
hundreds of proposals.

**One-stage detectors predict boxes and classes in a single pass.** YOLO
("You Only Look Once") and SSD skip the separate proposal stage entirely: a
single CNN pass over the whole image directly outputs a grid of candidate
boxes with class scores and confidences all at once, which then just needs
NMS to clean up. Trading the dedicated per-region attention for a single
shared pass is what makes YOLO fast enough to run on live video — the
original YOLO paper reported real-time speeds (tens of frames per second)
on hardware where two-stage detectors of the same era ran many times
slower. The general pattern — two careful stages vs. one fast unified pass —
recurs across computer vision wherever speed and accuracy trade off against
each other.

**Segmentation labels pixels instead of drawing boxes.** A bounding box is a
coarse rectangle; it includes background pixels caught inside the box along
with the object. **Semantic segmentation** labels *every pixel* in the
image with a class ("this pixel is `road`, that one is `car`") but doesn't
distinguish between two instances of the same class — two adjacent cars
both just get labelled `car`, merged into one blob. **Instance
segmentation** goes further, labelling every pixel *and* which specific
object instance it belongs to, so two adjacent cars come out as two
separate masks. Mask R-CNN is the classic instance-segmentation extension
of the two-stage R-CNN family — it adds a third output (a pixel mask) to
Faster R-CNN's box-plus-class output. More recently, Meta's **Segment
Anything Model (SAM)** takes a different approach: rather than being
trained on a fixed set of classes, it's trained to segment *any* object
given a point, box, or rough prompt, generalizing to objects it never saw a
labelled example of.

## In code

`pip install numpy` if you haven't already for a previous lesson.

Intersection-over-Union, computed directly from box corners:

```python
import numpy as np

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

box_x = [10, 10, 60, 60]
box_y = [30, 30, 80, 80]
box_z = [200, 200, 250, 250]

print(f"IoU(box_x, box_y) = {iou(box_x, box_y):.4f}   (overlapping boxes)")
print(f"IoU(box_x, box_z) = {iou(box_x, box_z):.4f}   (no overlap at all)")
print(f"IoU(box_x, box_x) = {iou(box_x, box_x):.4f}   (identical box)")
```

```
IoU(box_x, box_y) = 0.2195   (overlapping boxes)
IoU(box_x, box_z) = 0.0000   (no overlap at all)
IoU(box_x, box_x) = 1.0000   (identical box)
```

Non-max suppression: six candidate boxes in two clusters plus one isolated
box, sorted and greedily filtered by IoU against the highest-scoring survivor:

```python
import numpy as np

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

def nms(boxes, scores, iou_threshold):
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    order = scores.argsort()[::-1]   # highest confidence first

    keep = []
    while len(order) > 0:
        current = order[0]
        keep.append(int(current))
        rest = order[1:]

        # only keep candidates that DON'T overlap too much with the box we just kept
        remaining = [idx for idx in rest if iou(boxes[current], boxes[idx]) <= iou_threshold]
        order = np.array(remaining, dtype=int)

    return keep

candidate_boxes = [
    [10, 10, 60, 60],       # 0: cluster A, best score
    [22, 22, 72, 72],       # 1: cluster A, partial overlap with 0
    [15, 12, 65, 62],       # 2: cluster A, heavy overlap with 0
    [200, 200, 250, 250],   # 3: cluster B, best score
    [205, 202, 253, 248],   # 4: cluster B, heavy overlap with 3
    [400, 50, 440, 90],     # 5: isolated, no overlap with anything
]
candidate_scores = [0.92, 0.85, 0.78, 0.95, 0.70, 0.60]

for threshold in [0.5, 0.3]:
    kept = nms(candidate_boxes, candidate_scores, iou_threshold=threshold)
    print(f"\nIoU threshold = {threshold}")
    print(f"kept indices: {kept}")
    for i in kept:
        print(f"  box {i}: {candidate_boxes[i]}  score={candidate_scores[i]}")
```

```
IoU threshold = 0.5
kept indices: [3, 0, 1, 5]
  box 3: [200, 200, 250, 250]  score=0.95
  box 0: [10, 10, 60, 60]  score=0.92
  box 1: [22, 22, 72, 72]  score=0.85
  box 5: [400, 50, 440, 90]  score=0.6

IoU threshold = 0.3
kept indices: [3, 0, 5]
  box 3: [200, 200, 250, 250]  score=0.95
  box 0: [10, 10, 60, 60]  score=0.92
  box 5: [400, 50, 440, 90]  score=0.6
```

Box 1's IoU with box 0 is `0.4061` — moderate overlap. At threshold `0.5`
that's not enough to suppress it, so both survive as (arguably) two
separate detections. At threshold `0.3`, `0.4061 > 0.3`, so box 1 gets
suppressed as a duplicate of box 0. Box 2's IoU with box 0 is `0.7606` —
heavy overlap — so it's suppressed at *both* thresholds, and boxes 3 and 4
(IoU `0.7847`) behave the same way in cluster B. Box 5 never overlaps with
anything and survives every threshold. This is exactly the tradeoff
described above, made concrete: lower the threshold and you suppress more
aggressively, at the risk of merging boxes that were genuinely two
different, nearby objects.

Running an actual pretrained detector is a different scale of computation —
downloading weights and running a full CNN forward pass — so it's shown
here as **illustrative rather than executed**:

```python
# ILLUSTRATIVE — requires `pip install ultralytics` and a model download
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # a small, pretrained YOLOv8 model
results = model("street_photo.jpg") # runs detection + NMS internally

for box in results[0].boxes:
    cls_name = model.names[int(box.cls)]
    confidence = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{cls_name}: {confidence:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

What to expect if you run it yourself: one printed line per detected
object, each with a class name, a confidence between 0 and 1, and a box —
the exact `(box, class, confidence)` triples described above, with NMS
already applied internally so you see one line per object, not ten.

## Build this

Run the NMS code above and change `iou_threshold` to a few more values —
try `0.1`, `0.4`, `0.6`, `0.9` — and note at each one which of the six boxes
survive. Find the threshold where box 1 flips from "kept" to "suppressed";
you should land right around `0.4061`, box 1's exact IoU with box 0.

**Stretch:** the intuition behind **mAP** (mean Average Precision), the
standard metric for comparing detectors, builds directly on IoU: for a
chosen IoU threshold (commonly `0.5`), a predicted box counts as a "true
positive" only if it matches an unmatched ground-truth box with IoU at or
above that threshold — a box with a decent-looking confidence score but
that only grazes the true object (IoU below threshold) counts as a false
positive, not a hit. Take the candidate boxes above, invent one "ground
truth" box for cluster A and one for cluster B, and by hand classify each
kept prediction (after NMS at threshold `0.5`) as a true or false positive
against your ground-truth boxes at an IoU threshold of `0.5`.

## Go deeper

- [Jonathan Hui: Object detection — mAP explained](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173) — a clear, worked walkthrough of precision, recall and mAP built on IoU.
- [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/) — docs for the library used in this lesson's illustrative example, including running detection on your own images in a few lines.
- [Redmon et al., "You Only Look Once"](https://arxiv.org/abs/1506.02640) — the 2015 paper introducing YOLO's single-pass design.
- [Meta AI: Segment Anything (SAM)](https://segment-anything.com/) — demo and paper for the prompt-based segmentation model mentioned above.
- [PyTorch: TorchVision Object Detection Finetuning Tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) — official tutorial fine-tuning a Faster R-CNN / Mask R-CNN model on a custom dataset.

**Next:** [Generative Adversarial Networks (GANs)](47-gans.md)
