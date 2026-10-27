# NoobTorch_1.0 — Training Results

Every lesson was actually trained end-to-end on an **NVIDIA RTX 4050 Laptop GPU**
(CUDA 12.4, torch 2.5.1, Python 3.12) — not smoke-tested. The numbers below are the
test-set / validation metrics printed by each lesson's own `test.py` / training script.
Detection lessons use real COCO photographs as backgrounds (downloaded separately —
see the README's "Detection backgrounds (COCO)" section).

## Classification track

| Lesson | Model | Task | Test accuracy |
|---|---|---|---|
| 01 | Plain fully-connected net | MNIST | **97.70%** |
| 02 | First CNN (conv + pool) | MNIST | **98.91%** |
| 03 | Overfitting demo (2,000 train samples) | MNIST | train 100% / val 92.01% / **test 92.56%** |
| 04 | All training knobs applied | MNIST | **99.30%** |
| 05 | CNN with BatchNorm | CIFAR-10 | **88.15%** |
| 06 | ResNet-18 from scratch (25 epochs) | CIFAR-10 | **92.78%** |
| 07a | ResNet-18, frozen ImageNet head (3 epochs) | CIFAR-10 | **80.01%** |
| 07b | ResNet-18, fine-tuned upper stages (12 epochs) | CIFAR-10 | **95.77%** |

Lesson 03 deliberately trains on a small subset so the student can watch the
train/val gap widen (`results.png` is the saved learning curve). Top confusions are
4→9, 9→4, 8→5, 9→7 — the classic visually-similar MNIST pairs.

Lesson 07 shows the two stages of transfer learning: a frozen head gets 80% in
minutes, then `finetune.py` unfreezes layer3/layer4 with a small backbone LR and
reaches **95.77%** — clearly above training from scratch (92.78%).

## Lab A — CoordConv on classification (07x)

| Run | coords | Test accuracy |
|---|---|---|
| control | False | **99.23%** |
| CoordConv | True | **99.17%** |

Coordinates do not help ordinary MNIST classification — the network doesn't need
explicit position to label digits. This matches the CoordConv paper: the trick matters
only for coordinate-dependent tasks.

## Detection track

| Lesson | Model | Objects / image | AP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 09 | SimpleYOLO (single object) | 1 | **98.65%** | **88.65%** |
| 10 | SimpleYOLO + depthwise-separable conv | 4 | **72.77%** | **35.51%** |

Lesson 09 trains 120 epochs on 12,800 images; lesson 10 is the deliberately harder
multi-object setting (4 circles per image, direct width/height regression), 60 epochs
on 3,200 images. The lower AP is by design — more objects and the harder encoding
make localization and duplicate suppression harder — and motivates the later lessons.

## Lab B — CoordConv on detection (10x)

| Run | coords | AP@0.5 | mAP@0.5:0.95 | params |
|---|---|---|---|---|
| control | False | **72.77%** | **35.51%** | 726,213 |
| CoordConv | True | **73.09%** | **35.90%** | 726,501 |

The control and treatment start from identical shared weights — only the two new
coordinate channels differ (zero-initialized), so the comparison isolates that single
variable. Coordinates give a small edge on this coordinate-dependent task
(+0.3 / +0.4 points); the gap is within single-seed noise, so run 2–3 seeds before
treating it as a real effect.

## Environment

- GPU: NVIDIA RTX 4050 Laptop, 6 GB (WDDM), CUDA 12.4 build
- Interpreter: Python 3.12, torch 2.5.1+cu124, torchvision 0.20.1+cu124,
  numpy 2.2.x, opencv 4.12
- To reproduce: double-click the lesson's `train.py` (or `train_coords.py` for the
  CoordConv treatment), then `test.py`.
