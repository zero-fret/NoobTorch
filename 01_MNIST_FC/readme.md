# 01 — MNIST FC (your first network that works)
Classify handwritten digits (MNIST) with a fully connected network.

## The one new idea
The **training loop** — the same five steps every model uses forever:
```
zero gradients → forward → compute loss → backward → optimizer step
```

## Model
```
Input (784) → FC1 (128) → ReLU → FC2 (64) → ReLU → FC3 (10) → 10 logits
```
- **FC / linear**: every input connects to every output.
- **Logits**: raw scores before softmax; `CrossEntropyLoss` includes softmax itself.
- Parameters: `784·128 + 128 + 128·64 + 64 + 64·10 + 10 = 109,194` (~0.44 MB float32).

## What changed vs lesson 00
Instead of differentiating a toy formula, we differentiate a loss over thousands of parameters and let the optimizer update them. `train.py` keeps the whole loop **visible** — read it once here; later lessons reuse `common/engine.py`.

## Run
```
python train.py
```
Expected (~seed 1337):
```
Epoch [1/10], Loss: 0.2659
...
Epoch [10/10], Loss: 0.0211
Test Accuracy: 97.70%
```

## Checkpoint 1
- [ ] I can name the five steps of the training loop.
- [ ] I know what `model.train()` and `model.eval()` switch.
- [ ] I know why the logits have no softmax.

→ Next: **02_MNIST_CNN**. A flattening FC layer throws away spatial structure and is parameter-heavy; a CNN fixes both.
