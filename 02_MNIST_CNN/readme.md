# 02 — MNIST CNN
Convolution is a small window that slides.

## The one new idea
A fully connected layer flattens the image into one long vector, destroying spatial structure — shift a digit by two pixels and it looks like a different input. A **convolution** slides a small kernel across the image; the same weights visit every position, so:
- parameter count depends on kernel size and channel counts, **not** image size;
- local patterns (edges, corners) are found wherever they appear;
- deeper layers compose local patterns into global shapes.

```
Input (1×28×28)
  Conv 1→12, 3×3, pad=1 → ReLU → MaxPool 2×2   (12×14×14)
  Conv 12→24, 3×3, pad=1 → ReLU → MaxPool 2×2  (24×7×7)
  Flatten → FC 1176→70 → ReLU → Dropout(0.25) → FC 70→10
```

## Two side concepts
- **Max pooling** keeps the strongest activation in each 2×2 patch. It halves the spatial size, grows the receptive field, and adds a little translation invariance. Zero learnable parameters.
- **Dropout** randomly zeroes a fraction of activations during training only (`model.train()`), forcing the network not to rely on any single neuron. It is a cheap regularizer; automatically off in `eval()`.

As you go deeper, spatial size shrinks while channels grow: you trade *where* for *what*.

## What changed vs lesson 01
`model.py` is the only new thing — the loop now calls `train_one_epoch` / `evaluate` from `common/engine.py`.

## Run
```
python train.py
```
Expected (~seed 1337):
```
Epoch [1/10], Loss: 0.2410 ...
Epoch [10/10], Loss: 0.0237
Test Accuracy: 98.91%
```

## Checkpoint 2
- [ ] I can explain why a conv is parameter-efficient versus an FC layer.
- [ ] I know what max pooling does and why it has no parameters.
- [ ] I know when Dropout is active and why it exists.

→ Next: **03_Validation_and_Overfitting**. Is 98.9% good? How would we know when tuning tricks actually *help*? We need an honest evaluation protocol.
