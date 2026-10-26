# 10 — Efficient YOLO (Depthwise Separable Convolution)
Same task as lesson 09, now with multiple objects and a smaller/faster network.

## The one new idea
A standard 3×3 conv with 32→64 channels costs `3×3×32×64 = 18,432` weights. **Depthwise separable convolution** splits it:
- **Depthwise** — one 3×3 filter per channel: `3×3×32 = 288` weights;
- **Pointwise** — a 1×1 conv that mixes channels: `32×64 = 2,048` weights.

Total `2,336` vs `18,432` — roughly **87% fewer** weights, at almost no accuracy loss on a simple task. This is the building block behind MobileNet and modern YOLO variants.

## What else is different
- **Four circles per image** (one per quadrant), so the dataset and loss now handle multiple objects — one positive cell each.
- Objectness imbalance is handled with `BCEWithLogitsLoss(pos_weight=num_neg/num_pos)` instead of fixed hand-picked weights — the same idea as focal loss, but simpler.
- Width/height are regressed directly (no sigmoid); the decoder clamps negatives to zero.

## What changed vs lesson 09
The only architectural change is the conv block (`DepthwiseSeparableConv`). Encoding, NMS, AP, the data pipeline are reused.

## Run
```
python dataset_generator.py
python train.py
python test.py
```

## Benchmark it (the point of the lesson)
Compare the two detectors on the same data:
```
python -c "from model import SimpleYOLO_DSConv as M; print(sum(p.numel() for p in M().parameters()))"
```
Then do the same in lesson 09 for `SimpleYOLO`, and time one forward pass on a `(1,3,640,640)` tensor:
```python
import time, torch
x = torch.randn(1, 3, 640, 640)
t = time.time(); model(x); print(time.time() - t)
```
Record params vs AP for both — efficiency is always a tradeoff, never a free lunch.

## Checkpoint 10
- [ ] I can compute the parameter count of a depthwise and a pointwise conv.
- [ ] I can explain why the cost drops ~87%.
- [ ] I can compare two models on params, speed, and accuracy.

→ Next: **10x_Lab_CoordConv_YOLO** (optional). We proved the ablation method on classification; now run the same experiment on the detector.
