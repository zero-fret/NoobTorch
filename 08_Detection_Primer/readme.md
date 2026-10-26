# 08 — Detection primer (no training)
Classification outputs a class label. Detection must also say **where** every object is. Before building YOLO, learn its output vocabulary by hand.

## The one new idea
A detector replaces "10 class scores" with a list of **bounding boxes**, each with a confidence and (optionally) a class. Everything else this lesson is about exists just to produce, decode, and judge those boxes.

## 1. Box formats
- **YOLO label**: `(cx, cy, w, h)` normalized to [0,1] — fractions of the image.
- **Pixel `xyxy`**: `(x1, y1, x2, y2)`, top-left and bottom-right corners — used for drawing and IoU.
`boxes.py` converts between them.

## 2. IoU (Intersection over Union)
`IoU = area(intersection) / area(union)` of two boxes. 1.0 = identical, 0 = no overlap. It is the single number that judges "is this box correct?".

## 3. The grid
Simple YOLO splits the image into an `S × S` grid. The cell containing an object's center is responsible for it:
- `(gx, gy)` = the owning cell;
- `(tx, ty)` = the center's offset inside that cell;
- `(tw, th)` = box size;
- `conf` = objectness (is there an object in this cell at all?).
The network outputs `(5, S, S)` — one box per cell. `demo.py` shows how a label becomes a cell and an offset.

## 4. Objectness and sigmoid
A raw network logit can be any number; `sigmoid` maps it to (0,1). The confidence is a sigmoid; box offsets/sizes are also bounded with sigmoid so they stay in legal ranges.

## 5. Confidence threshold
Keep only boxes with `conf > threshold` (e.g. 0.5). A low threshold finds more objects but admits false positives — the precision/recall tradeoff from lesson 03, now in detection.

## 6. NMS (Non-Maximum Suppression)
Neighboring cells fire on the same object, producing duplicate boxes. NMS: keep the highest-scoring box, remove every lower-scoring box overlapping it above an IoU threshold, repeat.

## 7. AP / mAP
Sort predictions by confidence, match each to a ground-truth box (IoU ≥ threshold), and build a precision-recall curve. Its area is **AP**; averaging over IoU thresholds 0.5–0.95 gives **mAP**. `eval.py` implements it.

## Run
```
python boxes.py     # box / IoU / grid arithmetic
python demo.py      # full tour: threshold -> NMS -> AP -> picture
```

## Checkpoint 8
- [ ] I can convert a normalized YOLO label to pixel corners.
- [ ] I can compute IoU and explain NMS.
- [ ] I can describe the five channels a Simple YOLO cell predicts.
- [ ] I know what AP measures and how precision/recall trade off.

→ Next: **09_YOLO_Simple**. You already know the output; the only new thing is training a network to produce it.
