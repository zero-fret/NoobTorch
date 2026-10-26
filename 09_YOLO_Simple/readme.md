# 09 — Simple YOLO
You learned boxes, IoU, NMS and AP in lesson 08. Now train a network that emits them.

> **Before you run:** this lesson (and 10/10x) needs real COCO photos as backgrounds —
> see the **"Detection backgrounds (COCO)"** section in the root `README.md`, then run
> `python dataset_generator.py` here first.

## The one new idea
The **detection loss** — training a grid to output boxes and objectness. The 640×640 image is downsampled to a 20×20 feature map. Each cell predicts `(tx, ty, tw, th, conf)`; the cell containing the circle's center is the single positive cell.

The loss has two parts:
- **Coordinate loss** (MSE): how wrong the box is. Applied only to the positive cell(s).
- **Objectness loss** (weighted BCE): does this cell contain an object? With ~1 positive out of 400 cells, positive cells are weighted 3.0 and negative cells 0.5 so the vast empty background doesn't dominate.

Box encoding matches lesson 08:
- `tx, ty`: offset from the cell **center**, decoded `sigmoid(raw)·2 − 0.5`, then `cx = (gx + 0.5 + tx)/S·size`;
- `tw, th`: `sigmoid(raw)` × image size.

## The model
Six stride-2 conv blocks (640→320→160→80→40→20) plus a 1×1 detection head. Convolution + BatchNorm + LeakyReLU throughout. Total ~1.58 M parameters.

## What changed vs lesson 08
No new math — the same IoU/NMS/AP code in `utils.py` is used verbatim. The only new thing is `forward_loss` (MSE + weighted BCE) and the training loop.

## Run
```
python dataset_generator.py   # 12800 train + 3200 val synthetic circles
python train.py               # ~120 epochs, AP every 10
python test.py                # draws boxes into test_result/
```

## Checkpoint 9
- [ ] I can write the two loss components and say which cells each applies to.
- [ ] I can explain why negative cells are down-weighted.
- [ ] I can decode the raw `(5,20,20)` output into pixel boxes.

→ Next: **10_YOLO_DSC**. Same detector, but what does it take to make it smaller and faster on the same task?
