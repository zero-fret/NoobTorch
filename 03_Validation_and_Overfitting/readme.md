# 03 — Is it actually learning? (validation, overfitting, metrics)
Before trusting any accuracy number, you need an honest evaluation protocol.

## The one new idea
Split data into three disjoint sets:

| Set | Used for | How often touched |
|---|---|---|
| **train** | update parameters | every epoch |
| **val** | choose hyperparameters / spot overfitting | every epoch, never trained on |
| **test** | the single final number | once, at the end |

Training on your evaluation set is an **open-book exam**: a model that memorizes the answers scores 100% and learns nothing.

## Overfitting in three curves
Train loss and val loss plotted against epochs tell you what state you're in:

- **Underfitting**: both train and val loss are high and still falling → train longer or use a bigger model.
- **Healthy**: both fall, then flatten close together.
- **Overfitting**: train loss keeps falling (→ ~0), but val loss stops improving and turns **up**. The network is memorizing training images.

This script deliberately makes overfitting easy: a big `OverfitNet` on only `TRAIN_SUBSET = 2000` images. Open `results.png` and watch the lines diverge.

## Beyond accuracy
A single accuracy hides *which* classes fail.
- **Confusion matrix** `cm[true, predicted]`: diagonal = correct; off-diagonal = mistakes.
- **Precision** = of what I predicted, how much was right (`TP/(TP+FP)`).
- **Recall** = of what exists, how much I found (`TP/(TP+FN)`).
- **F1** = harmonic mean of the two.

## How to tune (the protocol)
1. change one knob (epochs, LR, width, augmentation, …);
2. train, compare the **val** metric;
3. keep the winner;
4. only after tuning, report the **test** result once.

## Run
```
python train.py
```
You'll see a per-epoch table, a saved `results.png`, the test accuracy, macro P/R/F1, and the most-confused digit pairs.

## Exercises (do these — they are the lesson)
- Set `TRAIN_SUBSET = 2000`, then `50000`; compare the gap between train and val.
- Set `EPOCHS = 15` then `60`; find the epoch where val loss turns upward.
- Halve the model width (`OverfitNet(hidden=32)`); is it still overfitting?
- Keep one notebook of changes and val results — that is how tuning actually works.

## Checkpoint 3
- [ ] I can explain why we hold out a val set and never tune on test.
- [ ] I can read loss curves and name the three training states.
- [ ] I can find a model's most confused classes with a confusion matrix.

→ Next: **04_Training_Knobs** — now that we can measure honestly, let's learn the four levers that fix overfitting and speed up convergence.
