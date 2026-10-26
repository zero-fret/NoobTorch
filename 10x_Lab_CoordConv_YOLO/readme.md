# 10x — Lab B: does CoordConv help detection? (optional)
Same controlled experiment as Lab A, but on the object detector.

## The experiment
`utils.py` and `dataset_generator.py` are byte-identical to lesson 10. The model is the lesson-10 network with one switch:

```bash
python train.py            # control: identical to lesson 10
python train.py --coords   # treatment: 2 coordinate channels at the input
```

The flag is the single switch; everything else (seed, epochs, LR, loss, data, thresholds) is fixed.

### Fair initialization (important)
Adding two input channels changes the shape of the first convolution, which would normally
shift the RNG initialization for every later layer and confound the comparison. `train.py`
therefore builds the control model first with the fixed seed, then builds the treatment,
copies the shared weights verbatim, and **zero-initializes only the two new coordinate
channels**. At step 0 the two networks are identical; the coordinate channels are the only
difference, so they are a clean single variable.

## Hypothesis
Detection predicts **positions**, so explicit coordinates could plausibly help more than they did for classification. But a convolutional detector is already built to localize, and the extra channels may add little or bias toward where objects usually appear. **Measure it.**

## Protocol
1. Run `python dataset_generator.py` once (shared data).
2. `python train.py` (control) → `python test.py`. Record AP.
3. `python train.py --coords` (treatment) → repeat. Weights are saved as `model_coords_False.pth` / `model_coords_True.pth`.
4. Repeat both at 2–3 seeds before trusting the difference.

| Run | coords | AP@0.5 | mAP@0.5:0.95 | params |
|---|---|---|---|---|
| control | False | 72.77% | 35.51% | 726,213 |
| CoordConv | True | 73.09% | 35.90% | 726,501 |

(One seed, RTX 4050. Coordinates give a small +0.3/+0.4 point edge here — directionally
consistent with the CoordConv paper that coordinates help coordinate-dependent tasks, but
the gap is within seed noise. Run 2–3 seeds before treating it as a real effect.)

(Hint: parameter count rises by only 288 in the stem — the extra cost is trivial.)

## Checkpoint
- [ ] I ran a controlled ablation where only one thing changed.
- [ ] I can state whether the trick helped **my** data and by how much.

→ Next: **11_Capstone** — leave the toy circles behind and solve a problem with data you collected.
