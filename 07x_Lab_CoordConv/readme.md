# 07x — Lab A: does CoordConv actually help? (optional)
A lesson is no longer "trust the tutorial". It's a controlled experiment.

## The idea
**CoordConv** concatenates two extra input channels telling every pixel its X/Y position (normalized to [-1, 1]). This hands the network explicit position priors — useful when position matters (e.g. objects always appear at a fixed location).

But convolutions are deliberately *translation invariant*: the same digit in a shifted spot should stay the same digit. Adding coordinates breaks that invariance. So on MNIST, CoordConv is plausibly useless or even harmful. **The data decides, not the paper.**

## Experimental setup
Change ONE variable per run, keep all the rest identical (seed, augmentation, scheduler, epochs, architecture):

| Run | `USE_COORDS` | `ACTIVATION` | val acc | test acc |
|---|---|---|---|---|
| 1 (control) | False | relu | | |
| 2 | False | leaky_relu | | |
| 3 (CoordConv) | True | relu | | |
| 4 (CoordConv) | True | leaky_relu | | |

## What to look for
- **Within noise (±0.1%)**: no effect on this dataset — a valid and common result.
- **Bigger train/val gap with coords**: it overfits to position, as predicted.
- **Repeat with `seed` 1, 2, 3** before believing a difference — a single lucky run proves nothing.

## What this teaches
The single most valuable skill after "make it run": **state a hypothesis → change one thing → measure → repeat**. A trick that helps one dataset can hurt another.

## Run
```
python train.py
```
(edit the two flags at the top between runs)

## Checkpoint
- [ ] I can design an ablation with one changed variable and a control.
- [ ] I can distinguish a real effect from seed noise.

→ Next: **08_Detection_Primer**. Classification answers "what?". Detection must also answer "where?" — and needs new tools before any training.
