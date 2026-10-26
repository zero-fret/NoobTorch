# 05 — CIFAR-10 + BatchNorm
Harder data (3-channel color photos, 10 classes) and a deeper network.

## The one new idea
As you stack layers, each one's input distribution keeps shifting as earlier weights change — this is *internal covariate shift*, and it makes deep nets slow and unstable.

**BatchNorm2d** normalizes each channel over the current mini-batch (to mean ~0, variance ~1) and then applies a learnable scale γ and shift β. Result:
- much faster training and higher usable learning rates;
- mild regularization (the batch statistics add noise);
- often no Dropout needed.

It needs a reasonable batch size (≥ 8) and behaves differently in `train()` vs `eval()` — exactly why `model.eval()` matters.

## The network
Three identical conv → conv → pool blocks (`32 → 64 → 128` channels, spatial `32 → 16 → 8 → 4`) then an FC head. BatchNorm is the only change in philosophy; convolutions are the same lesson-02 operation.

## Run the experiment
`train.py` exposes one flag:
```python
USE_BN = True   # set False -> BatchNorm becomes nn.Identity, same net
```
Run it with `True`, then `False`. With BatchNorm off, training starts slower and the deeper layers become finicky. This is the first time you can *feel* why BatchNorm made 100+ layer networks practical.

## Reading activation health
The script prints, per ReLU, the activation mean / std / fraction of dead (zero) units:
- std collapsing toward 0 → signal vanishing;
- dead fraction near 1 → dead neurons (see LeakyReLU in lesson 04);
- with BatchNorm these numbers stay well-scaled across all depths.

## What changed vs lesson 04
Same augmentation/optimizer knobs; the model is just deeper, with `BatchNorm` after each conv and a new diagnostic hook.

## Run
```
python train.py
```
Expected: high-80s % test accuracy with BN, slower/lower without.

## Checkpoint 5
- [ ] I can explain what BatchNorm normalizes and why it speeds training.
- [ ] I know why BatchNorm needs `train()`/`eval()` and a decent batch size.
- [ ] I can read an activation report and spot vanishing/dead activations.

→ Next: **06_CIFAR_ResNet18**. BatchNorm lets us stack blocks; ResNet makes the connections between blocks themselves learnable.
