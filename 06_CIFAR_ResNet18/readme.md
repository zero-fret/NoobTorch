# 06 — CIFAR ResNet18
Residual connections let a network become deep without degradation.

## The one new idea
Deep plain networks can get *worse* on training as layers are added — not just overfitting, but an optimization failure (vanishing gradients, degradation). A **residual block** adds the input directly to the output of its layers:

```
out = ReLU( W2·ReLU(W1·x) + x )
```
The block only has to learn the *difference* (residual). If the extra layers are useless, they learn zero and the block behaves as identity, so depth can't hurt — and gradient flows to earlier layers through the straight `+ x` shortcut. This is the "smart bypass" that made networks with hundreds of layers trainable.

## The model
- CIFAR stem: one 3×3 stride-1 conv (32×32 images are too small for ImageNet's 7×7 stride-2 stem).
- Four stages of 2 blocks each, channels `64 → 128 → 256 → 512`, the first block of each stage strides by 2.
- Global average pooling → 10-way linear.
- A `1×1` conv in the shortcut whenever dimensions change.

## Training recipe (the standard CIFAR one)
- **SGD + momentum 0.9**, `lr=0.1` (momentum and BatchNorm make a large LR safe and fast),
- `weight_decay=5e-4`,
- Cosine annealing,
- **strong augmentation**: RandAugment (2 random ops) + horizontal flip + padded crop + RandomErasing.

## What changed vs lesson 05
A new block structure (skip connections); everything else — BatchNorm, augmentation, scheduler, the fit loop — is reused.

## Run
```
python model.py     # prints output shape and parameter count
python train.py
```
Expected: ~90–93% test accuracy after 25 epochs (more epochs climb higher).

## Checkpoint 6
- [ ] I can draw a residual block and write the identity equation.
- [ ] I know why the shortcut needs a 1×1 conv when channels change.
- [ ] I know why a deep plain network can fail but a ResNet can't get worse.

→ Next: **07_Transfer_Learning**. Training every network from scratch is rare in practice — most real work reuses a pretrained backbone.
