# 04 — Training knobs
Lesson 03 gave us the ruler (val accuracy). Now learn the four levers that make a model generalize and converge faster.

## The one new idea
**One knob at a time, measure on val.** The knobs live as flags at the top of `train.py`:

```python
USE_AUGMENTATION = True
USE_SCHEDULER = True
USE_WEIGHT_DECAY = True
ACTIVATION = 'leaky_relu'
```

## The knobs

### 1. Data augmentation — "free data"
`RandomAffine` rotates/shifts/scales each image a little, every epoch. The model sees the same digit in new poses and must learn *invariant* features. Augmented runs often reach higher **test** accuracy even though the training loss converges more slowly.

### 2. Learning-rate schedule — Cosine Annealing
A fixed learning rate is a compromise: large steps early help escape bad minima; small steps late let you settle in. `CosineAnnealingLR` smoothly decays the LR from `LR` to ~0 on a cosine curve over `T_max` epochs. It doesn't raise the ceiling; it reaches it in fewer epochs.

### 3. Weight decay — an L2 penalty
`weight_decay` adds `λ·Σw²` to the loss, gently pulling every weight toward zero. Large weights let a model memorize noise; small weights keep the decision boundary smooth. This is the second classic regularizer alongside Dropout/augmentation.

### 4. Activation — LeakyReLU vs ReLU
A ReLU unit fed a strong negative gradient can become permanently zero ("dead neuron"). `LeakyReLU(0.01)` keeps a tiny slope for negatives, so gradients keep flowing. Cost: negligible.

### Bonus: the DataLoader is not free
`num_workers=N` loads/augments the next N batches in subprocesses while the GPU trains, and `pin_memory=True` speeds CPU→GPU transfer. On an easy task (MNIST) the loader can be the bottleneck; set it to roughly the number of CPU cores.

## What changed vs lesson 03
Nothing new in the loop — only `train.py` configuration. The model is the lesson-02 CNN with a swappable activation.

## Run & experiment
```
python train.py
```
Run it four times with one flag flipped each time; record the val/test accuracy in a table:

| augmentation | scheduler | weight decay | activation | val acc | test acc |
|---|---|---|---|---|---|
| off | off | off | relu | | |
| on | off | off | relu | | |
| on | on | off | relu | | |
| on | on | on | relu | | |
| on | on | on | leaky_relu | | |

## Checkpoint 4
- [ ] I know what each of the four knobs does and which problem it targets.
- [ ] I change one knob per run and judge on val.
- [ ] I understand what weight decay penalizes.

→ Next: **05_CIFAR_BatchNorm**. MNIST is done; now harder data, 3-channel color, and a deeper network that needs a new tool to train.
