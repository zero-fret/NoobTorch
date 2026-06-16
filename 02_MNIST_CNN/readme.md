# 02_MNIST_CNN
## What is CNN (Convolution Neural Network)
A fully connected layer flattens an image into a long vector, losing all spatial structure. An object shifted two pixels becomes a completely different input. Plus, connecting every pixel to every neuron explodes parameters.

A CNN **slides a small window (kernel)** across the image. As the kernel slides, each position produces one value, together forming a **feature map**, which is a 2D map of where each pattern was detected. The same weights scan every position, so parameters stay tiny. More importantly, the kernel detects local patterns (edges, corners) regardless of where they appear. CNN layers then hierarchically compose these local patterns into global shapes.

## CNN: Parameters & Data Flow
General formula: out_channels × (in_channels × kernel_h × kernel_w + 1). Parameter count depends **only on kernel size and channel counts**, not on image size. That's why a CNN trained on 32×32 images can process 640×640 images with zero changes.
Bias: Learnable constant. Kernal can output non-0 number even when input=0. It sits in "+1" in the formula.
Example: Conv2d(1→32, 3×3) = 32 × (1 × 3 × 3) + 32 = 320 parameters.
         Conv2d(32→64, 3×3) = 64 × (32 × 3 × 3) + 64 = 18,496 parameters.
         Linear(14×14×64 → 10) = 12,544 × 10 + 10 = 125,450 parameters.

## Max pooling: 
Max pooling slides a 2×2 window across each feature map, keeping only the strongest activation in each patch. This cuts spatial size in half—from 28×28 to 14×14 in one step. Why it works: it reduces computation, enlarges the receptive field for deeper layers, and adds translation invariance—a pattern shifted by a few pixels still produces the same pooled output. It has zero learnable parameters. For classification, global average pooling goes further: averaging each entire channel into a single number, forcing the network to recognize content without memorizing positions.

As layers go deeper, spatial size shrinks (28→14→7→...) while channel count grows (1→32→64→...). You trade "where" information for "what" information. Early layers know precise locations of simple edges. Deep layers know vague locations of complex concepts.

## Cosine Annealing 
Smoothly decays the learning rate following a cosine curve—starting high for exploration, ending near zero for fine-tuning.
The formula: `lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))`
Looks like: `y = cos(x) + 1, 0 < x < Pi`
This lets gradients take large steps early to escape bad minima, then tiny steps late to settle into the optimum. One line in PyTorch: scheduler = CosineAnnealingLR(optimizer, T_max=epochs). It doesn't improve accuracy ceilings. It reaches them in fewer epochs.

## Model Definition
```
Input (1×28×28)
    ↓
Conv1: 1→12 channels, 3×3 kernel, padding=1  (保持28×28)
    ↓ ReLU
MaxPool: 2×2, stride=2  (12×14×14)
    ↓
Conv2: 12→24 channels, 3×3 kernel, padding=1  (保持14×14)
    ↓ ReLU
MaxPool: 2×2, stride=2  (24×7×7)
    ↓ Flatten: 24×7×7 = 1176
    ↓
FC1: 1176 → 70
    ↓ ReLU + Dropout(0.25)
    ↓
FC2: 70 → 10
    ↓
Output (10 logits)
```

## Rsults
With seed `1337`, you will get:
```
Using device: cuda
Epoch [1/20], Loss: 0.2410, LR: 0.000994
Epoch [2/20], Loss: 0.0817, LR: 0.000976
Epoch [3/20], Loss: 0.0600, LR: 0.000946
Epoch [4/20], Loss: 0.0499, LR: 0.000905
Epoch [5/20], Loss: 0.0415, LR: 0.000854
Epoch [6/20], Loss: 0.0359, LR: 0.000794
Epoch [7/20], Loss: 0.0297, LR: 0.000727
Epoch [8/20], Loss: 0.0258, LR: 0.000655
Epoch [9/20], Loss: 0.0215, LR: 0.000578
Epoch [10/20], Loss: 0.0191, LR: 0.000500
Epoch [11/20], Loss: 0.0156, LR: 0.000422
Epoch [12/20], Loss: 0.0132, LR: 0.000345
Epoch [13/20], Loss: 0.0121, LR: 0.000273
Epoch [14/20], Loss: 0.0096, LR: 0.000206
Epoch [15/20], Loss: 0.0087, LR: 0.000146
Epoch [16/20], Loss: 0.0077, LR: 0.000095
Epoch [17/20], Loss: 0.0068, LR: 0.000054
Epoch [18/20], Loss: 0.0060, LR: 0.000024
Epoch [19/20], Loss: 0.0059, LR: 0.000006
Epoch [20/20], Loss: 0.0058, LR: 0.000000
Test Accuracy: 99.28%
```