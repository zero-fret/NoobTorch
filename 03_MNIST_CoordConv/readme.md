# 03_MNIST_CoordConv
## What is Coordinate Convolution
CoordConv adds 2 coordinate channels (X and Y position) to the input, 
telling the model **where each pixel is located**.

### Effects of CoordConv:
Captures positional priors:
- Model learns that certain objects appear in specific locations
- Most datasets have natural position biases (sky at top, ground at bottom)  

Improves generalization: 
- Especially on datasets where position matters (CIFAR-10, ImageNet)  

Small parameter cost:   
- Only adds 2 input channels (~0.25% parameter increase)

Slower Convergence:
- The model needs extra time to learn position correlations 

Translation Invariance Deliberately Broken:
- Good: Model can use position as a clue
- Bad: Model may overfit to location


## Rsults
With seed `1337`, you will get:
```
Using device: cuda
Epoch [1/30], Loss: 0.2766, LR: 0.000997
Epoch [2/30], Loss: 0.0925, LR: 0.000989
Epoch [3/30], Loss: 0.0680, LR: 0.000976
Epoch [4/30], Loss: 0.0558, LR: 0.000957
Epoch [5/30], Loss: 0.0477, LR: 0.000933
Epoch [6/30], Loss: 0.0403, LR: 0.000905
Epoch [7/30], Loss: 0.0343, LR: 0.000872
Epoch [8/30], Loss: 0.0303, LR: 0.000835
Epoch [9/30], Loss: 0.0266, LR: 0.000794
Epoch [10/30], Loss: 0.0231, LR: 0.000750
Epoch [11/30], Loss: 0.0204, LR: 0.000703
Epoch [12/30], Loss: 0.0181, LR: 0.000655
Epoch [13/30], Loss: 0.0163, LR: 0.000604
Epoch [14/30], Loss: 0.0144, LR: 0.000552
Epoch [15/30], Loss: 0.0129, LR: 0.000500
Epoch [16/30], Loss: 0.0114, LR: 0.000448
Epoch [17/30], Loss: 0.0089, LR: 0.000396
Epoch [18/30], Loss: 0.0086, LR: 0.000345
Epoch [19/30], Loss: 0.0074, LR: 0.000297
Epoch [20/30], Loss: 0.0065, LR: 0.000250
Epoch [21/30], Loss: 0.0061, LR: 0.000206
Epoch [22/30], Loss: 0.0062, LR: 0.000165
Epoch [23/30], Loss: 0.0052, LR: 0.000128
Epoch [24/30], Loss: 0.0046, LR: 0.000095
Epoch [25/30], Loss: 0.0043, LR: 0.000067
Epoch [26/30], Loss: 0.0039, LR: 0.000043
Epoch [27/30], Loss: 0.0042, LR: 0.000024
Epoch [28/30], Loss: 0.0037, LR: 0.000011
Epoch [29/30], Loss: 0.0035, LR: 0.000003
Epoch [30/30], Loss: 0.0039, LR: 0.000000
Test Accuracy: 99.30%
```