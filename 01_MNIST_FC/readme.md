# 01_MNIST_FC
Train a fully connected network that classifies written numbers (the MNIST dataset).
*The Hello World of Machine Learning*

## Requirements
`pip install torch torchvision numpy`

## Model Definition
`Input (784) → FC1 (128) → ReLU → FC2 (64) → ReLU → FC3 (10) → Output`
- Fully connected: Every node in layer `n` is connected to every  node in layer `n+1`
- ReLU: `y = max(0, x)`

## Model Size
```
Total parameters: 784×128 + 128 + 128×64 + 64 + 64×10 + 10 = 109,194
Size: 1.25 MB (float32)
```

## Rsults
With seed `1337`, you will get:
```
Using device: cuda
Epoch [1/10], Loss: 0.2659
Epoch [2/10], Loss: 0.1097
Epoch [3/10], Loss: 0.0765
Epoch [4/10], Loss: 0.0596
Epoch [5/10], Loss: 0.0488
Epoch [6/10], Loss: 0.0398
Epoch [7/10], Loss: 0.0335
Epoch [8/10], Loss: 0.0283
Epoch [9/10], Loss: 0.0254
Epoch [10/10], Loss: 0.0211
Test Accuracy: 97.70%
```

## Next Step: CNN
This FC network ignores pixel spatial relationships and is parameter-inefficient. 
A CNN can achieve 99.2%+ with fewer parameters.
→ Go to **02_MNIST_CNN**