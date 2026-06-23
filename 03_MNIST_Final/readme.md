# 03_MNIST_Final
This time we will **finish** the MNIST with 5 ~~noob~~ simple but useful improvements.
- CoordConv + CosineAnnealing + LeakyReLu + Dataloader + Augmentation
- 51k parameters, Test Accuracy: 99.45%

## CoordConv (Coordinate Convolution)
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

## Scheduler
Instead of using a fixed `0.001`, dynamically adjusts the learning rate for fewer epochs and better performance. In this lesson we use **Cosine Annealing**, the *no-brain* scheduler that will almost always work.

### Cosine Annealing 
Smoothly decays the learning rate following a cosine curve—starting high for exploration, ending near zero for fine-tuning.  
The formula: `lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))`  
Looks like: `y = cos(x) + 1, 0 < x < Pi`  
This lets gradients take large steps early to escape bad minima, then tiny steps late to settle into the optimum. One line in PyTorch: `scheduler = CosineAnnealingLR(optimizer, T_max=epochs)`. It doesn't improve accuracy ceilings. It reaches them in fewer epochs.  
*The noobest scheduler ever. Set epochs = 100 and go to sleep. It will handle the rest.*

## LeakyReLU
LeakyReLU is a variation of ReLU that allows a small, non-zero gradient when the input is negative, 
preventing "dead neurons" that permanently output zero.  
Formula: `f(x) = max(αx, x)` where α is the negative slope (default 0.01)  
PyTorch implementation: `torch.nn.LeakyReLU(negative_slope=0.01)`  
### Effects of LeakyReLU:
Prevents dying ReLU problem:
- Negative values get a small slope (typically 0.01) instead of zero
- Ensures gradients can still flow through inactive neurons

Maintains sparsity benefits:
- Still zero-centers the output distribution
- Retains most of ReLU's computational efficiency

Small computational overhead:
- Almost identical to ReLU in speed
- No significant parameter increase

Slightly better performance:
- Often improves accuracy by 1-2% over standard ReLU
- More stable training on deeper networks

Trade-offs:
- Good: Prevents dead neurons, more robust training
- Bad: Slightly slower convergence than ReLU due to extra operations
- The α parameter is an additional hyperparameter to tune  

*Set α = 0.01. It's almost always better than ReLu.*

## Dataloader
Sometimes the Dataloader running on the GPU becomes the bottleneck because task is too easy for GPU.  
Parallelizes data loading with `num_workers=4` speeds it up.
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4, # Uses 4 subprocesses to load data in parallel (speeds up I/O)
    pin_memory=True
)
```

## Data Augmentation
Artificially expands the training dataset by applying random transformations (like rotation, shifting, or scaling) to existing images, forcing the model to learn invariant features and generalize better to unseen data. The augmented model often achieves **higher test accuracy** despite slower training loss convergence.  
*Thus creating dataset for free. There is no reason not doing this.*
```python
    transforms.RandomAffine(
        degrees=10,            # random rotation between -10° and +10°
        translate=(0.1, 0.1),  # random horizontal/vertical shift up to 10% of image size
        scale=(0.9, 1.1)       # random scaling between 90% and 110%
    ),
```


## Rsults
With seed `1337`, you will get:
```
Test Accuracy: 99.45%
```

*Say goodbye to this hello world dataset. CIFAR-10 and ResNet are just ahead.*