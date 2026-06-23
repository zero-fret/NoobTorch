# 04_CIFAR_ResNet18
## CIFAR-10
CIFAR-10 is a benchmark dataset of 60,000 32x32 color images across 10 classes (e.g., airplanes, cats, trucks), with 6,000 images per class—split into 50,000 training and 10,000 test images (5,000 train / 1,000 test per class). It is significantly more challenging than MNIST.

## ResNet
In deep learning, the **vanishing gradient problem** occurs when gradients become exponentially small during backpropagation through many layers, causing early layers to stop learning.  
**ResNet (Residual Network)** solves this by introducing skip connections (residual connections). These connections add the input of a layer directly to its output a few layers later, providing a clean, unweighted shortcut for gradients to flow backward. This not only mitigates vanishing gradients but also addresses the degradation problem by allowing the network to easily learn identity mappings, ensuring that adding more layers does not hurt training accuracy and enabling the construction of networks with hundreds of layers.  
In short, ResNet is a **smart bypass shortcut** between layers.

## Data Augmentation
More complex data augmentation is used in the CIFAR-10 task.
```python
train_transform = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(p=0.5),         
    transforms.RandomCrop(32, padding=4),             
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.5,                   
                             scale=(0.02, 0.15),
                             ratio=(0.3, 3.3))
])
```
- RandAugment: Searches over a pool of 14 augmentation operations (e.g., rotate, shear, color jitter) and applies 2 random ones per image at magnitude 9
- RandomHorizontalFlip: Standard mirror augmentation to double effective dataset size
- RandomCrop: Padding + random cropping prevents overfitting to border pixels and adds translation invariance
- ToTensor: Mandatory conversion for PyTorch model input
- Normalize: Scales pixel values to standard normal distribution for stable gradient flow during training
- RandomErasing: Also called "Cutout" — randomly masks out small rectangular regions to force the model to use broader context, reducing over-reliance on specific features