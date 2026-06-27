# 06_YOLO_DSC (Depthwise Separable Convolution) and Focal Loss
This lesson demonstrates the application of **Depthwise Separable Convolution** (DSC) in a minimal YOLO-style object detection model. DSC is a key efficiency technique used in modern YOLO variants (like YOLOv5, YOLOv8) to reduce model size and computational cost while maintaining detection performance.

## What is Depthwise Separable Convolution
Depthwise Separable Convolution is a key **efficiency technique** used in modern YOLO variants (like YOLOv5, YOLOv8) to **reduce model size and computational cost** while maintaining detection performance.  
It factorizes a standard convolution into two separate operations:  
1. Depthwise Convolution (Split)
- Applies a **single filter per input channel**
- Preserves spatial dimensions but does NOT combine channels
- Parameters: `kernel_size² × in_channels`
- Example: For 3×3 kernel, 32 input channels → 3×3×32 = 288 parameters
2. Pointwise Convolution (1×1 Convolution Merge)
- Combines channels using 1×1 filters
- Projects depthwise output to desired output channels
- Parameters: `in_channels × out_channels`
- Example: 32→64 channels → 32×64 = 2,048 parameters  

### Comparison: Standard vs Depthwise Separable
Standard Conv: 3×3×32×64 = 18,432 parameters  
Depthwise Separable: 3×3×32 + 32×64 = 2,336 parameters  
Reduction: 87.3%  

## Focal Loss
Focal Loss modifies standard cross-entropy by adding a modulating factor (1-p_t)^γ that **down-weights easy examples** (where the model is confident and correct), forcing the model to **focus on hard examples** (where the model is uncertain or misclassified).  
For tasks with **severe class imbalance** (e.g., 1 positive to 100+ negatives in object detection), focal loss automatically prioritizes difficult samples, preventing the vast number of easy examples from overwhelming the gradient and allowing the model to **learn the rare class** effectively. It is much more efficient than hardcoding positive vs negative weights in loss because it automatically *focuses* on hard samples.  
Focal loss has 2 parameters:
- Gamma (γ) controls how aggressively easy examples are down-weighted.
- Alpha (α) balances the importance of positive vs. negative classes.
In SimpleYOLO_DSC, the ratio is severe (1~4 positive grids to 100 negative grids). Therefore focal loss is included.  

*In short: Split and merge to be lighter and faster*

## File Structure
```
06_YOLO_DSC/
├── model.py # SimpleYOLO with Depthwise Separable Convolutions
├── train.py # Training script (same as Lesson 05)
├── test.py # Testing and evaluation script
├── utils.py # Dataset, decoding, NMS, AP computation
├── dataset_generator.py # Synthetic circle dataset generator
├── background/ # Background images folder (create this)
├── data/ # Generated dataset
│ ├── train/
│ │ ├── images/ # Training images (3200)
│ │ └── labels/ # Training labels (YOLO format)
│ └── val/
│ ├── images/ # Validation images (800)
│ └── labels/ # Validation labels (YOLO format)
├── model.pth # Saved model weights
└── test_result/ # Test output visualizations
```
## Quick Start
```
copy backgrounds folder from `05_YOLO_Simple/` to `06_YOLO_DSC/`
run dataset_generator.py
run train.py
run test.py
```
