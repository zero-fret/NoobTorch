# NoobTorch 1.0
**PyTorch Walkthrough** for **Noobs**, Ultralytics Tuners and Everyone who thinks **AI** is cool.  
- Simple
- Minimum
- *extremely noob-friendly*

This course welcomes:
- Noob
- Trend follower
- Parameter tuner
- Anyone who has no access to professional ML courses for some reason

Knowledge required: 
- Math: High school + `∂x/∂y`
- Programming: Can read Python
- But a `Data Structures & Algorithms` is favored

Unlike traditional PyTorch courses, NoobTorch provides simple **click-to-run** code and **intuitive** description to help beginners start developing deep learning models with PyTorch as quickly as possible.  
*At least you will no longer be afraid of it*  
*And papers will no longer look like random chars*

## Setup (once)
```bash
pip install -r requirements.txt
```
Run each script from inside its lesson folder (`python train.py`). MNIST/CIFAR download once into a shared `data/` folder.

## Detection backgrounds (COCO)
Lessons 09, 10 and 10x draw synthetic circles on top of real COCO photos so the
network learns to ignore clutter. The images are **not bundled** in this repo —
download them yourself (CC BY 4.0; credit the COCO consortium).

1. Grab any COCO image split (raw `.jpg` files only, **no** annotation JSONs):
   - **val2017.zip** — ~1 GB, 5,000 images (recommended)
   - test2017.zip — ~6.6 GB, 40k images
   - train2017.zip — ~19 GB, 118k images

   Download page: https://cocodataset.org/#download
   Direct link:    http://images.cocodataset.org/zips/val2017.zip

2. Unzip and copy the `.jpg` files **flat** into `assets/backgrounds/` (do not
   leave them nested under a `val2017/` folder). The generator also accepts a
   lesson-local `background/` folder.

3. Generate the detection dataset for each lesson:
   ```bash
   cd 09_YOLO_Simple && python dataset_generator.py
   cd ../10_YOLO_DSC && python dataset_generator.py
   cd ../10x_Lab_CoordConv_YOLO && python dataset_generator.py
   ```

## From `n00b` to `1337`

| Lesson | Topic | What you will learn |
|--------|-------|---------------------|
| 00 | Hello PyTorch | Noob's kit: tensors, `autograd`, `.backward()`, save/load |
| 01 | MNIST FC | Fully-connected net, your 1st model that trains |
| 02 | MNIST CNN | Convolution is a window that slides; pooling; dropout |
| 03 | Validation & Overfitting | train/val/test split, learning curves, precision/recall |
| 04 | Training Knobs | augmentation, cosine LR, weight decay — one knob at a time |
| 05 | CIFAR + BatchNorm | 3-channel data, going deeper, BatchNorm |
| 06 | CIFAR ResNet18 | Residual connections are a smart shortcut |
| 07 | Transfer Learning | pretrained backbone, fine-tuning, inference on your images |
| 07x | Lab: CoordConv | run a controlled experiment, measure, decide *(optional)* |
| 08 | Detection Primer | boxes, IoU, grid cells, objectness, NMS, mAP (no training) |
| 09 | Simple YOLO | a grid detector + coordinate/objectness loss |
| 10 | YOLO Depthwise | split-merge conv for a smaller, faster net |
| 10x | Lab: CoordConv Detector | same experiment, now on detection *(optional)* |
| 11 | Capstone | bring your own data: label, train, evaluate, ship |

Tip: Jump to the level that matches your current skill.  
If you are already 1337, skip the entire course. It's not designed for you.

### Enter as: `n00b`  
### Leave as: 

| Level | Description                                   |
|-------|-----------------------------------------------|
| 1337  | Develop a model to solve a specific problem   |
| Good  | Optimize an existing model for better performance |
| Medium | Understand the code and structure            |
| Still N00b | Can't even make it run                  |

- This repository is released under the **GNU GPLv3** license.
- **Note for non-English readers**: Let an LLM do the translation.
