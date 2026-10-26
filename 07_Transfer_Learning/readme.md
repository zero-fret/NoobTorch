# 07 — Transfer learning & inference
Training a big network from scratch needs millions of labeled images. Almost nobody does that.

## The one new idea
**Transfer learning**: take a backbone trained on a huge dataset (e.g. ImageNet, 1.2M images) and reuse its low/mid-level features (edges, textures, parts). Two common recipes:

1. **Feature extraction (train.py)**: freeze the backbone (`requires_grad=False`), replace its final classifier with a fresh layer for your classes, train only that layer. Minutes, cheap, works on small data.
2. **Fine-tuning (finetune.py)**: load the frozen head, keep the low-level stem + layer1/layer2 frozen, but unfreeze layer3/layer4 and the head. Use a *small* backbone LR (2e-4) vs a larger head LR (1e-3) so pretrained features are adapted, not destroyed. This is where transfer learning pulls ahead of training from scratch.

## Save / load
- `torch.save(model.state_dict(), '....pth')` dumps only the learned tensors.
- Reconstruct the same architecture, then `model.load_state_dict(torch.load(...))`.
- Load with `weights_only=True` unless the file genuinely contains non-tensor data (a safety default).

## What changed vs lesson 06
We don't define a network at all — we instantiate `torchvision`'s ResNet18 with downloaded weights and touch only its `fc` layer. That is also why inference is its own script: saving a model you never use is pointless.

## Run
```
python train.py          # frozen head: downloads pretrained weights, trains head, saves .pth
python finetune.py       # unfreeze upper stages, fine-tune -> cifar10_resnet18_finetuned.pth
python inference.py                      # classify the first CIFAR test image
python inference.py --image my_photo.jpg # classify your own image
```

Results on CIFAR-10 (RTX 4050): frozen head **80.01%**, fine-tuned **95.77%**
(vs. ResNet18 from scratch in lesson 06 at 92.78%).

## Checkpoint 7
- [ ] I can freeze a backbone and train only a new head.
- [ ] I know when to fine-tune and why it needs a small LR.
- [ ] I can save, rebuild, and run a trained model on a new image.

→ Next: **07x_Lab_CoordConv** (optional). Instead of being told a trick works, run a controlled experiment and let the data decide.
