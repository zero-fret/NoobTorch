"""Transfer learning step 2: fine-tune the frozen probe.

The frozen head (train.py) only reached ~80% because the linear head trained for
3 epochs and the backbone's ImageNet features were never adapted to CIFAR. Here we:

  1. load the frozen-probe weights (cifar10_resnet18.pth),
  2. keep the low-level stem + layer1/layer2 FROZEN (edges/textures transfer
     unchanged), but unfreeze layer3, layer4 and the new head,
  3. train with a small backbone LR and a larger head LR, with light augmentation.

Usage:
    python finetune.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from common import set_seed, get_device, get_cifar10_loaders, fit, evaluate

BATCH_SIZE = 32
EPOCHS = 12
BACKBONE_LR = 2e-4     # small: adapt pretrained features, don't destroy them
HEAD_LR = 1e-3
FROZEN_PREFIXES = ('conv1', 'bn1', 'layer1', 'layer2')   # keep these frozen

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

# Same preprocessing as train.py: 224px + ImageNet stats. Add light flip/crop
# augmentation now that we are adapting the features.
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE, val_size=5000, augment='basic',
    resize=224, num_workers=4, norm='imagenet')

# Build the same architecture and load the frozen-probe checkpoint.
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('cifar10_resnet18.pth',
                                 map_location=device, weights_only=True))
model.to(device)

# Freeze everything first, then unfreeze the upper stages + head.
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if not name.startswith(FROZEN_PREFIXES):
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Fine-tuning trainable parameters: {trainable:,} '
      f'(stem/layer1/layer2 frozen)')

# Differential LR: slow for the pretrained backbone, faster for the fresh head.
backbone_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('fc.')]
head_params = [p for n, p in model.named_parameters()
               if p.requires_grad and n.startswith('fc.')]
optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': BACKBONE_LR, 'weight_decay': 1e-4},
    {'params': head_params,     'lr': HEAD_LR,     'weight_decay': 1e-4},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

if __name__ == '__main__':
    fit(model, train_loader, val_loader, criterion, optimizer, device,
        epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'Test accuracy (fine-tuned): {test_acc:.2f}%')

    torch.save(model.state_dict(), 'cifar10_resnet18_finetuned.pth')
    print('Saved model to cifar10_resnet18_finetuned.pth')
