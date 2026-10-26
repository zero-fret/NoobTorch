import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from common import set_seed, get_device, get_cifar10_loaders, fit, evaluate
from model import ResNet18

BATCH_SIZE = 128
EPOCHS = 25
LR = 0.1                 # SGD + momentum tolerates a much larger LR than Adam
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

# Strong augmentation is the norm for CIFAR: RandAugment, flip, crop, cutout.
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE, val_size=5000, augment='strong', num_workers=4)

model = ResNet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if __name__ == '__main__':
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  device, epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'Final test accuracy: {test_acc:.2f}%')
