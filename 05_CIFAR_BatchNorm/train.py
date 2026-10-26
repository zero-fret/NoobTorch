import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from common import set_seed, get_device, get_cifar10_loaders, fit, evaluate
from model import BNCNN

USE_BN = True          # flip to False to train the same net without BatchNorm
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 5e-4

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE, val_size=5000, augment='basic', num_workers=4)

model = BNCNN(use_bn=USE_BN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if __name__ == '__main__':
    print(f'BatchNorm enabled: {USE_BN}')
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  device, epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'Final test accuracy: {test_acc:.2f}%')

    # Health check: how do the activations look after training?
    probe_images, _ = next(iter(train_loader))
    print('\nActivation report (mean, std, dead-ReLU fraction):')
    model.activation_report(probe_images, device)
