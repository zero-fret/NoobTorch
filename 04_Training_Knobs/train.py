import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from common import set_seed, get_device, get_mnist_loaders, fit, evaluate
from model import KnobCNN

# ===== The four knobs. Flip ONE at a time and compare val accuracy. =====
USE_AUGMENTATION = True     # random rotation/shift/scale -> free data
USE_SCHEDULER = True        # cosine LR decay
USE_WEIGHT_DECAY = True     # L2 penalty on weights
ACTIVATION = 'leaky_relu'   # 'relu' or 'leaky_relu'
# ===========================

BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 1e-4 if USE_WEIGHT_DECAY else 0.0
NUM_WORKERS = 4

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

train_loader, val_loader, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=10000,
    augment=USE_AUGMENTATION, num_workers=NUM_WORKERS)

model = KnobCNN(activation=ACTIVATION).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if USE_SCHEDULER else None

if __name__ == '__main__':
    print(f'augmentation={USE_AUGMENTATION} scheduler={USE_SCHEDULER} '
          f'weight_decay={USE_WEIGHT_DECAY} activation={ACTIVATION}')
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  device, epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'Final test accuracy: {test_acc:.2f}%')
