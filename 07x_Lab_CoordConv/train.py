import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from common import set_seed, get_device, get_mnist_loaders, fit, evaluate
from model import CoordCNN

# The two experimental variables. Keep everything else fixed.
parser = argparse.ArgumentParser()
parser.add_argument('--coords', action='store_true',
                    help='enable coordinate channels (treatment); omit for control')
parser.add_argument('--activation', default='leaky_relu', choices=['relu', 'leaky_relu'])
_args = parser.parse_args()
USE_COORDS = _args.coords       # True = CoordConv, False = control
ACTIVATION = _args.activation   # 'relu' or 'leaky_relu'

BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

train_loader, val_loader, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=10000, augment=True, num_workers=4)

model = CoordCNN(use_coords=USE_COORDS, activation=ACTIVATION).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if __name__ == '__main__':
    print(f'coords={USE_COORDS} activation={ACTIVATION}')
    fit(model, train_loader, val_loader, criterion, optimizer,
        device, epochs=EPOCHS, scheduler=scheduler)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'Test accuracy: {test_acc:.2f}%')
