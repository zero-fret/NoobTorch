import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from common import set_seed, get_device, get_mnist_loaders, fit, evaluate
from common.metrics import confusion_matrix, per_class_precision_recall, plot_curves
from model import OverfitNet

# --- knobs to experiment with (see the readme exercises) ----------------
TRAIN_SUBSET = 2000   # tiny -> the big model memorizes; try 50000
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

# THREE sets: train (50k after holding out val) / val (10k) / test (10k).
train_loader, val_loader, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=10000, augment=False, num_workers=2)

# Keep only TRAIN_SUBSET training samples so overfitting shows up fast.
train_loader = torch.utils.data.DataLoader(
    Subset(train_loader.dataset, range(TRAIN_SUBSET)),
    batch_size=BATCH_SIZE, shuffle=True)

model = OverfitNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def main():
    history = fit(model, train_loader, val_loader, criterion, optimizer,
                  device, epochs=EPOCHS)
    plot_curves(history, save_path='results.png')

    # Final, honest number: the test set is used exactly once.
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f'\nTest Accuracy: {test_acc:.2f}%')

    cm = confusion_matrix(preds, labels, num_classes=10)
    precision, recall, f1 = per_class_precision_recall(cm)
    print(f'Macro precision {precision.mean():.3f} | recall {recall.mean():.3f} '
          f'| F1 {f1.mean():.3f}')

    # Most common confusions (which digits get mistaken for which).
    pairs = []
    for t in range(10):
        for p in range(10):
            if t != p and cm[t, p] > 0:
                pairs.append((int(cm[t, p]), t, p))
    print('Top confusions (true -> predicted):')
    for n, t, p in sorted(pairs, reverse=True)[:5]:
        print(f'  {t} -> {p}: {n} times')


if __name__ == '__main__':
    main()
