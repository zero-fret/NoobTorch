import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from common import set_seed, get_device, get_mnist_loaders, train_one_epoch, evaluate
from model import SimpleCNN

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

train_loader, _, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=0, augment=False, num_workers=2)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}, Acc: {acc:.2f}%')
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'Test Accuracy: {test_acc:.2f}%')
