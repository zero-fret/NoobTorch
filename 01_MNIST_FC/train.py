import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from common import set_seed, get_device, get_mnist_loaders
from model import SimpleNet

# This lesson shows the FULL training loop on purpose. From lesson 02 onward we
# reuse common.engine.fit so you don't read these 40 lines every time.

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

# val_size=0 on purpose here: lesson 01 is only train + test.
# Lesson 03 introduces the validation set and explains why we need it.
train_loader, _, test_loader = get_mnist_loaders(
    batch_size=BATCH_SIZE, val_size=0, augment=False, num_workers=2)

model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()                 # log-softmax + NLL in one module
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    for epoch in range(EPOCHS):
        model.train()                             # enable dropout/batchnorm updates
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                 # clear last batch's gradients
            outputs = model(images)               # forward
            loss = criterion(outputs, labels)
            loss.backward()                       # backward: fill .grad
            optimizer.step()                      # update parameters
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}')


def test():
    model.eval()                                  # disable dropout, freeze batchnorm
    correct = total = 0
    with torch.no_grad():                         # no gradients -> less memory
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100*correct/total:.2f}%')


if __name__ == '__main__':
    train()
    test()
