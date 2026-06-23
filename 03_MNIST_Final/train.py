import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from model import CoordCNN


# ========== HYPERPARAMETERS ==========
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 200


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1337)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== DATA TRANSFORMS (with augmentation for training) ==========
# Training transform: add random affine augmentation (rotation, translation, scaling)
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,            # random rotation between -10° and +10°
        translate=(0.1, 0.1),  # random horizontal/vertical shift up to 10% of image size
        scale=(0.9, 1.1)       # random scaling between 90% and 110%
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Test transform: no augmentation, only standard normalisation
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ========== LOAD DATASETS ==========
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=train_transform      # augmented training data
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=test_transform       # untouched test data
)

# ========== DATA LOADERS ==========
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ========== MODEL, LOSS, OPTIMIZER ==========
model = CoordCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')


def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    train()
    test()