import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from model import ResNet18_CoordConv
import random
import numpy as np

BATCH_SIZE = 64
LEARNING_RATE = 0.1        
EPOCHS = 20             
WEIGHT_DECAY = 5e-4        
MOMENTUM = 0.9             

def set_seed(seed=42):
    random.seed(seed)            
    np.random.seed(seed)        
    torch.manual_seed(seed)         
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(1337)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


train_transform = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(p=0.5),         
    transforms.RandomCrop(32, padding=4),             
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.5,                   
                             scale=(0.02, 0.15),
                             ratio=(0.3, 3.3))
])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = ResNet18_CoordConv(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), 
                      lr=LEARNING_RATE, 
                      momentum=MOMENTUM, 
                      weight_decay=WEIGHT_DECAY)


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
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, LR: {current_lr:.6f}')


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
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train()
    test()