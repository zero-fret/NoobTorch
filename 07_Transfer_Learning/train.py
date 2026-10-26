import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from common import set_seed, get_device, get_cifar10_loaders, fit, evaluate

BATCH_SIZE = 64
EPOCHS = 3                 # frozen backbone trains fast; set ~10 for full fine-tuning
LR = 1e-3

set_seed(1337)
device = get_device()
print(f'Using device: {device}')

# A torchvision ImageNet ResNet18 has a 7x7 stride-2 conv plus a maxpool, so it
# is built for ~224px inputs. Feeding it 32/64px CIFAR images throws away most
# of the signal and freezes the benefit of pretraining -- hence 224px here,
# together with ImageNet normalization (norm='imagenet') that the backbone expects.
# No spatial augmentation here: this lesson is about transfer learning, not augmentation.
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=BATCH_SIZE, val_size=5000, augment=None,
    resize=224, num_workers=4, norm='imagenet')

# --- Transfer learning step 1: use a backbone trained on ImageNet ----------
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# --- Step 2: freeze everything, then replace the final layer ----------------
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)   # only this new layer trains
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

if __name__ == '__main__':
    print('Feature-extraction mode: backbone frozen, training the new head.')
    fit(model, train_loader, val_loader, criterion, optimizer,
        device, epochs=EPOCHS)
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f'Test accuracy (frozen backbone): {test_acc:.2f}%')

    torch.save(model.state_dict(), 'cifar10_resnet18.pth')
    print('Saved model to cifar10_resnet18.pth')

    # --- Step 3 (optional): full fine-tuning --------------------------------
    # Unfreeze, then re-optimize the whole network with a small LR for a few
    # epochs so the pretrained features are adjusted but not destroyed:
    #
    # for p in model.parameters():
    #     p.requires_grad = True
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # fit(model, train_loader, val_loader, criterion, optimizer, device, epochs=5)
