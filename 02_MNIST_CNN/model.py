import torch.nn as nn


class SimpleCNN(nn.Module):
    """Two conv+pool blocks then a small FC head. ~86k params."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)     # 120 params: 12*(1*3*3 + 1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)   # 2,616 params
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 7 * 7, 70)           # 82,390 params
        self.fc2 = nn.Linear(70, 10)                   # 710 params
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))   # 14 -> 7
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))  # dropout only in the head
        x = self.fc2(x)
        return x
