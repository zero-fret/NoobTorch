import torch
import torch.nn as nn

class CoordCNN(nn.Module):
    def __init__(self):
        super(CoordCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)     # 336 params 
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)    # 2,616 params 
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 7 * 7, 70)            # 82,390 params 
        self.fc2 = nn.Linear(70, 10)                    # 710 params 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def add_coords(self, x):
        batch, _, H, W = x.shape
        # Normalize to [-1, 1]
        y_coords = torch.linspace(-1, 1, steps=H, device=x.device)
        y_coords = y_coords.view(1, 1, H, 1).expand(batch, 1, H, W)

        x_coords = torch.linspace(-1, 1, steps=W, device=x.device)
        x_coords = x_coords.view(1, 1, 1, W).expand(batch, 1, H, W)

        return torch.cat([x, y_coords, x_coords], dim=1)
        
    def forward(self, x):
        x = self.add_coords(x)  # [B, 1, 28, 28] → [B, 3, 28, 28]
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x