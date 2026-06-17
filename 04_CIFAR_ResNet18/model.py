import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_CoordConv(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CoordConv, self).__init__()
        
        self.in_channels = 64

        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def add_coords(self, x):
        batch, _, H, W = x.shape
        
        # [-1, 1]
        y_coords = torch.linspace(-1, 1, steps=H, device=x.device)
        y_coords = y_coords.view(1, 1, H, 1).expand(batch, 1, H, W)
        
        # [-1, 1]
        x_coords = torch.linspace(-1, 1, steps=W, device=x.device)
        x_coords = x_coords.view(1, 1, 1, W).expand(batch, 1, H, W)
        

        return torch.cat([x, y_coords, x_coords], dim=1)
    
    def forward(self, x):

        x = self.add_coords(x)  # [B, 3, 32, 32] → [B, 5, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    model = ResNet18_CoordConv(num_classes=10)
    x = torch.randn(1, 3, 32, 32) 
    y = model(x)
    print(f" {y.shape}")  # [1, 10]
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,}")
   
