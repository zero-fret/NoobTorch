import torch.nn as nn


class SimpleYOLO(nn.Module):
    """Fully convolutional detector: 640x640 image -> 5 x 20 x 20 output.

    Each of the 20x20 cells predicts 5 values:
        tx, ty, tw, th, conf_logits
    """
    def __init__(self, S=20, negative_slope=0.01):
        super().__init__()
        self.S = S
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(512, 5, 1),   # detection head
        )

    def forward(self, x):
        return self.features(x)
