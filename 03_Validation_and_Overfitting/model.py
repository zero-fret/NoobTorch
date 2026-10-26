import torch.nn as nn


class OverfitNet(nn.Module):
    """Deliberately over-sized FC network. With few training samples it can
    memorize them, which makes the overfitting lesson obvious."""
    def __init__(self, hidden=256):
        super(OverfitNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
