import torch
import torch.nn as nn


class CoordCNN(nn.Module):
    """Lesson-02 sized CNN used as an experiment platform.

    use_coords=True  -> two normalized coordinate channels (X, Y) are
                        concatenated to the image, i.e. CoordConv.
    use_coords=False -> identical network with 1 input channel (control).
    """
    def __init__(self, use_coords=True, activation='leaky_relu'):
        super(CoordCNN, self).__init__()
        self.use_coords = use_coords
        in_ch = 3 if use_coords else 1

        self.conv1 = nn.Conv2d(in_ch, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 7 * 7, 40)
        self.fc2 = nn.Linear(40, 10)
        self.act = nn.LeakyReLU(0.01) if activation == 'leaky_relu' else nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def add_coords(self, x):
        """Append two channels encoding each pixel's position in [-1, 1]."""
        batch, _, H, W = x.shape
        y = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(batch, 1, H, W)
        xc = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(batch, 1, H, W)
        return torch.cat([x, y, xc], dim=1)

    def forward(self, x):
        if self.use_coords:
            x = self.add_coords(x)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)
