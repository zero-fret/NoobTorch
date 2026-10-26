import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3,
                                   stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.pointwise(self.act(self.bn1(self.depthwise(x))))))


class SimpleYOLO_DSConv_Coord(nn.Module):
    """Lesson-10 network + optional coordinate channels.

    use_coords=False -> exactly the lesson-10 network (control).
    use_coords=True  -> 2 extra X/Y channels at the input (treatment).
    Everything else is identical, so any AP change is caused by the
    coordinates alone.
    """
    def __init__(self, S=20, use_coords=True):
        super().__init__()
        self.S = S
        self.use_coords = use_coords
        stem_in = 5 if use_coords else 3
        self.features = nn.Sequential(
            nn.Conv2d(stem_in, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            nn.Conv2d(512, 5, 1),
        )

    def add_coords(self, x):
        B, _, H, W = x.shape
        y = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xc = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, y, xc], dim=1)

    def forward(self, x):
        if self.use_coords:
            x = self.add_coords(x)
        return self.features(x)
