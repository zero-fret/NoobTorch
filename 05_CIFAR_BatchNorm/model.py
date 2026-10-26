import torch
import torch.nn as nn


class BNCNN(nn.Module):
    """A small 3-block CIFAR classifier.

    use_bn=False replaces every BatchNorm with nn.Identity, so you can run
    the same network with and without BatchNorm and compare training.
    """
    def __init__(self, num_classes=10, use_bn=True):
        super(BNCNN, self).__init__()

        def norm(c):
            return nn.BatchNorm2d(c) if use_bn else nn.Identity()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                norm(cout),
                nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                norm(cout),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

        self.block1 = conv_block(3, 32)    # 32 -> 16
        self.block2 = conv_block(32, 64)   # 16 -> 8
        self.block3 = conv_block(64, 128)  # 8 -> 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)

    @torch.no_grad()
    def activation_report(self, images, device):
        """Run one batch and report mean/std/dead fraction after each ReLU.
        Healthy activations: mean ~0-ish, std ~1, dead fraction small (<0.5)."""
        self.train()  # BatchNorm stats matter here
        stats = []

        def hook(module, inputs, output):
            stats.append((round(output.mean().item(), 3),
                          round(output.std().item(), 3),
                          round((output == 0).float().mean().item(), 3)))

        handles = [m.register_forward_hook(hook) for m in self.modules()
                   if isinstance(m, nn.ReLU)]
        self(images.to(device))
        for h in handles:
            h.remove()
        print(f"{'ReLU layer':>10} | {'mean':>7} {'std':>7} {'dead %':>7}")
        for i, (mean, std, dead) in enumerate(stats):
            print(f"{i:>10} | {mean:>7} {std:>7} {dead:>7}")
        return stats
