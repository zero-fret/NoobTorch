import torch.nn as nn


class SimpleNet(nn.Module):
    """A 3-layer fully connected network for MNIST.

    28x28 image -> flatten -> 128 -> 64 -> 10 class scores.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)   # 784 inputs -> 128 neurons
        self.fc2 = nn.Linear(128, 64)        # 128 -> 64
        self.fc3 = nn.Linear(64, 10)         # 64 -> 10 digit classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)            # flatten: (B, 1, 28, 28) -> (B, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                      # raw scores (logits); CrossEntropyLoss adds softmax
        return x
