import torch.nn as nn


def make_activation(kind):
    if kind == 'relu':
        return nn.ReLU()
    if kind == 'leaky_relu':
        return nn.LeakyReLU(0.01)   # small negative slope prevents dead neurons
    raise ValueError(kind)


class KnobCNN(nn.Module):
    """Same shape as the lesson-02 CNN, but the activation is swappable."""
    def __init__(self, activation='relu'):
        super(KnobCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 7 * 7, 70)
        self.fc2 = nn.Linear(70, 10)
        self.act1 = make_activation(activation)
        self.act2 = make_activation(activation)
        self.act3 = make_activation(activation)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.act3(self.fc1(x)))
        x = self.fc2(x)
        return x
