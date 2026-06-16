import torch.nn as nn

class SimpleNet(nn.Module):
    # A simple 3-layer neural network for MNIST digit classification.
    # This network takes 28x28 pixel images and outputs probabilities for 10 digits (0-9).
    def __init__(self):
        super(SimpleNet, self).__init__() # Always call super().__init__() first when subclassing nn.Module
        # nn.Linear(input_features, output_features) creates a fully connected layer
        self.fc1 = nn.Linear(28 * 28, 128)   # Layer 1: 28*28 = 784 input pixels → 128 neurons. fc = "fully connected"
        self.fc2 = nn.Linear(128, 64) # Layer 2: 128 neurons from previous layer → 64 neurons
        self.fc3 = nn.Linear(64, 10) # Layer 3: 64 neurons → 10 output classes (digits 0-9)
        self.relu = nn.ReLU() # ReLU activation function: max(0, x). Very common non-linear function.
    
    def forward(self, x): # Defines how data flows through the network. X: Input tensor of shape (batch_size, 1, 28, 28) from MNIST
        x = x.view(x.size(0), -1) # Flatten the image: (batch_size, 1, 28, 28) → (batch_size, 784). view() reshapes tensors; -1 means "infer this dimension from others"
        x = self.relu(self.fc1(x)) # Pass through layer 1, then apply ReLU activation
        x = self.relu(self.fc2(x)) # Pass through layer 2, then apply ReLU activation
        x = self.fc3(x) # Pass through layer 3 (no activation - raw logits for CrossEntropyLoss)
        
        return x # Output tensor of shape (batch_size, 10) with raw scores for each digit
