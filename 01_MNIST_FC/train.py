import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SimpleNet

# ========== HYPERPARAMETERS (settings we choose before training) ==========
BATCH_SIZE = 64          # Number of images processed together in one step
LEARNING_RATE = 0.001    # How big of a step the optimizer takes when learning
EPOCHS = 5              # Number of complete passes through the training data


def set_seed(seed=1337):
    # Set random seeds for reproducibility.
    random.seed(seed)                        # Python's random module
    np.random.seed(seed)                     # NumPy's random module
    torch.manual_seed(seed)                  # PyTorch CPU random
    torch.cuda.manual_seed(seed)             # PyTorch GPU random (single GPU)
    torch.cuda.manual_seed_all(seed)         # PyTorch GPU random (all GPUs)
    torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    torch.backends.cudnn.benchmark = False     # Disable auto-optimization


set_seed(1337)  # Call the function to set all seeds

# ========== SET UP THE DEVICE (CPU or GPU) ==========
# Check if CUDA (NVIDIA GPU) is available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')   # Let the user know what's being used

# ========== DATA PREPROCESSING ==========
# transforms.Compose chains multiple transformations together
transform = transforms.Compose([
    # transforms.ToTensor(): 
    # - Converts PIL image or numpy array to PyTorch tensor
    # - Scales pixel values from [0, 255] to [0.0, 1.0]
    transforms.ToTensor(),
    
    # transforms.Normalize(mean, std):
    # - Normalizes tensor with: output = (input - mean) / std
    # - MNIST has pixel mean ~0.1307, std ~0.3081
    # - This gives roughly zero mean and unit variance (good for training)
    transforms.Normalize((0.1307,), (0.3081,))
])

# ========== LOAD DATASETS ==========
# Download MNIST (handwritten digits) if not already present
# train=True gets the training set (60,000 images)
train_dataset = torchvision.datasets.MNIST(
    root='./data',          # Directory to store downloaded data
    train=True,             # Load training set
    download=True,          # Download if not already there
    transform=transform     # Apply our preprocessing
)

# train=False gets the test set (10,000 images)
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,            # Load test set
    download=True,
    transform=transform
)

# ========== CREATE DATA LOADERS ==========
# DataLoader splits dataset into batches and shuffles (for training)
# Benefits:
# - Efficiently loads data in parallel
# - Shuffling prevents the model from learning order-based patterns
# - Batching makes training more stable and faster
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,  # Number of images per batch
    shuffle=True           # Randomize order each epoch
)

# For testing, we usually don't shuffle (order doesn't matter)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# ========== CREATE THE MODEL ==========
# Move the model to GPU (if available) for faster training
model = SimpleNet().to(device)

# ========== LOSS FUNCTION AND OPTIMIZER ==========
# CrossEntropyLoss combines LogSoftmax and Negative Log Likelihood Loss
# Good for multi-class classification problems (like digit classification)
criterion = nn.CrossEntropyLoss()

# Adam is an adaptive learning rate optimizer
# It's a good default choice for many problems
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(): # Train the model on the training dataset.
    
    
    model.train()  # Set model to training mode (enables dropout, batch norm, etc.)
    
    for epoch in range(EPOCHS):
        running_loss = 0.0  # Accumulator for loss across batches
        
        # Loop through all batches in the training data
        # images shape: (BATCH_SIZE, 1, 28, 28)
        # labels shape: (BATCH_SIZE,) containing digit labels (0-9)
        for images, labels in train_loader:
            # Move data to the same device as the model (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            
            # IMPORTANT: Zero the gradients from the previous iteration
            # Otherwise, gradients accumulate (usually not what we want)
            optimizer.zero_grad()
            
            # Forward pass: pass images through the model to get predictions
            outputs = model(images)  # outputs shape: (BATCH_SIZE, 10)
            
            # Calculate how wrong the predictions are
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients of loss with respect to model parameters
            # Automatic differentiation
            loss.backward()
            
            # Update model parameters using the computed gradients
            optimizer.step()
            
            # Add this batch's loss to the running total
            running_loss += loss.item()  # .item() extracts scalar from tensor
        
        # Print progress after each epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')
        # Lower loss means the model is learning!


def test(): # Evaluate the model on the test dataset (unseen data).
    
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    correct = 0   # Count of correctly classified images
    total = 0     # Total number of images processed
    
    # Disable gradient computation for efficiency and memory
    # We don't need gradients for testing since we're not training
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get the predicted class (index with highest score)
            # torch.max returns (values, indices)
            # dim=1 means along the class dimension
            _, predicted = torch.max(outputs, 1)  # predicted shape: (BATCH_SIZE,)
            
            # Update counters
            total += labels.size(0)                      # add batch size
            correct += (predicted == labels).sum().item() # count correct predictions
    
    # Calculate and print accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    # Typical accuracy for this simple network: ~97-98%


# ========== RUN THE TRAINING AND TESTING ==========
if __name__ == '__main__':
    # This check ensures training only runs if this script is executed directly
    # (not if it's imported as a module)
    train()   # Train the model for EPOCHS epochs
    test()    # Evaluate on test data after training