# Noob's Hello World
- Minimum knowledge for a Neural Network that works
## Machine Learning Basics
### Model
- **Fully connected layer**: Every input node is connected to every output node
- **Activation function**: A nonlinear function. a model needs nonlinearity to reason, otherwise it’s just a linear transformation. Example: ReLU, y = max(x, 0)
- **Initialization** : setting initial values for model weights and biases before training. Proper initialization helps the model converge efficiently.
- **Forward propagation**: prediction, information flows from modelinput to output
- **Loss function**: difference between the model’s output and the correct prediction
- **Back Propagation**: uses the chain rule from calculus to compute the partial derivative of each node with respect to the loss function. It represents how to change the node to reduce loss. It does NOT "predict" from model output to model input.
- **Learning Rate**: How big of a step the optimizer takes when learning

### Training
- **One batch**: forward propagation → loss function → backpropagation
- **Epoch**: how many times to go through the data
- **Batch size**: the amount of data processed before each weight update
- **Dataloader**: feed the model with the correct form of data. Model usually reject raw data.
- **Train and Val**: Training and validation sets must be separate, otherwise it’s an open-book exam and overfits.

## PyTorch Basics
### Tensors
**Arrays** that can run on **GPUs** and automatically track operations for **gradient calculation**. Syntax is similar to Numpy. They are the **fundamental data structure** of PyTorch. 
- Number = 0 dimensional. 
- Vector = 1 dimensonal.
- Matrix = 2 dimensional.
- RGB Image = 3 dimensional.
- Deep learning models = even more dimensional.

### Automatic Differentiation 
PyTorch’s Autograd automatically **records operations** on tensors during forward propagation with requires_grad=True. **backward()** calculates derivatives for optimization based on the recordings.

## Example Code
```python
import torch
x = torch.tensor([2.0], requires_grad=True) 
# x is Tensor. 
# With requires_grad=True, 
# PyTorch records gradients in x.grad. 
# x must be float.
print(x.grad) # None. No backwards performed yet
y = x ** 2 + 3 * x + 1 # y = f(x). Recorded
# Operator Overloading. They are PyTorch operators.
y.backward()  # Automatic Differentiation. x.grad = dy/dx
print(x.grad)  # dy/dx = 2 * x + 3 = 7
x.grad.zero_() # Clear up gradients. Don't forget this.
print(x.grad) # Cleared up. 0

# None
# tensor([7.])
# tensor([0.])
```
## Begin Your Journey
Now, go to **01_MNIST_FC** to try your *1st* model. Feel free to come back if you forget something.