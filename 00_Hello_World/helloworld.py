import torch
x = torch.tensor([2.0], requires_grad=True) # x is Tensor. With requires_grad=True, PyTorch records gradients in x.grad. x must be float.
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