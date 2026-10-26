import torch

# --- Tensors + autograd -------------------------------------------------
x = torch.tensor([2.0], requires_grad=True)  # requires_grad=True: PyTorch records gradients into x.grad. x must be float.
print(x.grad)  # None -- no backward() has happened yet

y = x ** 2 + 3 * x + 1  # y = f(x), every operation here is recorded
y.backward()            # automatic differentiation: x.grad = dy/dx
print(x.grad)           # dy/dx = 2*x + 3 = 7

x.grad.zero_()          # clear gradients (in place; note the trailing underscore)
print(x.grad)           # tensor([0.])

# --- Save / load --------------------------------------------------------
# You save a trained model by dumping its parameters (a state_dict) to disk.
# There is no model here yet, so we just save the tensor -- the call is identical
# for a model: torch.save(model.state_dict(), 'model.pth').
torch.save({'x': x, 'grad': x.grad}, 'demo.pth')
loaded = torch.load('demo.pth', weights_only=False)
print('loaded:', loaded)

# Expected output:
# None
# tensor([7.])
# tensor([0.])
# loaded: {'x': tensor([2.], requires_grad=True), 'grad': tensor([0.])}
