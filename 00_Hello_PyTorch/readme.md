# 00 — Hello PyTorch
The minimum you need before the first trainable network: a **tensor** and **autograd**.

## The one new idea
A **tensor** is an array (0-D number, 1-D vector, 2-D matrix, 3-D RGB image, …) that can run on GPU and remembers the operations used to build it. If `requires_grad=True`, calling `.backward()` fills each tensor's `.grad` with its derivative — no need to derive anything by hand.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()      # x.grad = dy/dx = 2*x + 3
```

Three habits this lesson establishes:
- gradients **accumulate**, so zero them before each new step (`x.grad.zero_()`, later `optimizer.zero_grad()`);
- tensors that need gradients must be **floats**;
- trailing `_` means **in place** (`zero_`, `add_`, `copy_`).

## Run it
```
pip install -r ../requirements.txt
python helloworld.py
```

## Expected
```
None
tensor([7.])
tensor([0.])
loaded: {'x': tensor([2.], requires_grad=True), 'grad': tensor([0.])}
```

## Glossary (reference — don't memorize, come back when you need it)
- **Model / network**: a function with many parameters.
- **Layer**: one transformation of the data (e.g. `nn.Linear`, `nn.Conv2d`).
- **Activation**: a nonlinear function (e.g. ReLU `max(x, 0)`); without it a deep net is just one linear transform.
- **Forward pass**: compute predictions from inputs.
- **Loss**: a number saying how wrong the predictions are.
- **Backward pass**: compute each parameter's gradient of the loss via the chain rule.
- **Optimizer**: uses the gradients to change parameters (e.g. Adam, SGD).
- **Learning rate**: step size of the optimizer.
- **Batch / epoch**: a batch = samples processed before one weight update; an epoch = one full pass over the data.
- **DataLoader**: feeds batches in the shape the model expects.
- **train / val / test**: train adjusts parameters; val is used to tune; test is used exactly once.

## Checkpoint 0
- [ ] I can compute a derivative with `.backward()` and read `.grad`.
- [ ] I can explain why I clear gradients.
- [ ] I can save and load a `state_dict`.

→ Next: **01_MNIST_FC**, your first model that actually works.
