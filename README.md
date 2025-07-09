## AutoGrads: From Scalars to Tensors
From-scratch lighweight implementations of automatic differentiation engine in Python. This project demonstrates the core mechanics of backpropagation by tracking scalar values (and later, tensors) and their gradients through a dynamically constructed computational graph. This was built for educational purposes to learn how machine learning libraries like PyTorch work under the hood.
### `ScalarGrad.py`
- The `Value` object holds a scalar value and automatically tracks gradients through computations.
- Calling `.backward()` on the final result efficiently computes gradients for all intermediate nodes using reverse-mode autodiff.
- Supports addition, subtraction, multiplication, division, power, and `tanh`. This allows you to build and train small neural networks manually.

#### Usage Example

```python
# from scalargrad.py import Value

# Define expression: y = tanh(a*b + c)
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b
e = d + c
y = e.tanh()

# Compute gradients for all nodes
y.backward()

# Print values and their gradients
print(f'{a=}')
print(f'{b=}')
print(f'{c=}')
print(f'{y=}')

# Expected output (example):
# a=Value(data=2.0, grad=-1.5)
# b=Value(data=-3.0, grad=1.0)
# c=Value(data=10.0, grad=0.5)
# y=Value(data=0.999329299739067, grad=1.0)
```
### `TensorGrad.py`
Upgraded my initial [`ScalarGrad.py`](ScalarGrad.py) to support tensors.
- The `Value` object wraps a `numpy` array and automatically tracks gradients through tensor computations.
- Handles NumPy's broadcasting rules properly during backpropagation.
- Calling `.backward()` on the final result efficiently computes gradients for all tensors in the graph using reverse-mode autodiff.
- Supports element-wise addition, multiplication, matrix multiplication (`@`), power, `relu`, `exp`, `log`, and `sum`.
- Provides the fundamental building blocks to construct and train neural networks like Multi-Layer Perceptrons (MLPs).
#### Usage Example
```python
import numpy as np
# from tensorgrap.py import Value

# Defining a simple matrix operation: z = relu(x @ W + b)
x = Value(np.array([[1.0, 2.0, 3.0]]))
W = Value(np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]))
b = Value(np.array([[0.5, -0.5]]))

# Forward pass
z = (x @ W + b).relu()

# Computes gradients for all nodes
z.sum().backward()

# Prints values and their gradients
print("Input and Parameters:")
print(f"x = {x.data}, grad =\n{x.grad}\n")
print(f"W = {W.data}, grad =\n{W.grad}\n")
print(f"b = {b.data}, grad =\n{b.grad}\n")

print(f"Output:\nz = {z.data}, grad =\n{z.grad}")
```
