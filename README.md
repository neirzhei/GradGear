A minimal, from-scratch implementation of an automatic differentiation engine in Python. This project demonstrates the core mechanics of backpropagation by tracking scalar values and their gradients through a dynamically constructed computational graph. This was built for educational purposes to learn how automatic differentiation works under the hood.

### What Can It Do?

- The `Value` object holds a scalar value and automatically tracks gradients through computations.
- Calling `.backward()` on the final result efficiently computes gradients for all intermediate nodes using reverse-mode autodiff.
- Supports addition, subtraction, multiplication, division, power, and `tanh`. This allows you to build and train small neural networks manually.

### Usage Example

```python
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
print(f'{a=}, {b=}, {c=}')
print(f'{d=}, {e=}, {y=}')

# Expected output:
# a=Value(data=2.0, grad=-1.5), b=Value(data=-3.0, grad=1.0) ...
