# extended the ScalarGrad to support multidimensional numpy arrays.

import numpy as np

class Value:
# stores the array's value, gradient and the operation that produced it.

    def __init__(self, data, _children=(), _op=''):
        # The underlying tensor data.
        self.data = np.asarray(data, dtype=np.float64)
        # The gradient of the graph's final output with respect to this value's data.
        self.grad = np.zeros_like(self.data)
        # A function that computes the local gradient for the children of this value.
        self._backward = lambda: None
        # The set of child Value objects that were inputs to the operation creating this value.
        self._prev = set(_children)
        # The operation that created this value (kept for debugging).
        self._op = _op

    # ----------- Core Mathematical Operations --------------

    def __add__(self, other):
        # '+' operator.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Applies the chain rule for addition.
            # Broadcasting rules need to be handled for the gradients.
            self.grad += self._get_broadcasted_grad(out.grad)
            other.grad += other._get_broadcasted_grad(out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other):
        # '*' operator for element-wise multiplication.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Applies the chain rule for multiplication.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        # '**' operator for scalar powers.
        assert isinstance(power, (int, float)), "Only supports scalar powers"
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            # Applies the chain rule for the power function.
            self.grad += (power * self.data**(power - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        # '@' operator for matrix multiplication.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            # Applies the chain rule for matrix multiplication.
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ----------- Activation Functions and Other Ops ----------------

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            # Gradient is 1 for positive inputs and 0 for negative inputs.
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self,), 'exp')
        def _backward():
            # As the derivative of e^x is e^x.
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        x = self.data
        # to avoid log(0).
        out = Value(np.log(x.clip(min=1e-8)), (self,), 'log')
        def _backward():
            # As the derivative of log(x) is 1/x.
            self.grad += (1 / x.clip(min=1e-8)) * out.grad
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        # Summation over tensor axes.
        out = Value(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            # Gradient of sum distributes over the elements.
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    # ------------- Backpropagation ------------------

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        # Builds a topological sort of the graph.
        build_topo(self)
        
        # Initializes the gradient of the final output Value to 1.
        self.grad = np.ones_like(self.data)

        # Applies the chain rule in reverse order of the topological sort.
        # This ensures that gradients are computed in the correct order.
        for node in reversed(topo):
            node._backward()

    def _get_broadcasted_grad(self, grad):
        if self.data.shape == grad.shape:
            return grad
        
        # Handle dimensions that were added during broadcasting.
        axis = tuple(range(grad.ndim - self.data.ndim))
        axis += tuple(i for i, (s, g) in enumerate(zip(self.data.shape, grad.shape[-self.data.ndim:])) if s == 1 and g > 1)
        
        return np.sum(grad, axis=axis, keepdims=True).reshape(self.data.shape)

    # for more natural mathematical expressions.
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1
    def __rmul__(self, other): return self * other
    def __radd__(self, other): return self + other

    def __repr__(self):
        # clean string representation of the Value object.
        return f"Value(shape={self.data.shape})"
