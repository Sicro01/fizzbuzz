"""
Our neural net will b emade up of layers.
Each layer needs to pass it's inputs forwards
and propogate gradiets backwards. For example,
a neural net might look like this:

inputs -> Linear -> Tanh -> Linear -> outputs
"""
import numpy as np
from typing import Dict, Callable

from tensor import tensor

class Layer:
  def __init__(self) -> None:
    self.params: Dict[str, tensor] = {}
    self.grads: Dict[str, tensor] = {}

  def forward(self, inputs: tensor) -> tensor:
    """
    Produce the output corresponding to these inputs
    """
    raise NotImplementedError

  def backward(self, grad: tensor) -> tensor:
    """
    Backpropogate this gradient through the layer
    """
    raise NotImplementedError

class Linear(Layer):
  """
  Computes output = inputs @ w + b (where @ is a matrix)
  """
  def __init__(self, input_size: int, output_size: int) -> None:
    # inputs will be (batch size, input_size)
    # outputs will be (batch_size, outputs_size)
    # initialise the parameters
    super().__init__()
    self.params["w"] = np.random.randn(input_size, output_size)
    self.params["b"] = np.random.randn(output_size)
  
  def forward(self, inputs: tensor) -> tensor:
    """
    outputs = inputs @ w + b
    """
    self.inputs = inputs
    return inputs @ self.params["w"] + self.params["b"]

  def backward(self, grad: tensor) -> tensor:
    """
    e.g. f'(5) = slope of tangent line at 5 or rate of change py y with respect to x of f
    if y = f(x) and x = a * b + c
    then dy/da = f'(x) * b
    and dy/db = f'(x) * a
    and dy/dc = f'(x)

    if y = f(x) and x = a @ b + c
    then dy/da = f'(x) @ b.T
    and dy/db = a.T @ f'(x)
    and dy/dc = f'(x)
    """
    self.grads["b"] = np.sum(grad, axis=0)
    self.grads["w"] = self.inputs.T @ grad 
    return grad @ self.params["w"].T

F = Callable[[tensor], tensor]

class Activation(Layer):
  """
  An activation layer just applies a function elementwise to it's inputs
  """
  def __init__(self, f: F, f_prime: F) -> None:
    super().__init__()
    self.f = f
    self.f_prime = f_prime

  def forward(self, inputs: tensor) -> tensor:
    self.inputs = inputs
    return self.f(inputs)

  def backward(self, grad: tensor) -> tensor:
    """
    if y = f(x) and x = g(z)
    then dy/dz = f'(x) * g'(z)
    """
    return self.f_prime(self.inputs) * grad


def tanh(x: tensor) -> tensor:
  return np.tanh(x)

def tanh_prime(x: tensor) -> tensor:
  y = tanh(x)
  return 1 - y ** 2

class Tanh(Activation):
  def __init__(self):
    super().__init__(tanh, tanh_prime)