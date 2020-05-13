"""
A neural net is just a collection of layers.
It behanve sa lot like a layer itself, although
we're not going to make it one
"""
from typing import Sequence, Iterator, Tuple

from tensor import tensor
from layers import Layer

class NeuralNet:
  def __init__(self, layers: Sequence[Layer]) -> None:
    self.layers = layers

  def forward(self, inputs: tensor) -> tensor:
    for layer in self.layers:
      inputs = layer.forward(inputs)
    return inputs
  
  def backward(self, grad: tensor) -> tensor:
    for layer in reversed(self.layers):
      grad = layer.backward(grad)
    return grad

  def params_and_grads(self) -> Iterator[Tuple[tensor, tensor]]:
    for layer in self.layers:
      for name, param in layer.params.items():
        grad = layer.grads[name]
        yield param, grad