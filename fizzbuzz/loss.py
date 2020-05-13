"""
A loss function measures how good our poredictions are,
we can use this to adjust the parametrs of our network
"""
import numpy as np

from tensor import tensor

class Loss:
  def loss(self, predicted: tensor, actual: tensor) -> float:
    raise NotImplementedError

  def grad(self, predicted: tensor, actual: tensor) -> tensor:
    raise NotImplementedError

class MSE(Loss):
  """
  MSE is mean squared error, although we're 
  just going to do squared error
  """
  def loss(self, predicted: tensor, actual: tensor) -> float:
    return np.sum((predicted - actual) ** 2)

  def grad(self, predicted: tensor, actual: tensor) -> tensor:
    return 2 * (predicted - actual)