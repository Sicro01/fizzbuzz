"""
We use am optimiser to adjust the parameters of our
network based on the gradients computed during backpropagation
"""
from nn import NeuralNet

class Optimiser:
  def step(self, net: NeuralNet) -> None:
    raise NotImplementedError

class SGD(Optimiser):
  def __init__(self, lr: float = 0.01) -> None:
    super().__init__()
    self.lr = lr

  def step(self, net: NeuralNet) -> None:
    for param, grad in net.params_and_grads():
      param -= self.lr * grad