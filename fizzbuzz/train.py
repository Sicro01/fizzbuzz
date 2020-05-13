"""
Here's a function that can train a Neural Net
"""
from tensor import tensor
from nn import NeuralNet
from loss import Loss, MSE
from optim import Optimiser, SGD
from data import DataIterator, BatchIterator

def train(net: NeuralNet,
          inputs: tensor,
          targets: tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimiser: Optimiser = SGD()) -> None:

    for epoch in range(num_epochs):
      epoch_loss = 0.0
      for batch in iterator(inputs, targets):
        predicted = net.forward(batch.inputs)
        epoch_loss += loss.loss(predicted, batch.targets)
        grad = loss.grad(predicted, batch.targets)
        net.backward(grad)
        optimiser.step(net)
      print(epoch, epoch_loss)


