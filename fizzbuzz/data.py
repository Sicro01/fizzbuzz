"""
We'll feed our data into our network in batches.
So here are some tools for iterating over data in batches
"""
from typing import Iterator, NamedTuple
import numpy as np

from tensor import tensor

Batch = NamedTuple("Batch", [("inputs", tensor), ("targets", tensor)])

class DataIterator:
  def __call__(self, inputs: tensor, targets: tensor) -> Iterator:
    raise NotImplementedError

class BatchIterator(DataIterator):
  def __init__(self, batch_size: int = 32, shuffle = True) -> None:
    self.batch_size = batch_size
    self.shuffle = shuffle

  def __call__(self, inputs: tensor, targets: tensor) -> Iterator:
    starts = np.arange(0, len(inputs), self.batch_size)
    if self.shuffle:
      np.random.shuffle(starts)

    for start in starts:
      end = start + self.batch_size
      batch_inputs = inputs[start:end]
      batch_targets = targets[start:end]
      yield Batch(batch_inputs, batch_targets)