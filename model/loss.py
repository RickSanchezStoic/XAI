from collections.abc import Callable

import torch
from torch import nn


def crossEntropyLoss() -> Callable[[], nn.Module]:
    """
    Returns a CrossEntropyLoss loss function from PyTorch.

    This function provides the PyTorch implementation of the
    cross-entropy loss, which is typically used for classification tasks
    where the model's output is a probability distribution (softmax).

    Returns:
        Callable[[], nn.Module]: A callable that returns the CrossEntropyLoss module.
    """
    return torch.nn.CrossEntropyLoss
