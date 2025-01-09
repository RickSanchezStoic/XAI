import torch
import torch.nn as nn

import torch.nn as nn


class LinearModel(nn.Module):
    """
    A customizable PyTorch neural network model.

    Args:
        input_size (int): Size of the input features.
        output_size (int): Number of output classes or regression targets.
        hidden_layers (list[int], optional): List of units in each hidden layer. Defaults to None.
        activation_fn (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
        dropout_rate (float, optional): Dropout rate to apply between layers. Defaults to None.
    """

    def __init__(self, input_size, output_size, hidden_layers=None, activation_fn=nn.ReLU, dropout_rate=None):
        super(LinearModel, self).__init__()

        # Build the layers
        self.layers = nn.ModuleList()
        prev_size = input_size

        # Add hidden layers
        if hidden_layers:
            for hidden_size in hidden_layers:
                self.layers.append(nn.Linear(prev_size, hidden_size))
                self.layers.append(activation_fn())
                if dropout_rate is not None:
                    self.layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size

        # Add output layer
        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x






