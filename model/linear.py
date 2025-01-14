from typing import Optional, List

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    A customizable PyTorch neural network model for regression or classification tasks.

    This class allows the construction of a fully connected feedforward neural network with customizable
    input size, output size, hidden layers, activation functions, and optional dropout.

    Args:
        input_size (int): Size of the input features.
        output_size (int): Number of output classes (for classification) or regression targets.
        hidden_layers (Optional[List[int]], optional): List of integers representing the number of units
                                                      in each hidden layer. Defaults to None (no hidden layers).
        activation_fn (nn.Module, optional): The activation function to use between layers. Defaults to nn.ReLU.
        dropout_rate (Optional[float], optional): Dropout rate to apply between layers for regularization.
                                                  Defaults to None (no dropout).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: Optional[List[int]] = None,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None,
    ):
        super(LinearModel, self).__init__()

        # Initialize the layer list
        self.layers = nn.ModuleList()
        prev_size = input_size

        # Add hidden layers
        if hidden_layers:
            for hidden_size in hidden_layers:
                self.layers.append(nn.Linear(prev_size, hidden_size))  # Linear layer
                self.layers.append(activation_fn())  # Activation function
                if dropout_rate is not None:
                    self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer
                prev_size = hidden_size

        # Add output layer
        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model. Passes the input tensor through the layers of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)  # Apply each layer sequentially
        return x
