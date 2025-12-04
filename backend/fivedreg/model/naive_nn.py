"""Naive MLP models for 5D to 1D regression."""

from typing import List
import torch
import torch.nn as nn


class NaiveMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for regression.

    Architecture:
        - Input layer: 5 features
        - Hidden layers: Configurable number and sizes
        - Output layer: 1 value
        - Activation: ReLU
        - Optional dropout for regularization
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        """
        Initialize the MLP model.

        Args:
            hidden_dims: List of hidden layer dimensions. Default: [64, 32]
            dropout: Dropout probability (0 = no dropout). Default: 0.0
            activation: Activation function ('relu', 'tanh', 'sigmoid'). Default: 'relu'
        """
        super().__init__()

        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        input_dim = 5  # 5D input

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU, Xavier for tanh/sigmoid
                if isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 5)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (inference mode).

        Args:
            x: Input tensor of shape (batch_size, 5)

        Returns:
            Output tensor of shape (batch_size,) - squeezed
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output.squeeze()

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the model."""
        n_params = self.get_num_parameters()
        return (
            f"NaiveMLP(hidden_dims={self.hidden_dims}, "
            f"dropout={self.dropout}, "
            f"n_params={n_params:,})"
        )


def create_model(config: dict) -> NaiveMLP:
    """
    Factory function to create a model from a config dictionary.

    Args:
        config: Dictionary with model configuration. Expected keys:
            - "hidden_dims": List[int] - Hidden layer dimensions
            - "dropout": float - Dropout probability (optional, default: 0.0)
            - "activation": str - Activation function (optional, default: "relu")

    Returns:
        Initialized NaiveMLP model

    Example:
        config = {
            "hidden_dims": [128, 64, 32],
            "dropout": 0.1,
            "activation": "relu"
        }
        model = create_model(config)
    """
    hidden_dims = config.get("hidden_dims", [64, 32])
    dropout = config.get("dropout", 0.0)
    activation = config.get("activation", "relu")

    model = NaiveMLP(
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation
    )

    print(f"Created model: {model}")
    return model
