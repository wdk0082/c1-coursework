from typing import List
import torch
import torch.nn as nn


class NaiveMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for regression.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        """
        Initialize the MLP model.

        Parameters
        ----------
        hidden_dims : list of int, default=[64, 32]
            List of hidden layer dimensions.
        dropout : float, default=0.0
            Dropout probability (0 = no dropout).
        activation : {'relu', 'tanh', 'sigmoid'}, default='relu'
            Activation function to use.
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
        """
        Initialize network weights using Xavier/He initialization.

        Uses He initialization for ReLU activation and Xavier initialization
        for tanh/sigmoid activations.
        """
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

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 5).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions in inference mode.

        Sets the model to evaluation mode and disables gradient computation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 5).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size,), squeezed from (batch_size, 1).
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output.squeeze()

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns
        -------
        int
            Total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """
        Return string representation of the model.

        Returns
        -------
        str
            String showing model configuration and parameter count.
        """
        n_params = self.get_num_parameters()
        return (
            f"NaiveMLP(hidden_dims={self.hidden_dims}, "
            f"dropout={self.dropout}, "
            f"n_params={n_params:,})"
        )


def create_model(config: dict) -> NaiveMLP:
    """
    Factory function to create a model from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Dictionary with model configuration. Supported keys:

        - **hidden_dims** : list of int - Hidden layer dimensions (required)
        - **dropout** : float - Dropout probability (optional, default: 0.0)
        - **activation** : str - Activation function (optional, default: 'relu')

    Returns
    -------
    NaiveMLP
        Initialized NaiveMLP model instance.

    Examples
    --------
    >>> config = {
    ...     "hidden_dims": [128, 64, 32],
    ...     "dropout": 0.1,
    ...     "activation": "relu"
    ... }
    >>> model = create_model(config)
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
