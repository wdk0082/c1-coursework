"""Tests for the NaiveMLP model."""

import numpy as np
import pytest
import torch

from fivedreg.model.naive_nn import NaiveMLP, create_model


class TestNaiveMLP:
    """Tests for NaiveMLP model initialization and structure."""

    def test_default_initialization(self):
        """Test model initializes with default parameters."""
        model = NaiveMLP()
        assert model.hidden_dims == [64, 32]
        assert model.dropout == 0.0
        assert isinstance(model.activation, torch.nn.ReLU)

    def test_custom_hidden_dims(self):
        """Test model with custom hidden dimensions."""
        model = NaiveMLP(hidden_dims=[128, 64, 32])
        assert model.hidden_dims == [128, 64, 32]
        assert model.get_num_parameters() > 0

    def test_activation_relu(self):
        """Test ReLU activation."""
        model = NaiveMLP(activation='relu')
        assert isinstance(model.activation, torch.nn.ReLU)

    def test_activation_tanh(self):
        """Test Tanh activation."""
        model = NaiveMLP(activation='tanh')
        assert isinstance(model.activation, torch.nn.Tanh)

    def test_activation_sigmoid(self):
        """Test Sigmoid activation."""
        model = NaiveMLP(activation='sigmoid')
        assert isinstance(model.activation, torch.nn.Sigmoid)

    def test_invalid_activation(self):
        """Test that invalid activation raises an error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            NaiveMLP(activation='invalid')

    def test_dropout_applied(self):
        """Test that dropout layers are added when dropout > 0."""
        model = NaiveMLP(hidden_dims=[32], dropout=0.5)
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        assert has_dropout

    def test_no_dropout_when_zero(self):
        """Test that no dropout layers exist when dropout = 0."""
        model = NaiveMLP(hidden_dims=[32], dropout=0.0)
        dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(dropout_layers) == 0


class TestForwardPass:
    """Tests for the forward pass."""

    def test_forward_shape(self):
        """Test output shape from forward pass."""
        model = NaiveMLP(hidden_dims=[32, 16])
        x = torch.randn(10, 5)
        output = model(x)
        assert output.shape == (10, 1)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        model = NaiveMLP()
        x = torch.randn(1, 5)
        output = model(x)
        assert output.shape == (1, 1)

    def test_forward_large_batch(self):
        """Test forward pass with large batch."""
        model = NaiveMLP()
        x = torch.randn(1000, 5)
        output = model(x)
        assert output.shape == (1000, 1)

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic in eval mode."""
        model = NaiveMLP(dropout=0.5)
        model.eval()
        x = torch.randn(10, 5)
        output1 = model(x)
        output2 = model(x)
        torch.testing.assert_close(output1, output2)


class TestPredict:
    """Tests for the predict method."""

    def test_predict_shape(self):
        """Test output shape from predict."""
        model = NaiveMLP()
        x = torch.randn(10, 5)
        output = model.predict(x)
        assert output.shape == (10,)  # Squeezed

    def test_predict_single_sample(self):
        """Test predict with single sample."""
        model = NaiveMLP()
        x = torch.randn(1, 5)
        output = model.predict(x)
        # Single sample should still return a scalar or 1D tensor
        assert output.dim() == 0 or output.shape == (1,)

    def test_predict_no_grad(self):
        """Test that predict disables gradients."""
        model = NaiveMLP()
        x = torch.randn(10, 5, requires_grad=True)
        output = model.predict(x)
        # Output should not require grad (detached)
        assert not output.requires_grad


class TestModelProperties:
    """Tests for model properties and utilities."""

    def test_get_num_parameters(self):
        """Test parameter counting."""
        model = NaiveMLP(hidden_dims=[32])
        n_params = model.get_num_parameters()
        # Input(5) -> Hidden(32): 5*32 + 32 = 192
        # Hidden(32) -> Output(1): 32*1 + 1 = 33
        # Total: 225
        assert n_params == 225

    def test_get_num_parameters_larger(self):
        """Test parameter counting for larger model."""
        model = NaiveMLP(hidden_dims=[64, 32])
        n_params = model.get_num_parameters()
        # 5->64: 5*64+64 = 384
        # 64->32: 64*32+32 = 2080
        # 32->1: 32*1+1 = 33
        # Total: 2497
        assert n_params == 2497

    def test_repr(self):
        """Test string representation."""
        model = NaiveMLP(hidden_dims=[32], dropout=0.1)
        repr_str = repr(model)
        assert 'NaiveMLP' in repr_str
        assert 'hidden_dims=[32]' in repr_str
        assert 'dropout=0.1' in repr_str


class TestCreateModel:
    """Tests for the create_model factory function."""

    def test_create_with_full_config(self):
        """Test creating model with full config."""
        config = {
            'hidden_dims': [128, 64],
            'dropout': 0.2,
            'activation': 'tanh'
        }
        model = create_model(config)
        assert model.hidden_dims == [128, 64]
        assert model.dropout == 0.2
        assert isinstance(model.activation, torch.nn.Tanh)

    def test_create_with_minimal_config(self):
        """Test creating model with minimal config (uses defaults)."""
        config = {}
        model = create_model(config)
        assert model.hidden_dims == [64, 32]
        assert model.dropout == 0.0
        assert isinstance(model.activation, torch.nn.ReLU)

    def test_create_with_partial_config(self):
        """Test creating model with partial config."""
        config = {'hidden_dims': [256]}
        model = create_model(config)
        assert model.hidden_dims == [256]
        assert model.dropout == 0.0  # Default


class TestGradientFlow:
    """Tests for gradient computation."""

    def test_gradients_flow(self):
        """Test that gradients flow through the network."""
        model = NaiveMLP(hidden_dims=[32])
        x = torch.randn(10, 5)
        y = torch.randn(10, 1)

        output = model(x)
        loss = torch.nn.MSELoss()(output, y)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)

    def test_training_step(self):
        """Test a single training step updates weights."""
        model = NaiveMLP(hidden_dims=[32])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        x = torch.randn(10, 5)
        y = torch.randn(10, 1)

        # Get initial weights
        initial_weights = model.network[0].weight.clone()

        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.equal(initial_weights, model.network[0].weight)
