"""Shared pytest fixtures for the test suite."""

import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path


@pytest.fixture
def sample_data():
    """Generate sample 5D->1D regression data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] * 0.5 - X[:, 2] + np.random.randn(n_samples) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture
def sample_data_with_nan():
    """Generate sample data with some NaN values."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] * 0.5).astype(np.float32)
    X[5, 0] = np.nan
    X[10, 2] = np.nan
    X[15, 4] = np.nan
    return X, y


@pytest.fixture
def temp_npz_file(sample_data):
    """Create a temporary .npz file with sample data."""
    X, y = sample_data
    f = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    np.savez(f.name, X=X, y=y)
    f.close()
    yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_npz_alt_keys(sample_data):
    """Create a temporary .npz file with alternative key names."""
    X, y = sample_data
    f = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    np.savez(f.name, inputs=X, outputs=y)
    f.close()
    yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_pkl_file(sample_data):
    """Create a temporary .pkl file with sample data."""
    X, y = sample_data
    f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    pickle.dump({'X': X, 'y': y}, f)
    f.close()
    yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def model_config():
    """Default model configuration."""
    return {
        'hidden_dims': [32, 16],
        'dropout': 0.1,
        'activation': 'relu'
    }


@pytest.fixture
def training_config(tmp_path):
    """Default training configuration for fast tests."""
    return {
        'learning_rate': 1e-3,
        'batch_size': 16,
        'epochs': 5,
        'early_stopping_patience': 0,
        'weight_decay': 0.0,
        'checkpoint_dir': str(tmp_path)
    }
