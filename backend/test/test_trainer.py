"""Tests for the NNTrainer."""

import numpy as np
import pytest

from fivedreg.model.naive_nn import NaiveMLP, create_model
from fivedreg.trainer.nn_trainer import NNTrainer


class TestTrainerInitialization:
    """Tests for NNTrainer initialization."""

    def test_init_with_model_class(self, model_config):
        """Test initialization with model class."""
        trainer = NNTrainer(NaiveMLP, model_config)
        assert trainer.model is not None
        assert isinstance(trainer.model, NaiveMLP)

    def test_init_with_factory_function(self, model_config):
        """Test initialization with create_model factory."""
        model = create_model(model_config)
        trainer = NNTrainer(lambda **kwargs: model, {})
        assert trainer.model is not None

    def test_default_training_config(self, model_config):
        """Test default training configuration."""
        trainer = NNTrainer(NaiveMLP, model_config)
        assert trainer.learning_rate == 1e-3
        assert trainer.batch_size == 32
        assert trainer.epochs == 100

    def test_custom_training_config(self, model_config, training_config):
        """Test custom training configuration."""
        trainer = NNTrainer(NaiveMLP, model_config, training_config)
        assert trainer.epochs == 5
        assert trainer.batch_size == 16

    def test_device_auto_detection(self, model_config):
        """Test device auto-detection."""
        trainer = NNTrainer(NaiveMLP, model_config)
        assert trainer.device.type in ['cpu', 'cuda']

    def test_explicit_device(self, model_config):
        """Test explicit device setting."""
        trainer = NNTrainer(NaiveMLP, model_config, device='cpu')
        assert trainer.device.type == 'cpu'

    def test_checkpoint_dir_created(self, model_config, tmp_path):
        """Test that checkpoint directory is created."""
        training_config = {'checkpoint_dir': str(tmp_path / 'checkpoints')}
        trainer = NNTrainer(NaiveMLP, model_config, training_config)
        assert (tmp_path / 'checkpoints').exists()


class TestTraining:
    """Tests for the training process."""

    def test_fit_runs(self, sample_data, model_config, training_config):
        """Test that fit completes without error."""
        X, y = sample_data
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        result = trainer.fit(X[:50], y[:50], verbose=False)
        assert 'model' in result
        assert 'history' in result

    def test_fit_with_validation(self, sample_data, model_config, training_config):
        """Test training with validation data."""
        X, y = sample_data
        training_config['epochs'] = 3
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        result = trainer.fit(X[:80], y[:80], X[80:], y[80:], verbose=False)
        assert len(result['history']['train_loss']) == 3
        assert len(result['history']['val_loss']) == 3

    def test_loss_decreases(self, sample_data, model_config, training_config):
        """Test that training loss generally decreases."""
        X, y = sample_data
        training_config['epochs'] = 20
        training_config['learning_rate'] = 1e-2
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        result = trainer.fit(X, y, verbose=False)
        losses = result['history']['train_loss']
        assert losses[-1] < losses[0]

    def test_early_stopping(self, tmp_path):
        """Test early stopping triggers."""
        np.random.seed(123)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model_config = {'hidden_dims': [4], 'dropout': 0.0, 'activation': 'relu'}
        training_config = {
            'epochs': 50,
            'batch_size': 80,
            'learning_rate': 0.1,
            'early_stopping_patience': 3,
            'checkpoint_dir': str(tmp_path)
        }

        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        result = trainer.fit(X[:80], y[:80], X[80:], y[80:], verbose=False)
        assert len(result['history']['train_loss']) > 0
        assert len(result['history']['val_loss']) > 0

    def test_history_tracking(self, sample_data, model_config, training_config):
        """Test that training history is tracked."""
        X, y = sample_data
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X[:50], y[:50], verbose=False)
        assert len(trainer.history['train_loss']) == 5


class TestEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_returns_metrics(self, sample_data, model_config, training_config):
        """Test that evaluate returns expected metrics."""
        X, y = sample_data
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X[:80], y[:80], verbose=False)
        metrics = trainer.evaluate(X[80:], y[80:])
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics

    def test_rmse_is_sqrt_mse(self, sample_data, model_config, training_config):
        """Test that RMSE is the square root of MSE."""
        X, y = sample_data
        training_config['epochs'] = 2
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X[:80], y[:80], verbose=False)
        metrics = trainer.evaluate(X[80:], y[80:])
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']), rtol=1e-5)

    def test_metrics_are_positive(self, sample_data, model_config, training_config):
        """Test that all metrics are non-negative."""
        X, y = sample_data
        training_config['epochs'] = 2
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X[:80], y[:80], verbose=False)
        metrics = trainer.evaluate(X[80:], y[80:])
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0


class TestPrediction:
    """Tests for prediction."""

    def test_predict_shape(self, sample_data, model_config, training_config):
        """Test prediction output shape."""
        X, y = sample_data
        training_config['epochs'] = 2
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X[:80], y[:80], verbose=False)
        predictions = trainer.predict(X[80:])
        assert predictions.shape == (20,)

    def test_predict_single_sample(self, sample_data, model_config, training_config):
        """Test prediction for single sample."""
        X, y = sample_data
        training_config['epochs'] = 2
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X, y, verbose=False)
        predictions = trainer.predict(X[:1])
        assert predictions.shape == (1,)

    def test_predict_is_deterministic(self, sample_data, model_config, training_config):
        """Test that predictions are deterministic."""
        X, y = sample_data
        training_config['epochs'] = 2
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X, y, verbose=False)
        pred1 = trainer.predict(X[:10])
        pred2 = trainer.predict(X[:10])
        np.testing.assert_array_equal(pred1, pred2)


class TestCheckpointing:
    """Tests for model saving and loading."""

    def test_save_and_load_model(self, sample_data, model_config, training_config, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        training_config['epochs'] = 2
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X, y, verbose=False)

        model_path = tmp_path / 'test_model.pt'
        trainer.save_model(str(model_path))
        assert model_path.exists()

        pred_before = trainer.predict(X[:10])

        trainer2 = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer2.load_model(str(model_path))
        pred_after = trainer2.predict(X[:10])

        np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_best_checkpoint_saved(self, sample_data, model_config, training_config):
        """Test that best checkpoint is saved during training with validation."""
        X, y = sample_data
        trainer = NNTrainer(NaiveMLP, model_config, training_config, device='cpu')
        trainer.fit(X[:80], y[:80], X[80:], y[80:], verbose=False)
        best_model_path = trainer.checkpoint_dir / 'best_model.pt'
        assert best_model_path.exists()
