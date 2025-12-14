"""Tests for the FastAPI endpoints."""

import io
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app, datasets, models


@pytest.fixture
def client():
    """Get a fresh test client with cleared storage."""
    datasets.clear()
    models.clear()
    return TestClient(app)


@pytest.fixture
def npz_bytes(sample_data):
    """Create NPZ file bytes from sample data."""
    X, y = sample_data
    buffer = io.BytesIO()
    np.savez(buffer, X=X, y=y)
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def uploaded_dataset(client, npz_bytes):
    """Upload a dataset and return the dataset_id."""
    response = client.post(
        "/upload",
        files={"file": ("test.npz", npz_bytes, "application/octet-stream")}
    )
    return response.json()["dataset_id"]


@pytest.fixture
def trained_model(client, uploaded_dataset):
    """Train a model and return the model_id."""
    response = client.post(
        "/train",
        json={
            "dataset_id": uploaded_dataset,
            "training_params": {"epochs": 2, "early_stopping_patience": 0}
        }
    )
    return response.json()["model_id"]


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "5D Regression API"
        assert "device" in data

    def test_health_shows_counts(self, client):
        """Test health endpoint shows dataset and model counts."""
        response = client.get("/health")
        data = response.json()
        assert data["datasets_loaded"] == 0
        assert data["models_trained"] == 0


class TestUploadEndpoint:
    """Tests for /upload endpoint."""

    def test_upload_valid_npz(self, client, npz_bytes):
        """Test uploading a valid NPZ file."""
        response = client.post(
            "/upload",
            files={"file": ("test.npz", npz_bytes, "application/octet-stream")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert data["num_samples"] == 100
        assert data["X_shape"] == [100, 5]

    def test_upload_wrong_extension(self, client):
        """Test that non-NPZ files are rejected."""
        response = client.post(
            "/upload",
            files={"file": ("test.csv", b"x,y\n1,2", "text/csv")}
        )
        assert response.status_code == 400
        assert "npz" in response.json()["detail"].lower()

    def test_upload_wrong_feature_dimensions(self, client):
        """Test that wrong feature dimensions are rejected."""
        X = np.random.randn(50, 3).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        buffer = io.BytesIO()
        np.savez(buffer, X=X, y=y)
        buffer.seek(0)

        response = client.post(
            "/upload",
            files={"file": ("test.npz", buffer.read(), "application/octet-stream")}
        )
        assert response.status_code == 400

    def test_upload_creates_dataset_entry(self, client, npz_bytes):
        """Test that upload creates a dataset entry."""
        response = client.post(
            "/upload",
            files={"file": ("test.npz", npz_bytes, "application/octet-stream")}
        )
        data = response.json()
        assert data["dataset_id"] in datasets


class TestDatasetsEndpoint:
    """Tests for /datasets endpoint."""

    def test_list_empty_datasets(self, client):
        """Test listing datasets when none exist."""
        response = client.get("/datasets")
        assert response.status_code == 200
        data = response.json()
        assert data["datasets"] == []
        assert data["total"] == 0

    def test_list_datasets_after_upload(self, client, uploaded_dataset):
        """Test listing datasets after uploading."""
        response = client.get("/datasets")
        data = response.json()
        assert data["total"] == 1
        assert len(data["datasets"]) == 1


class TestTrainEndpoint:
    """Tests for /train endpoint."""

    def test_train_nonexistent_dataset(self, client):
        """Test training with nonexistent dataset."""
        response = client.post(
            "/train",
            json={"dataset_id": "nonexistent"}
        )
        assert response.status_code == 404

    def test_train_minimal_config(self, client, uploaded_dataset):
        """Test training with minimal configuration."""
        response = client.post(
            "/train",
            json={
                "dataset_id": uploaded_dataset,
                "training_params": {"epochs": 2, "early_stopping_patience": 0}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "best_epoch" in data
        assert "test_metrics" in data
        assert "mse" in data["test_metrics"]

    def test_train_custom_architecture(self, client, uploaded_dataset):
        """Test training with custom architecture."""
        response = client.post(
            "/train",
            json={
                "dataset_id": uploaded_dataset,
                "architecture": {
                    "hidden_dims": [16, 8],
                    "dropout": 0.1,
                    "activation": "tanh"
                },
                "training_params": {"epochs": 2, "early_stopping_patience": 0}
            }
        )
        assert response.status_code == 200

    def test_train_creates_model_entry(self, client, uploaded_dataset):
        """Test that training creates a model entry."""
        response = client.post(
            "/train",
            json={
                "dataset_id": uploaded_dataset,
                "training_params": {"epochs": 2, "early_stopping_patience": 0}
            }
        )
        model_id = response.json()["model_id"]
        assert model_id in models


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_nonexistent_model(self, client):
        """Test prediction with nonexistent model."""
        response = client.post(
            "/predict",
            json={"model_id": "nonexistent", "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]}
        )
        assert response.status_code == 404

    def test_predict_single_sample(self, client, trained_model):
        """Test prediction for single sample."""
        response = client.post(
            "/predict",
            json={"model_id": trained_model, "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_predictions"] == 1
        assert len(data["predictions"]) == 1

    def test_predict_multiple_samples(self, client, trained_model):
        """Test prediction for multiple samples."""
        inputs = [[float(i) for i in range(5)] for _ in range(10)]
        response = client.post(
            "/predict",
            json={"model_id": trained_model, "inputs": inputs}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_predictions"] == 10

    def test_predict_wrong_dimensions(self, client, trained_model):
        """Test that wrong input dimensions are rejected."""
        response = client.post(
            "/predict",
            json={"model_id": trained_model, "inputs": [[1.0, 2.0, 3.0]]}
        )
        assert response.status_code == 500


class TestModelsEndpoint:
    """Tests for /models endpoints."""

    def test_list_empty_models(self, client):
        """Test listing models when none exist."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
        assert data["total"] == 0

    def test_list_models_after_training(self, client, trained_model):
        """Test listing models after training."""
        response = client.get("/models")
        data = response.json()
        assert data["total"] == 1

    def test_get_model_details(self, client, trained_model):
        """Test getting specific model details."""
        response = client.get(f"/models/{trained_model}")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == trained_model
        assert "architecture" in data
        assert "test_metrics" in data

    def test_get_nonexistent_model(self, client):
        """Test getting nonexistent model returns 404."""
        response = client.get("/models/nonexistent")
        assert response.status_code == 404


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, client, sample_data):
        """Test complete upload -> train -> predict workflow."""
        X, y = sample_data
        # Make y a simple function of X for predictable results
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
        buffer = io.BytesIO()
        np.savez(buffer, X=X, y=y)
        buffer.seek(0)

        # Upload
        upload_response = client.post(
            "/upload",
            files={"file": ("test.npz", buffer.read(), "application/octet-stream")}
        )
        assert upload_response.status_code == 200
        dataset_id = upload_response.json()["dataset_id"]

        # Train
        train_response = client.post(
            "/train",
            json={
                "dataset_id": dataset_id,
                "architecture": {"hidden_dims": [32, 16]},
                "training_params": {"epochs": 10, "learning_rate": 0.01, "early_stopping_patience": 5}
            }
        )
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        # Predict
        predict_response = client.post(
            "/predict",
            json={"model_id": model_id, "inputs": [[1.0, 0.5, 0.0, 0.0, 0.0]]}
        )
        assert predict_response.status_code == 200
        assert len(predict_response.json()["predictions"]) == 1

    def test_multiple_models(self, client, uploaded_dataset):
        """Test training multiple models on same dataset."""
        model_ids = []
        for hidden_dims in [[32], [64, 32]]:
            response = client.post(
                "/train",
                json={
                    "dataset_id": uploaded_dataset,
                    "architecture": {"hidden_dims": hidden_dims},
                    "training_params": {"epochs": 2, "early_stopping_patience": 0}
                }
            )
            model_ids.append(response.json()["model_id"])
            time.sleep(1.1)  # Ensure unique timestamp-based IDs

        assert len(set(model_ids)) == 2
