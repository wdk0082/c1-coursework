# 5D to 1D Regression API

A FastAPI-based backend for training and deploying neural network models for 5-dimensional to 1-dimensional regression tasks.

## Overview

This API provides endpoints for:
- Uploading datasets in `.npz` format
- Training neural network models with configurable architectures
- Making predictions on new 5D input vectors
- Managing datasets and trained models

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install the package with dependencies:
```bash
pip install -e .
```

This will install:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch for neural networks
- `numpy` - Numerical computing
- `python-multipart` - File upload support

## Running the API

Start the server:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

**To stop the server:**
Press `Ctrl+C` (or `Cmd+C` on macOS) in the terminal to gracefully shut down the server.

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

**GET** `/health`

Check service status and available resources.

**Response:**
```json
{
  "status": "healthy",
  "service": "5D Regression API",
  "version": "1.0.0",
  "device": "cpu",
  "datasets_loaded": 0,
  "models_trained": 0
}
```

### 2. Upload Dataset

**POST** `/upload`

Upload a dataset in `.npz` format.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload with key `file`

**Dataset Format:**
The `.npz` file should contain:
- `X`: NumPy array of shape `(n_samples, 5)` - 5D input features
- `y`: NumPy array of shape `(n_samples,)` - 1D output targets

Alternative key names supported:
- `inputs`/`outputs`
- `features`/`targets`

**Example:**
```python
import numpy as np

# Create dataset
X = np.random.randn(1000, 5)
y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2]

# Save as .npz
np.savez('my_dataset.npz', X=X, y=y)
```

```bash
curl -X POST -F "file=@my_dataset.npz" http://localhost:8000/upload
```

**Response:**
```json
{
  "dataset_id": "dataset_20231204_123456",
  "filename": "my_dataset.npz",
  "num_samples": 1000,
  "X_shape": [1000, 5],
  "y_shape": [1000],
  "uploaded_at": "20231204_123456"
}
```

### 3. Train Model

**POST** `/train`

Train a neural network model on an uploaded dataset.

**Request Body:**
```json
{
  "dataset_id": "dataset_20231204_123456",
  "architecture": {
    "hidden_dims": [64, 32],
    "dropout": 0.0,
    "activation": "relu"
  },
  "training_params": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "weight_decay": 0.0
  },
  "split_ratios": [0.7, 0.15, 0.15],
  "standardize": true,
  "missing_strategy": "ignore"
}
```

**Parameters:**
- `dataset_id` (required): ID of uploaded dataset
- `architecture` (optional):
  - `hidden_dims`: List of hidden layer sizes (default: `[64, 32]`)
  - `dropout`: Dropout probability (default: `0.0`)
  - `activation`: Activation function - `"relu"`, `"tanh"`, or `"sigmoid"` (default: `"relu"`)
- `training_params` (optional):
  - `learning_rate`: Learning rate (default: `0.001`)
  - `batch_size`: Batch size (default: `32`)
  - `epochs`: Maximum number of epochs (default: `100`)
  - `early_stopping_patience`: Early stopping patience (default: `10`)
  - `weight_decay`: L2 regularization (default: `0.0`)
- `split_ratios` (optional): `[train, val, test]` ratios (default: `[0.7, 0.15, 0.15]`)
- `standardize` (optional): Standardize features (default: `true`)
- `missing_strategy` (optional): Strategy for handling missing values (default: `"ignore"`):
  - `"ignore"`: Remove rows with any missing values
  - `"mean"`: Fill missing values with column mean
  - `"median"`: Fill missing values with column median
  - `"zero"`: Fill missing values with zeros
  - `"forward_fill"`: Forward fill missing values

**Response:**
```json
{
  "model_id": "model_20231204_123500",
  "best_epoch": 45,
  "best_val_loss": 0.0234,
  "final_train_loss": 0.0189,
  "test_metrics": {
    "mse": 0.0245,
    "rmse": 0.1565,
    "mae": 0.1123
  },
  "training_time": "20231204_123500"
}
```

### 4. Make Predictions

**POST** `/predict`

Make predictions using a trained model.

**Request Body:**
```json
{
  "model_id": "model_20231204_123500",
  "inputs": [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [0.5, 1.5, 2.5, 3.5, 4.5]
  ]
}
```

**Parameters:**
- `model_id` (required): ID of trained model
- `inputs` (required): List of 5D input vectors

**Response:**
```json
{
  "model_id": "model_20231204_123500",
  "predictions": [7.234, 3.456],
  "num_predictions": 2
}
```

### 5. List Datasets

**GET** `/datasets`

List all uploaded datasets.

**Response:**
```json
{
  "datasets": [
    {
      "dataset_id": "dataset_20231204_123456",
      "filename": "my_dataset.npz",
      "num_samples": 1000,
      "uploaded_at": "20231204_123456"
    }
  ],
  "total": 1
}
```

### 6. List Models

**GET** `/models`

List all trained models.

**Response:**
```json
{
  "models": [
    {
      "model_id": "model_20231204_123500",
      "dataset_id": "dataset_20231204_123456",
      "best_epoch": 45,
      "best_val_loss": 0.0234,
      "test_metrics": {
        "mse": 0.0245,
        "rmse": 0.1565,
        "mae": 0.1123
      },
      "trained_at": "20231204_123500"
    }
  ],
  "total": 1
}
```

### 7. Get Model Details

**GET** `/models/{model_id}`

Get detailed information about a specific model.

**Response:**
```json
{
  "model_id": "model_20231204_123500",
  "dataset_id": "dataset_20231204_123456",
  "architecture": {
    "hidden_dims": [64, 32],
    "dropout": 0.0,
    "activation": "relu"
  },
  "training_params": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
  },
  "split_ratios": [0.7, 0.15, 0.15],
  "standardize": true,
  "missing_strategy": "ignore",
  "best_epoch": 45,
  "best_val_loss": 0.0234,
  "test_metrics": {
    "mse": 0.0245,
    "rmse": 0.1565,
    "mae": 0.1123
  },
  "trained_at": "20231204_123500"
}
```

## Example Workflow

### Using Python Requests

```python
import requests
import numpy as np

API_URL = "http://localhost:8000"

# 1. Create and upload dataset
X = np.random.randn(1000, 5)
y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2]
np.savez('dataset.npz', X=X, y=y)

with open('dataset.npz', 'rb') as f:
    response = requests.post(f"{API_URL}/upload", files={"file": f})
dataset_id = response.json()["dataset_id"]
print(f"Uploaded dataset: {dataset_id}")

# 2. Train model
train_config = {
    "dataset_id": dataset_id,
    "architecture": {
        "hidden_dims": [64, 32],
        "dropout": 0.1,
        "activation": "relu"
    },
    "training_params": {
        "learning_rate": 0.001,
        "epochs": 50
    },
    "missing_strategy": "mean"  # Handle missing values with column mean
}

response = requests.post(f"{API_URL}/train", json=train_config)
model_id = response.json()["model_id"]
print(f"Trained model: {model_id}")
print(f"Test RMSE: {response.json()['test_metrics']['rmse']:.4f}")

# 3. Make predictions
predict_data = {
    "model_id": model_id,
    "inputs": [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ]
}

response = requests.post(f"{API_URL}/predict", json=predict_data)
predictions = response.json()["predictions"]
print(f"Predictions: {predictions}")
```

### Using cURL

```bash
# 1. Upload dataset
curl -X POST -F "file=@dataset.npz" http://localhost:8000/upload

# 2. Train model
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset_20231204_123456",
    "architecture": {"hidden_dims": [64, 32]},
    "training_params": {"epochs": 50},
    "missing_strategy": "mean"
  }'

# 3. Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_20231204_123500",
    "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]
  }'
```

## Project Structure

```
backend/
├── api.py                      # FastAPI application
├── pyproject.toml             # Project configuration
├── README.md                  # This file
├── fivedreg/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py       # Data loading utilities
│   ├── model/
│   │   ├── __init__.py
│   │   └── naive_nn.py        # Neural network models
│   └── trainer/
│       ├── __init__.py
│       └── nn_trainer.py      # Training utilities
└── data/
    ├── uploads/               # Uploaded datasets (created at runtime)
    └── models/                # Trained models (created at runtime)
```

## Data Storage

- **Uploaded datasets**: `./data/uploads/`
- **Trained models**: `./data/models/`
- **Model metadata**: `./data/models/{model_id}_metadata.json`

Models and datasets are kept in memory for fast access during the API session.

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid data format, missing required fields)
- `404`: Resource not found (dataset or model doesn't exist)
- `500`: Internal server error (training failed, prediction error)

Error responses include a `detail` field with information about the error:

```json
{
  "detail": "Dataset 'dataset_invalid' not found"
}
```

## GPU Support

The API automatically detects and uses CUDA if available. Check the `/health` endpoint to see which device is being used:

```json
{
  "device": "cuda"  // or "cpu"
}
```

## Development

### Running Tests

You can test individual components:

```python
# Test data loading
from fivedreg.data.load_data import load_data
data = load_data("dataset.npz", split_ratios=(0.7, 0.15, 0.15))

# Test model
from fivedreg.model.naive_nn import NaiveMLP
model = NaiveMLP(hidden_dims=[64, 32])

# Test trainer
from fivedreg.trainer.nn_trainer import NNTrainer
trainer = NNTrainer(NaiveMLP, {"hidden_dims": [64, 32]})
```

## Notes

- This is a course project focused on simplicity and clarity
- The API stores data in memory, which is cleared when the server restarts
- For production use, consider adding persistent storage, authentication, and rate limiting
- Models are saved to disk and can be loaded between sessions if needed

## License

Academic project for Cambridge C1 coursework.
