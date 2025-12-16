FastAPI Endpoints
=================

The backend provides a REST API for dataset management, model training, and inference. Interactive documentation is available at http://localhost:8000/docs when the server is running.

Base URL: ``http://localhost:8000``

Endpoints Overview
------------------

+----------------+--------+-------------------------------------------+
| Endpoint       | Method | Description                               |
+================+========+===========================================+
| /health        | GET    | Health check and system status            |
+----------------+--------+-------------------------------------------+
| /upload        | POST   | Upload a dataset file                     |
+----------------+--------+-------------------------------------------+
| /datasets      | GET    | List all uploaded datasets                |
+----------------+--------+-------------------------------------------+
| /train         | POST   | Train a model on a dataset                |
+----------------+--------+-------------------------------------------+
| /models        | GET    | List all trained models                   |
+----------------+--------+-------------------------------------------+
| /models/{id}   | GET    | Get details of a specific model           |
+----------------+--------+-------------------------------------------+
| /predict       | POST   | Make predictions with a trained model     |
+----------------+--------+-------------------------------------------+

Health Check
------------

**GET /health**

.. code-block:: bash

   curl http://localhost:8000/health

Response:

.. code-block:: json

   {
     "status": "healthy",
     "service": "5D Regression API",
     "version": "1.0.0",
     "device": "cpu",
     "datasets_loaded": 2,
     "models_trained": 1
   }

Upload Dataset
--------------

**POST /upload**

Upload a ``.npz`` or ``.pkl`` file containing ``X`` (shape: n_samples x 5) and ``y`` (shape: n_samples).

.. code-block:: bash

   curl -X POST http://localhost:8000/upload \
     -F "file=@dataset.pkl"

Response:

.. code-block:: json

   {
     "dataset_id": "dataset_20231215_143022",
     "filename": "dataset.pkl",
     "num_samples": 1000,
     "X_shape": [1000, 5],
     "y_shape": [1000],
     "uploaded_at": "20231215_143022"
   }

List Datasets
-------------

**GET /datasets**

.. code-block:: bash

   curl http://localhost:8000/datasets

Response:

.. code-block:: json

   {
     "datasets": [
       {
         "dataset_id": "dataset_20231215_143022",
         "filename": "dataset.pkl",
         "num_samples": 1000,
         "X_shape": [1000, 5],
         "y_shape": [1000],
         "uploaded_at": "20231215_143022"
       }
     ],
     "total": 1
   }

Train Model
-----------

**POST /train**

.. code-block:: bash

   curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "dataset_20231215_143022",
       "architecture": {
         "hidden_dims": [64, 32],
         "dropout": 0.1,
         "activation": "relu"
       },
       "training_params": {
         "learning_rate": 0.001,
         "batch_size": 32,
         "epochs": 100
       },
       "split_ratios": [0.7, 0.15, 0.15],
       "standardize": true,
       "missing_strategy": "ignore"
     }'

**Request Parameters:**

- ``dataset_id`` (required): ID of uploaded dataset
- ``architecture``: Model configuration

  - ``hidden_dims``: Hidden layer sizes (default: [64, 32])
  - ``dropout``: Dropout rate (default: 0.0)
  - ``activation``: "relu", "tanh", or "sigmoid" (default: "relu")

- ``training_params``: Training hyperparameters

  - ``learning_rate``: Learning rate (default: 0.001)
  - ``batch_size``: Batch size (default: 32)
  - ``epochs``: Number of epochs (default: 100)
  - ``weight_decay``: L2 regularization (default: 0.0)

- ``split_ratios``: Train/val/test split (default: [0.7, 0.15, 0.15])
- ``standardize``: Standardize features (default: true)
- ``missing_strategy``: "ignore", "mean", "median", "zero", or "forward_fill"

Response:

.. code-block:: json

   {
     "model_id": "model_20231215_143500",
     "best_epoch": 42,
     "best_val_loss": 0.0234,
     "final_train_loss": 0.0198,
     "test_metrics": {
       "mse": 0.0256,
       "rmse": 0.1600,
       "mae": 0.1234,
       "r2": 0.9456
     },
     "training_time_seconds": 12.5,
     "training_memory_mb": 45.2
   }

List Models
-----------

**GET /models**

.. code-block:: bash

   curl http://localhost:8000/models

Response:

.. code-block:: json

   {
     "models": [
       {
         "model_id": "model_20231215_143500",
         "dataset_id": "dataset_20231215_143022",
         "best_epoch": 42,
         "best_val_loss": 0.0234,
         "test_metrics": {
           "mse": 0.0256,
           "rmse": 0.1600,
           "mae": 0.1234,
           "r2": 0.9456
         },
         "trained_at": "20231215_143500"
       }
     ],
     "total": 1
   }

Get Model Details
-----------------

**GET /models/{model_id}**

.. code-block:: bash

   curl http://localhost:8000/models/model_20231215_143500

Response:

.. code-block:: json

   {
     "model_id": "model_20231215_143500",
     "dataset_id": "dataset_20231215_143022",
     "architecture": {
       "hidden_dims": [64, 32],
       "dropout": 0.1,
       "activation": "relu"
     },
     "training_params": {
       "learning_rate": 0.001,
       "batch_size": 32,
       "epochs": 100
     },
     "best_epoch": 42,
     "best_val_loss": 0.0234,
     "test_metrics": {
       "mse": 0.0256,
       "rmse": 0.1600,
       "mae": 0.1234,
       "r2": 0.9456
     },
     "training_time_seconds": 12.5,
     "training_memory_mb": 45.2,
     "trained_at": "20231215_143500"
   }

Make Predictions
----------------

**POST /predict**

.. code-block:: bash

   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "model_20231215_143500",
       "inputs": [
         [1.0, 2.0, 3.0, 4.0, 5.0],
         [0.5, 0.5, 0.5, 0.5, 0.5]
       ]
     }'

Response:

.. code-block:: json

   {
     "model_id": "model_20231215_143500",
     "predictions": [2.3456, 1.2345],
     "num_predictions": 2,
     "inference_memory_mb": 5.2
   }

Python Example
--------------

.. code-block:: python

   import requests

   BASE_URL = "http://localhost:8000"

   # Upload dataset
   with open("dataset.pkl", "rb") as f:
       response = requests.post(f"{BASE_URL}/upload", files={"file": f})
   dataset_id = response.json()["dataset_id"]

   # Train model
   response = requests.post(f"{BASE_URL}/train", json={
       "dataset_id": dataset_id,
       "architecture": {"hidden_dims": [64, 32]},
       "training_params": {"epochs": 50}
   })
   model_id = response.json()["model_id"]

   # Make predictions
   response = requests.post(f"{BASE_URL}/predict", json={
       "model_id": model_id,
       "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]
   })
   predictions = response.json()["predictions"]

Error Responses
---------------

All endpoints return errors in this format:

.. code-block:: json

   {"detail": "Error message describing what went wrong"}

Common status codes: 400 (bad request), 404 (not found), 500 (server error).
