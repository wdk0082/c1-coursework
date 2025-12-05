"""FastAPI backend for 5D to 1D regression model training and inference."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np
import torch
from pathlib import Path
import shutil
import json
from datetime import datetime

from fivedreg.data.load_data import load_data
from fivedreg.model.naive_nn import NaiveMLP
from fivedreg.trainer.nn_trainer import NNTrainer

# Initialize FastAPI app
app = FastAPI(
    title="5D Regression API",
    description="API for training and inference on 5D to 1D regression models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Storage directories
UPLOAD_DIR = Path("./data/uploads")
MODEL_DIR = Path("./data/models")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage for datasets and models
datasets: Dict[str, Dict[str, Any]] = {}
models: Dict[str, Dict[str, Any]] = {}


# ==================== Request/Response Models ====================

class TrainRequest(BaseModel):
    """Request model for training endpoint."""
    dataset_id: str = Field(..., description="ID of uploaded dataset")
    architecture: Dict[str, Any] = Field(
        default_factory=lambda: {
            "hidden_dims": [64, 32],
            "dropout": 0.0,
            "activation": "relu"
        },
        description="Model architecture configuration"
    )
    training_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10,
            "weight_decay": 0.0
        },
        description="Training hyperparameters"
    )
    split_ratios: List[float] = Field(
        default_factory=lambda: [0.7, 0.15, 0.15],
        description="Train/Val/Test split ratios"
    )
    standardize: bool = Field(default=True, description="Standardize features")
    missing_strategy: str = Field(
        default="ignore",
        description="Strategy for handling missing values: 'ignore', 'mean', 'median', 'zero', or 'forward_fill'"
    )


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    model_id: str = Field(..., description="ID of trained model")
    inputs: List[List[float]] = Field(
        ...,
        description="List of 5D input vectors to predict on"
    )


class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    model_id: str
    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    test_metrics: Dict[str, float]
    training_time: str


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    model_id: str
    predictions: List[float]
    num_predictions: int


# ==================== Helper Functions ====================

def get_device():
    """Get the best available device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== API Endpoints ====================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns service status and available resources.
    """
    return {
        "status": "healthy",
        "service": "5D Regression API",
        "version": "1.0.0",
        "device": get_device(),
        "datasets_loaded": len(datasets),
        "models_trained": len(models)
    }


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(..., description="NPZ file containing dataset")
):
    """
    Upload a dataset in .npz format.

    The .npz file should contain arrays with keys 'X' and 'y'
    (or 'inputs'/'outputs', or 'features'/'targets'):
    - X: shape (n_samples, 5) - 5D input features
    - y: shape (n_samples,) - 1D output targets

    Returns:
        dataset_id: Unique identifier for the uploaded dataset
        filename: Original filename
        num_samples: Number of samples in the dataset
        uploaded_at: Timestamp of upload
    """
    # Validate file extension
    if not file.filename.endswith('.npz'):
        raise HTTPException(
            status_code=400,
            detail="Only .npz files are supported"
        )

    # Generate unique dataset ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = f"dataset_{timestamp}"

    # Save uploaded file
    file_path = UPLOAD_DIR / f"{dataset_id}.npz"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

    # Load and validate dataset
    try:
        data = load_data(str(file_path))
        X = data["X"]
        y = data["y"]

        # Store dataset info
        datasets[dataset_id] = {
            "filename": file.filename,
            "filepath": str(file_path),
            "num_samples": len(X),
            "uploaded_at": timestamp,
            "X_shape": X.shape,
            "y_shape": y.shape
        }

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "num_samples": len(X),
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "uploaded_at": timestamp
        }

    except Exception as e:
        # Clean up file if validation fails
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset format: {str(e)}"
        )


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train a model on an uploaded dataset.

    Args:
        request: Training configuration including:
            - dataset_id: ID of uploaded dataset
            - architecture: Model architecture (hidden layers, dropout, etc.)
            - training_params: Training hyperparameters (learning rate, epochs, etc.)
            - split_ratios: Train/Val/Test split ratios
            - standardize: Whether to standardize features
            - missing_strategy: Strategy for handling missing values ('ignore', 'mean', 'median', 'zero', 'forward_fill')

    Returns:
        model_id: Unique identifier for the trained model
        best_epoch: Epoch with best validation performance
        best_val_loss: Best validation loss achieved
        final_train_loss: Final training loss
        test_metrics: Test set evaluation (MSE, RMSE, MAE)
        training_time: Timestamp when training completed
    """
    # Check if dataset exists
    if request.dataset_id not in datasets:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found"
        )

    dataset_info = datasets[request.dataset_id]

    try:
        # Load dataset with splitting and standardization
        data = load_data(
            dataset_info["filepath"],
            missing_strategy=request.missing_strategy,
            split_ratios=tuple(request.split_ratios),
            standardize=request.standardize
        )

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        scaler = data["scaler"]

        # Create trainer
        trainer = NNTrainer(
            model_class=NaiveMLP,
            model_config=request.architecture,
            training_config=request.training_params,
            device=get_device()
        )

        # Train model
        print(f"\n{'='*60}")
        print(f"Training model on dataset: {request.dataset_id}")
        print(f"{'='*60}")

        result = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            verbose=True
        )

        # Evaluate on test set
        test_metrics = trainer.evaluate(X_test, y_test)

        # Generate model ID and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{timestamp}"
        model_path = MODEL_DIR / f"{model_id}.pt"

        # Save model
        trainer.save_model(str(model_path))

        # Save model metadata
        metadata = {
            "model_id": model_id,
            "dataset_id": request.dataset_id,
            "architecture": request.architecture,
            "training_params": request.training_params,
            "split_ratios": request.split_ratios,
            "standardize": request.standardize,
            "missing_strategy": request.missing_strategy,
            "scaler": {
                "mean": scaler["mean"].tolist() if scaler else None,
                "std": scaler["std"].tolist() if scaler else None
            } if scaler else None,
            "best_epoch": result["best_epoch"],
            "best_val_loss": float(result["best_val_loss"]),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "trained_at": timestamp
        }

        metadata_path = MODEL_DIR / f"{model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Store model info
        models[model_id] = {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "trainer": trainer,
            "scaler": scaler,
            **metadata
        }

        print(f"\n{'='*60}")
        print(f"Training completed! Model ID: {model_id}")
        print(f"Test MSE: {test_metrics['mse']:.6f}")
        print(f"Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"Test MAE: {test_metrics['mae']:.6f}")
        print(f"{'='*60}\n")

        return TrainResponse(
            model_id=model_id,
            best_epoch=result["best_epoch"],
            best_val_loss=float(result["best_val_loss"]),
            final_train_loss=float(result["history"]["train_loss"][-1]),
            test_metrics={k: float(v) for k, v in test_metrics.items()},
            training_time=timestamp
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make predictions using a trained model.

    Args:
        request: Prediction request containing:
            - model_id: ID of trained model
            - inputs: List of 5D input vectors

    Returns:
        model_id: ID of model used for prediction
        predictions: List of predicted values
        num_predictions: Number of predictions made
    """
    # Check if model exists
    if request.model_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found"
        )

    model_info = models[request.model_id]

    try:
        # Validate input shape
        X = np.array(request.inputs, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Input must be 2-dimensional, got shape: {X.shape}")

        if X.shape[1] != 5:
            raise ValueError(f"Input must have 5 features, got: {X.shape[1]}")

        # Apply standardization if model was trained with it
        scaler = model_info.get("scaler")
        if scaler is not None:
            mean = np.array(scaler["mean"])
            std = np.array(scaler["std"])
            X = (X - mean) / std

        # Make predictions
        trainer = model_info["trainer"]
        predictions = trainer.predict(X)

        return PredictResponse(
            model_id=request.model_id,
            predictions=predictions.tolist(),
            num_predictions=len(predictions)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/datasets")
async def list_datasets():
    """
    List all uploaded datasets.

    Returns:
        List of dataset information including ID, filename, and metadata.
    """
    return {
        "datasets": [
            {
                "dataset_id": dataset_id,
                **info
            }
            for dataset_id, info in datasets.items()
        ],
        "total": len(datasets)
    }


@app.get("/models")
async def list_models():
    """
    List all trained models.

    Returns:
        List of model information including ID, dataset used, and metrics.
    """
    return {
        "models": [
            {
                "model_id": model_id,
                "dataset_id": info["dataset_id"],
                "best_epoch": info["best_epoch"],
                "best_val_loss": info["best_val_loss"],
                "test_metrics": info["test_metrics"],
                "trained_at": info["trained_at"]
            }
            for model_id, info in models.items()
        ],
        "total": len(models)
    }


@app.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """
    Get detailed information about a specific model.

    Args:
        model_id: ID of the model

    Returns:
        Detailed model information including configuration and metrics.
    """
    if model_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    info = models[model_id]
    return {
        "model_id": model_id,
        "dataset_id": info["dataset_id"],
        "architecture": info["architecture"],
        "training_params": info["training_params"],
        "split_ratios": info["split_ratios"],
        "standardize": info["standardize"],
        "missing_strategy": info["missing_strategy"],
        "best_epoch": info["best_epoch"],
        "best_val_loss": info["best_val_loss"],
        "test_metrics": info["test_metrics"],
        "trained_at": info["trained_at"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
