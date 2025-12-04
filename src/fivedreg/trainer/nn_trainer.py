"""Neural network trainer for 5D to 1D regression."""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


class NNTrainer:
    """
    Trainer for neural network models.

    Handles model instantiation, training loop, validation, and checkpointing.
    """

    def __init__(
        self,
        model_class,
        model_config: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model_class: Model class to instantiate (e.g., NaiveMLP or create_model function)
            model_config: Configuration dict for model initialization
            training_config: Training configuration with keys:
                - "learning_rate": float (default: 1e-3)
                - "batch_size": int (default: 32)
                - "epochs": int (default: 100)
                - "early_stopping_patience": int (default: 10, 0 to disable)
                - "weight_decay": float (default: 0.0)
                - "checkpoint_dir": str (default: "./checkpoints")
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        if callable(model_class):
            self.model = model_class(**model_config)
        else:
            self.model = model_class
        self.model.to(self.device)

        # Training configuration
        self.training_config = training_config or {}
        self.learning_rate = self.training_config.get("learning_rate", 1e-3)
        self.batch_size = self.training_config.get("batch_size", 32)
        self.epochs = self.training_config.get("epochs", 100)
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 10)
        self.weight_decay = self.training_config.get("weight_decay", 0.0)
        self.checkpoint_dir = Path(self.training_config.get("checkpoint_dir", "./checkpoints"))

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": []
        }

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model: {self.model}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features of shape (n_train, 5)
            y_train: Training targets of shape (n_train,)
            X_val: Validation features of shape (n_val, 5) (optional)
            y_val: Validation targets of shape (n_val,) (optional)
            verbose: If True, print training progress

        Returns:
            Dictionary containing:
                - "model": Trained model
                - "history": Training history
                - "best_epoch": Epoch with best validation loss
                - "best_val_loss": Best validation loss achieved
        """
        # Prepare data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        else:
            val_loader = None

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        # Training loop
        for epoch in range(self.epochs):
            # Train one epoch
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate if validation data provided
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint(epoch, val_loss, best=True)
                else:
                    patience_counter += 1

                # Print progress
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.epochs}] "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f}"
                    )

                # Early stopping
                if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                # No validation set
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] Train Loss: {train_loss:.6f}")

        # Load best model if validation was used
        if val_loader is not None:
            self._load_best_checkpoint()
            if verbose:
                print(f"\nTraining completed!")
                print(f"Best epoch: {best_epoch+1} with validation loss: {best_val_loss:.6f}")
        else:
            best_val_loss = None
            if verbose:
                print(f"\nTraining completed!")

        return {
            "model": self.model,
            "history": self.history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        }

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _save_checkpoint(self, epoch: int, val_loss: float, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }

        if best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)

    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        path = self.checkpoint_dir / "best_model.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features of shape (n_test, 5)
            y_test: Test targets of shape (n_test,)

        Returns:
            Dictionary with evaluation metrics:
                - "mse": Mean Squared Error
                - "rmse": Root Mean Squared Error
                - "mae": Mean Absolute Error
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.FloatTensor(y_test).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze()

        mse = nn.MSELoss()(predictions, y_tensor).item()
        mae = nn.L1Loss()(predictions, y_tensor).item()
        rmse = np.sqrt(mse)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features of shape (n_samples, 5)

        Returns:
            Predictions as numpy array of shape (n_samples,)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze()

        return predictions.cpu().numpy()

    def save_model(self, filepath: str):
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Model loaded from {filepath}")
