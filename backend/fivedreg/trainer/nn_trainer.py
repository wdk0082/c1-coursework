from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import time
import tracemalloc


class NNTrainer:
    """
    Trainer for neural network models.

    Examples
    --------
    >>> from fivedreg.model.naive_nn import NaiveMLP
    >>> trainer = NNTrainer(
    ...     model_class=NaiveMLP,
    ...     model_config={"hidden_dims": [64, 32]},
    ...     training_config={"epochs": 50, "learning_rate": 1e-3}
    ... )
    >>> result = trainer.fit(X_train, y_train, X_val, y_val)
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

        Parameters
        ----------
        model_class : type or callable
            Model class to instantiate (e.g., NaiveMLP) or factory function.
        model_config : dict
            Configuration dictionary for model initialization.
        training_config : dict, optional
            Training configuration. See class docstring for supported keys.
        device : str, optional
            Device to use ('cuda', 'cpu', or None for auto-detect).
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

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model.

        Parameters
        ----------
        X_train : ndarray of shape (n_train, 5)
            Training features.
        y_train : ndarray of shape (n_train,)
            Training targets.
        X_val : ndarray of shape (n_val, 5), optional
            Validation features.
        y_val : ndarray of shape (n_val,), optional
            Validation targets.
        verbose : bool, default=True
            If True, print training progress every 10 epochs.

        Returns
        -------
        dict
            Dictionary containing:

            - **model** : torch.nn.Module - Trained model
            - **history** : dict - Training history with loss values
            - **best_epoch** : int - Epoch with best validation loss
            - **best_val_loss** : float or None - Best validation loss achieved
            - **training_time_seconds** : float - Total training time in seconds
            - **training_memory_mb** : float - Peak memory usage in MB
        """
        # Start tracking time and memory
        start_time = time.time()
        tracemalloc.start()

        # Prepare data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        else:
            val_loader = None

        # Best model tracking
        best_val_loss = float('inf')
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

                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    # Save best model
                    self._save_checkpoint(epoch, val_loss, best=True)

                # Print progress
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.epochs}] "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f}"
                    )
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

        # Stop tracking time and memory
        training_time_seconds = time.time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        training_memory_mb = peak / (1024 * 1024)

        if verbose:
            print(f"Training time: {training_time_seconds:.2f}s")
            print(f"Peak memory: {training_memory_mb:.2f} MB")

        return {
            "model": self.model,
            "history": self.history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_time_seconds": training_time_seconds,
            "training_memory_mb": training_memory_mb
        }

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.

        Returns
        -------
        float
            Average training loss for the epoch.
        """
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
        """
        Validate the model on validation data.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        float
            Average validation loss.
        """
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
        """
        Create a PyTorch DataLoader from numpy arrays.

        Parameters
        ----------
        X : ndarray
            Input features.
        y : ndarray
            Target values.
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        DataLoader
            PyTorch DataLoader instance.
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _save_checkpoint(self, epoch: int, val_loss: float, best: bool = False):
        """
        Save model checkpoint to disk.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        val_loss : float
            Current validation loss.
        best : bool, default=False
            If True, save as best model checkpoint.
        """
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
        """
        Load the best model checkpoint from disk.

        Loads the model state from 'best_model.pt' in the checkpoint directory.
        """
        path = self.checkpoint_dir / "best_model.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Parameters
        ----------
        X_test : ndarray of shape (n_test, 5)
            Test features.
        y_test : ndarray of shape (n_test,)
            Test targets.

        Returns
        -------
        dict
            Dictionary with evaluation metrics:

            - **mse** : float - Mean Squared Error
            - **rmse** : float - Root Mean Squared Error
            - **mae** : float - Mean Absolute Error
            - **r2** : float - R-squared (coefficient of determination)
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.FloatTensor(y_test).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze()

        mse = nn.MSELoss()(predictions, y_tensor).item()
        mae = nn.L1Loss()(predictions, y_tensor).item()
        rmse = np.sqrt(mse)

        # Calculate R2 score
        ss_res = torch.sum((y_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_tensor - y_tensor.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    def predict(self, X: np.ndarray, track_memory: bool = False) -> np.ndarray | tuple:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 5)
            Input features.
        track_memory : bool, default=False
            If True, also return peak memory usage in MB.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values. Returned alone if ``track_memory=False``.
        memory_mb : float
            Peak memory usage in MB. Only returned if ``track_memory=True``,
            in which case return value is ``(predictions, memory_mb)``.
        """
        if track_memory:
            tracemalloc.start()

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze(-1)

        # Ensure predictions is always 1D, even for single sample
        predictions_np = predictions.cpu().numpy()
        if predictions_np.ndim == 0:
            predictions_np = predictions_np.reshape(1)

        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / (1024 * 1024)
            return predictions_np, memory_mb

        return predictions_np

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model file.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Model loaded from {filepath}")
