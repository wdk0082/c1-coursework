"""
Data loading utilities for 5D to 1D regression.

This module provides functions for loading, validating, and preprocessing
data from pickle or npz files.
"""

import pickle
from pathlib import Path
from typing import Tuple, Literal, Optional, Dict, Any
import numpy as np


def load_data(
    filepath: str,
    missing_strategy: Literal["ignore", "mean", "median", "zero", "forward_fill"] = "ignore",
    standardize: bool = False,
    split_ratios: Optional[Tuple[float, float, float]] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Load and validate data from a pickle or npz file for 5D to 1D regression.

    Parameters
    ----------
    filepath : str
        Path to the .pkl or .npz file containing the data.
    missing_strategy : {'ignore', 'mean', 'median', 'zero', 'forward_fill'}, default='ignore'
        Strategy for handling missing values:

        - 'ignore': Remove rows with any missing values
        - 'mean': Fill missing values with column mean
        - 'median': Fill missing values with column median
        - 'zero': Fill missing values with zeros
        - 'forward_fill': Forward fill missing values

    standardize : bool, default=False
        If True, standardize features to zero mean and unit variance.
    split_ratios : tuple of float, optional
        Tuple of (train, val, test) ratios that must sum to 1.0.
        If None, returns the full dataset. Example: (0.7, 0.15, 0.15).
    random_seed : int, default=42
        Random seed for reproducible splits.

    Returns
    -------
    dict
        Dictionary containing the loaded data:

        If ``split_ratios`` is None:
            - **X** : ndarray of shape (n_samples, 5) - Input features
            - **y** : ndarray of shape (n_samples,) - Output targets
            - **scaler** : dict or None - Contains 'mean' and 'std' if standardize=True

        If ``split_ratios`` is provided:
            - **X_train**, **y_train** : Training data
            - **X_val**, **y_val** : Validation data
            - **X_test**, **y_test** : Test data
            - **scaler** : dict or None - Scaler fitted on training data only

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If data format is invalid, dimensions don't match, or file format is unsupported.
    KeyError
        If expected keys ('X'/'y', 'inputs'/'outputs', or 'features'/'targets')
        are missing from the data file.

    Examples
    --------
    Load data without splitting:

    >>> data = load_data("data.pkl")
    >>> X, y = data["X"], data["y"]

    Load with train/val/test split and standardization:

    >>> data = load_data(
    ...     "data.pkl",
    ...     standardize=True,
    ...     split_ratios=(0.7, 0.15, 0.15)
    ... )
    >>> X_train, y_train = data["X_train"], data["y_train"]

    Notes
    -----
    When ``standardize=True`` and ``split_ratios`` is provided, the scaler
    is fitted only on the training data to prevent data leakage.
    """
    # Detect file format and load data
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    file_extension = file_path.suffix.lower()

    try:
        if file_extension == '.npz':
            # Load NPZ file
            npz_data = np.load(filepath)
            # Convert NpzFile to dict for consistent processing
            data = {key: npz_data[key] for key in npz_data.files}
        elif file_extension == '.pkl' or file_extension == '.pickle':
            # Load pickle file
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats are: .npz, .pkl, .pickle"
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading {file_extension} file: {str(e)}")

    # Extract X and y based on common data formats
    if isinstance(data, dict):
        # Try common key names
        if 'X' in data and 'y' in data:
            X, y = data['X'], data['y']
        elif 'inputs' in data and 'outputs' in data:
            X, y = data['inputs'], data['outputs']
        elif 'features' in data and 'targets' in data:
            X, y = data['features'], data['targets']
        else:
            raise KeyError(
                f"Expected keys 'X' and 'y' (or 'inputs'/'outputs' or 'features'/'targets') "
                f"in data file. Found keys: {list(data.keys())}"
            )
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        X, y = data[0], data[1]
    else:
        raise ValueError(
            f"Expected data file to contain either a dict with 'X' and 'y' keys, "
            f"or a tuple/list of (X, y). Got type: {type(data)}"
        )

    # Convert to numpy arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Validate dimensions
    if X.ndim != 2:
        raise ValueError(f"Input X must be 2-dimensional, got shape: {X.shape}")

    if X.shape[1] != 5:
        raise ValueError(f"Input X must have 5 features (columns), got: {X.shape[1]}")

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()  # Convert (n, 1) to (n,)
    elif y.ndim != 1:
        raise ValueError(f"Output y must be 1-dimensional, got shape: {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Number of samples mismatch: X has {X.shape[0]} samples, "
            f"y has {y.shape[0]} samples"
        )

    # Handle missing values
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        X, y = _handle_missing_values(X, y, missing_strategy)

    # Validate split ratios if provided
    if split_ratios is not None:
        if len(split_ratios) != 3:
            raise ValueError(f"split_ratios must be a tuple of 3 values, got {len(split_ratios)}")
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")
        if any(r < 0 or r > 1 for r in split_ratios):
            raise ValueError(f"split_ratios must be between 0 and 1, got {split_ratios}")

    # Split data if requested
    if split_ratios is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
            X, y, split_ratios, random_seed
        )

        # Standardize features if requested (fit on training data only)
        scaler = None
        if standardize:
            X_train, scaler = _standardize_features(X_train, fit=True)
            X_val, _ = _standardize_features(X_val, fit=False, scaler=scaler)
            X_test, _ = _standardize_features(X_test, fit=False, scaler=scaler)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "scaler": scaler
        }
    else:
        # Standardize features if requested
        scaler = None
        if standardize:
            X, scaler = _standardize_features(X, fit=True)

        return {
            "X": X,
            "y": y,
            "scaler": scaler
        }


def _handle_missing_values(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values in the dataset according to the specified strategy.

    Parameters
    ----------
    X : ndarray
        Input features of shape (n_samples, n_features).
    y : ndarray
        Output targets of shape (n_samples,).
    strategy : str
        Missing value handling strategy.

    Returns
    -------
    X_clean : ndarray
        Input features with missing values handled.
    y_clean : ndarray
        Output targets with missing values handled.
    """
    if strategy == "ignore":
        # Remove rows with any missing values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        n_removed = len(X) - len(X_clean)
        if n_removed > 0:
            print(f"Removed {n_removed} samples with missing values ({n_removed/len(X)*100:.1f}%)")
        return X_clean, y_clean

    elif strategy == "mean":
        # Fill missing values with column mean
        X_filled = X.copy()
        for col in range(X.shape[1]):
            col_mean = np.nanmean(X[:, col])
            X_filled[np.isnan(X[:, col]), col] = col_mean

        y_filled = y.copy()
        if np.any(np.isnan(y)):
            y_filled[np.isnan(y)] = np.nanmean(y)

        return X_filled, y_filled

    elif strategy == "median":
        # Fill missing values with column median
        X_filled = X.copy()
        for col in range(X.shape[1]):
            col_median = np.nanmedian(X[:, col])
            X_filled[np.isnan(X[:, col]), col] = col_median

        y_filled = y.copy()
        if np.any(np.isnan(y)):
            y_filled[np.isnan(y)] = np.nanmedian(y)

        return X_filled, y_filled

    elif strategy == "zero":
        # Fill missing values with zeros
        X_filled = np.nan_to_num(X, nan=0.0)
        y_filled = np.nan_to_num(y, nan=0.0)
        return X_filled, y_filled

    elif strategy == "forward_fill":
        # Forward fill missing values (use previous valid value)
        X_filled = X.copy()
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                # Forward fill
                idx = np.where(~mask, np.arange(len(mask)), 0)
                np.maximum.accumulate(idx, out=idx)
                X_filled[:, col] = X_filled[idx, col]

        y_filled = y.copy()
        if np.any(np.isnan(y)):
            mask = np.isnan(y)
            idx = np.where(~mask, np.arange(len(mask)), 0)
            np.maximum.accumulate(idx, out=idx)
            y_filled = y_filled[idx]

        return X_filled, y_filled

    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")


def _standardize_features(
    X: np.ndarray,
    fit: bool = True,
    scaler: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Standardize features to zero mean and unit variance.

    Parameters
    ----------
    X : ndarray
        Input features of shape (n_samples, n_features).
    fit : bool, default=True
        If True, compute mean and std from X. If False, use provided scaler.
    scaler : dict, optional
        Dictionary with 'mean' and 'std' arrays. Required if fit=False.

    Returns
    -------
    X_standardized : ndarray
        Standardized features with zero mean and unit variance.
    scaler : dict or None
        Dictionary containing 'mean' and 'std' arrays used for transformation.
    """
    if fit:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Avoid division by zero for constant features
        std = np.where(std == 0, 1.0, std)
        scaler = {"mean": mean, "std": std}
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        mean = scaler["mean"]
        std = scaler["std"]

    X_standardized = (X - mean) / std
    return X_standardized, scaler


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    split_ratios: Tuple[float, float, float],
    random_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.

    Parameters
    ----------
    X : ndarray
        Input features of shape (n_samples, n_features).
    y : ndarray
        Output targets of shape (n_samples,).
    split_ratios : tuple of float
        Tuple of (train, val, test) ratios that sum to 1.0.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_train : ndarray
        Training features.
    X_val : ndarray
        Validation features.
    X_test : ndarray
        Test features.
    y_train : ndarray
        Training targets.
    y_val : ndarray
        Validation targets.
    y_test : ndarray
        Test targets.
    """
    n_samples = len(X)
    train_ratio, val_ratio, _ = split_ratios  # test_ratio not needed (implicit)

    # Calculate split indices
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    # Shuffle indices
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
