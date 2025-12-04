"""Experiment runner for the 5D regression coursework dataset."""

from __future__ import annotations

from pathlib import Path
import argparse
import pickle
from typing import Dict, Tuple

import numpy as np


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Load the pickled dataset bundled with the coursework."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return (
        payload["X"].astype(np.float64, copy=False),
        payload["y"].astype(np.float64, copy=False),
        payload["metadata"],
    )


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically shuffle and split the dataset."""
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    cutoff = int(n_samples * train_ratio)
    train_idx = indices[:cutoff]
    test_idx = indices[cutoff:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(
    train_X: np.ndarray,
    test_X: np.ndarray,
    reference: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize features using either the train or test split as the scaler."""
    if reference not in {"train", "test"}:
        msg = "reference must be either 'train' or 'test'"
        raise ValueError(msg)

    ref = train_X if reference == "train" else test_X
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    std[std == 0.0] = 1.0  # avoid division by zero
    return (train_X - mean) / std, (test_X - mean) / std


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a linear regressor via least squares with bias term."""
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    design = np.concatenate([X, ones], axis=1)
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return weights


def evaluate(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Compute MAE and MSE for the provided weights."""
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    design = np.concatenate([X, ones], axis=1)
    preds = design @ weights
    residuals = preds - y
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    return mae, mse


def run_experiment(dataset_path: Path) -> Dict[str, Dict[str, float]]:
    X, y, metadata = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_ratio=0.8,
        seed=int(metadata.get("seed", 0)),
    )

    results: Dict[str, Dict[str, float]] = {}
    for reference in ("train", "test"):
        scaled_train, scaled_test = standardize(X_train, X_test, reference)
        weights = fit_linear_regression(scaled_train, y_train)
        mae, mse = evaluate(scaled_test, y_test, weights)
        results[reference] = {"mae": mae, "mse": mse}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare train-vs-test standardization strategies."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "coursework_dataset.pkl",
        help="Path to the pickled dataset.",
    )
    args = parser.parse_args()
    results = run_experiment(args.dataset)
    for reference, metrics in results.items():
        print(
            f"{reference:>5} scaler -> MAE: {metrics['mae']:.6f}, "
            f"MSE: {metrics['mse']:.6f}"
        )


if __name__ == "__main__":
    main()
