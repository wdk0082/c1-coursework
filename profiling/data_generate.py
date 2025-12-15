"""Data generation utilities for 5D regression profiling."""

import pickle
from typing import Literal, Tuple, Optional, Callable, Dict
import numpy as np
from pathlib import Path


# Type alias for function types
FunctionType = Literal["linear", "polynomial", "sin", "expo", "mixed"]


def generate_5d_data(
    n_samples: int,
    function_type: FunctionType = "linear",
    scale: Tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 5D regression data.

    Args:
        n_samples: Number of samples to generate
        function_type: Type of function to generate y values:
            - "linear": y = sum of weighted features
            - "polynomial": y = polynomial combination of features
            - "sin": y = sinusoidal combination of features
            - "expo": y = exponential combination of features
            - "mixed": y = combination of different function types
        scale: Tuple of (min, max) for feature values range
        noise_std: Standard deviation of Gaussian noise added to y
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where:
            - X: numpy array of shape (n_samples, 5)
            - y: numpy array of shape (n_samples,)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if scale[0] >= scale[1]:
        raise ValueError(f"scale[0] must be less than scale[1], got {scale}")

    rng = np.random.RandomState(random_seed)

    # Generate X uniformly in the given scale
    X = rng.uniform(scale[0], scale[1], size=(n_samples, 5))

    # Get the function to compute y
    func = _get_function(function_type)
    y = func(X)

    # Add noise
    if noise_std > 0:
        y = y + rng.normal(0, noise_std, size=n_samples)

    return X, y


def _get_function(function_type: FunctionType) -> Callable[[np.ndarray], np.ndarray]:
    """Get the function corresponding to the function type."""
    functions: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "linear": _linear_function,
        "polynomial": _polynomial_function,
        "sin": _sin_function,
        "expo": _expo_function,
        "mixed": _mixed_function,
    }

    if function_type not in functions:
        raise ValueError(
            f"Unknown function type: {function_type}. "
            f"Available types: {list(functions.keys())}"
        )

    return functions[function_type]


def _linear_function(X: np.ndarray) -> np.ndarray:
    """
    Linear function: y = 2*x1 + 3*x2 - 1.5*x3 + 0.5*x4 - 2*x5
    """
    weights = np.array([2.0, 3.0, -1.5, 0.5, -2.0])
    return X @ weights


def _polynomial_function(X: np.ndarray) -> np.ndarray:
    """
    Polynomial function: y = x1^2 + x2^2 + x3*x4 + x5^3
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return x1**2 + x2**2 + x3 * x4 + x5**3


def _sin_function(X: np.ndarray) -> np.ndarray:
    """
    Sinusoidal function: y = sin(pi*x1) + cos(pi*x2) + sin(2*pi*x3) + cos(x4*x5)
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (
        np.sin(np.pi * x1)
        + np.cos(np.pi * x2)
        + np.sin(2 * np.pi * x3)
        + np.cos(x4 * x5)
    )


def _expo_function(X: np.ndarray) -> np.ndarray:
    """
    Exponential function: y = exp(x1/2) + exp(-x2) + x3*exp(x4/2) + log(1+exp(x5))
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (
        np.exp(x1 / 2)
        + np.exp(-x2)
        + x3 * np.exp(x4 / 2)
        + np.log(1 + np.exp(x5))
    )


def _mixed_function(X: np.ndarray) -> np.ndarray:
    """
    Mixed function combining linear, polynomial, and sinusoidal components:
    y = x1 + x2^2 + sin(pi*x3) + x4*x5 + exp(x1/3)
    """
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (
        x1
        + x2**2
        + np.sin(np.pi * x3)
        + x4 * x5
        + np.exp(x1 / 3)
    )


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    filepath: str,
) -> None:
    """
    Save generated dataset to a .pkl file.

    Args:
        X: Input features of shape (n_samples, 5)
        y: Output targets of shape (n_samples,)
        filepath: Path to save the .pkl file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"X": X, "y": y}

    with open(path, "wb") as f:
        pickle.dump(data, f)


def generate_and_save(
    filepath: str,
    n_samples: int,
    function_type: FunctionType = "linear",
    scale: Tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 5D regression data and save to .pkl file.

    Args:
        filepath: Path to save the .pkl file
        n_samples: Number of samples to generate
        function_type: Type of function to generate y values
        scale: Tuple of (min, max) for feature values range
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y)
    """
    X, y = generate_5d_data(
        n_samples=n_samples,
        function_type=function_type,
        scale=scale,
        noise_std=noise_std,
        random_seed=random_seed,
    )

    save_dataset(X, y, filepath)

    return X, y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 5D regression data")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--function_type",
        type=str,
        default="linear",
        choices=["linear", "polynomial", "sin", "expo", "mixed"],
        help="Function type for generating y",
    )
    parser.add_argument("--scale_min", type=float, default=-1.0, help="Minimum scale value")
    parser.add_argument("--scale_max", type=float, default=1.0, help="Maximum scale value")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Noise standard deviation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data.pkl", help="Output file path")

    args = parser.parse_args()

    X, y = generate_and_save(
        filepath=args.output,
        n_samples=args.n_samples,
        function_type=args.function_type,
        scale=(args.scale_min, args.scale_max),
        noise_std=args.noise_std,
        random_seed=args.seed,
    )

    print(f"Generated {args.n_samples} samples with function type '{args.function_type}'")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Saved to: {args.output}")
