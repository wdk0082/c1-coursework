"""Profiling script for 5D regression model benchmarking."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd

from data_generate import generate_and_save, FunctionType
from fivedreg.data.load_data import load_data
from fivedreg.model.naive_nn import NaiveMLP
from fivedreg.trainer.nn_trainer import NNTrainer


# Configuration
FUNCTION_TYPES: List[FunctionType] = ["linear", "polynomial", "sin", "expo", "mixed"]
DATASET_SIZES: List[int] = [1000, 5000, 10000]
SCALES: Dict[str, Tuple[float, float]] = {
    "small": (-1.0, 1.0),
    "large": (-10.0, 10.0),
}

# Model and training configuration
MODEL_CONFIG = {
    "hidden_dims": [64, 32],
    "dropout": 0.0,
    "activation": "relu",
}

TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
    "weight_decay": 0.0,
}

SPLIT_RATIOS = (0.7, 0.15, 0.15)
NOISE_STD = 0.1
RANDOM_SEED = 42

# Directories
PROFILING_DIR = Path(__file__).parent
DATA_DIR = PROFILING_DIR / "data"
RESULTS_DIR = PROFILING_DIR / "results"


def get_data_filename(func_type: str, size: int, scale_name: str) -> str:
    """Generate filename for synthetic data."""
    return f"{func_type}_{size}_{scale_name}.pkl"


def generate_all_datasets() -> None:
    """Generate and save all synthetic datasets to profiling/data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = len(FUNCTION_TYPES) * len(DATASET_SIZES) * len(SCALES)
    current = 0

    print("Generating synthetic datasets...")
    for func_type in FUNCTION_TYPES:
        for scale_name, scale in SCALES.items():
            for size in DATASET_SIZES:
                current += 1
                filename = get_data_filename(func_type, size, scale_name)
                filepath = DATA_DIR / filename

                print(f"[{current}/{total}] {filename}")
                generate_and_save(
                    filepath=str(filepath),
                    n_samples=size,
                    function_type=func_type,
                    scale=scale,
                    noise_std=NOISE_STD,
                    random_seed=RANDOM_SEED,
                )

    print(f"Generated {total} datasets in {DATA_DIR}")


def run_single_experiment(
    data_filepath: str,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run a single profiling experiment.

    Args:
        data_filepath: Path to the .pkl data file
        verbose: Print training progress

    Returns:
        Dictionary with metrics:
            - training_time: seconds
            - training_memory_mb: MB
            - inference_memory_mb: MB
            - train_mse: MSE on training set
            - train_r2: R2 on training set
            - test_mse: MSE on test set
            - test_r2: R2 on test set
    """
    # Load and split data using existing load_data function
    data = load_data(
        filepath=data_filepath,
        missing_strategy="ignore",
        standardize=True,
        split_ratios=SPLIT_RATIOS,
        random_seed=RANDOM_SEED,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Create trainer
    trainer = NNTrainer(
        model_class=NaiveMLP,
        model_config=MODEL_CONFIG,
        training_config=TRAINING_CONFIG,
        device="cpu",  # Use CPU for consistent profiling
    )

    # Train
    result = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=verbose,
    )

    # Evaluate on train set
    train_metrics = trainer.evaluate(X_train, y_train)

    # Evaluate on test set
    test_metrics = trainer.evaluate(X_test, y_test)

    # Inference memory (on test set, single batch)
    _, inference_memory_mb = trainer.predict(X_test, track_memory=True)

    return {
        "training_time": result["training_time_seconds"],
        "training_memory_mb": result["training_memory_mb"],
        "inference_memory_mb": inference_memory_mb,
        "train_mse": train_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "test_mse": test_metrics["mse"],
        "test_r2": test_metrics["r2"],
    }


def create_column_name(size: int, scale_name: str) -> str:
    """Create column name like '1K small' or '10K large'."""
    size_str = f"{size // 1000}K"
    return f"{size_str} {scale_name}"


def run_profiling() -> Dict[str, pd.DataFrame]:
    """
    Run full profiling across all function types, sizes, and scales.

    Returns:
        Dictionary mapping function_type -> DataFrame with results
    """
    results = {}

    # Define row labels
    row_labels = [
        "training_time",
        "training_memory_mb",
        "inference_memory_mb",
        "train_mse",
        "train_r2",
        "test_mse",
        "test_r2",
    ]

    # Define column order: 1K small, 5K small, 10K small, 1K large, 5K large, 10K large
    columns = []
    for scale_name in ["small", "large"]:
        for size in DATASET_SIZES:
            columns.append(create_column_name(size, scale_name))

    total_experiments = len(FUNCTION_TYPES) * len(DATASET_SIZES) * len(SCALES)
    current = 0

    for func_type in FUNCTION_TYPES:
        print(f"\n{'='*60}")
        print(f"Function type: {func_type}")
        print(f"{'='*60}")

        # Initialize data for this function type
        data = {col: {} for col in columns}

        for scale_name in SCALES.keys():
            for size in DATASET_SIZES:
                current += 1
                col_name = create_column_name(size, scale_name)
                print(f"\n[{current}/{total_experiments}] {col_name}...")

                # Load from saved data file
                filename = get_data_filename(func_type, size, scale_name)
                filepath = DATA_DIR / filename

                metrics = run_single_experiment(
                    data_filepath=str(filepath),
                    verbose=False,
                )

                # Store metrics
                for metric_name in row_labels:
                    data[col_name][metric_name] = metrics[metric_name]

                print(f"  Training time: {metrics['training_time']:.2f}s")
                print(f"  Test MSE: {metrics['test_mse']:.6f}")
                print(f"  Test R2: {metrics['test_r2']:.4f}")

        # Create DataFrame for this function type
        df = pd.DataFrame(data, index=row_labels)
        df = df[columns]  # Ensure correct column order
        results[func_type] = df

    return results


def save_results(results: Dict[str, pd.DataFrame]) -> None:
    """Save profiling results to files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual CSVs for each function type
    for func_type, df in results.items():
        csv_path = RESULTS_DIR / f"{func_type}_{timestamp}.csv"
        df.to_csv(csv_path)
        print(f"Saved: {csv_path}")

    # Save combined results as JSON
    json_data = {
        "timestamp": timestamp,
        "config": {
            "model_config": MODEL_CONFIG,
            "training_config": TRAINING_CONFIG,
            "split_ratios": SPLIT_RATIOS,
            "noise_std": NOISE_STD,
            "random_seed": RANDOM_SEED,
        },
        "results": {
            func_type: df.to_dict() for func_type, df in results.items()
        },
    }

    json_path = RESULTS_DIR / f"profiling_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_path}")

    # Print summary tables
    print("\n" + "=" * 80)
    print("PROFILING RESULTS SUMMARY")
    print("=" * 80)

    for func_type, df in results.items():
        print(f"\n{'='*60}")
        print(f"Function Type: {func_type.upper()}")
        print(f"{'='*60}")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))


def main():
    """Main entry point."""
    print("Starting profiling...")
    print(f"Function types: {FUNCTION_TYPES}")
    print(f"Dataset sizes: {DATASET_SIZES}")
    print(f"Scales: {SCALES}")

    # Step 1: Generate all datasets
    generate_all_datasets()

    # Step 2: Run profiling experiments
    print("\nRunning profiling experiments...")
    results = run_profiling()

    # Step 3: Save results
    save_results(results)

    print("\nProfiling complete!")


if __name__ == "__main__":
    main()
