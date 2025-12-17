"""Plot profiling results from saved JSON files."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


# Configuration
DATASET_SIZES = [1000, 5000, 10000]
DATASET_SIZE_LABELS = ["1K", "5K", "10K"]
SCALES = ["small", "large"]

PROFILING_DIR = Path(__file__).parent
RESULTS_DIR = PROFILING_DIR / "results"


def load_profiling_results(json_path: str) -> Dict[str, Any]:
    """Load profiling results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_metric_by_scale(
    func_results: Dict[str, Dict[str, float]],
    metric: str,
    scale: str,
) -> List[float]:
    """
    Extract metric values for a given scale across all dataset sizes.

    Args:
        func_results: Results dict for one function type
        metric: Metric name (e.g., 'training_time')
        scale: Scale name ('small' or 'large')

    Returns:
        List of metric values for [1K, 5K, 10K]
    """
    values = []
    for size in DATASET_SIZES:
        col_name = f"{size // 1000}K {scale}"
        values.append(func_results[col_name][metric])
    return values


def plot_function_type(
    func_type: str,
    func_results: Dict[str, Dict[str, float]],
    output_dir: Path,
    timestamp: str,
) -> None:
    """
    Create a figure with 4 subplots for one function type.

    Subplots:
        1. Training time vs dataset size
        2. Memory (train + inference) vs dataset size
        3. MSE (train + test) vs dataset size
        4. R² (train + test) vs dataset size
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Profiling Results: {func_type.upper()}", fontsize=14, fontweight="bold")

    x = np.arange(len(DATASET_SIZE_LABELS))

    # Colors for scales
    colors = {"small": "tab:blue", "large": "tab:orange"}
    markers = {"small": "o", "large": "s"}

    # Subplot 1: Training Time
    ax1 = axes[0, 0]
    for scale in SCALES:
        values = extract_metric_by_scale(func_results, "training_time", scale)
        ax1.plot(x, values, marker=markers[scale], color=colors[scale],
                 label=f"{scale} scale", linewidth=2, markersize=8)
    ax1.set_xlabel("Dataset Size")
    ax1.set_ylabel("Training Time (s)")
    ax1.set_title("Training Time vs Dataset Size")
    ax1.set_xticks(x)
    ax1.set_xticklabels(DATASET_SIZE_LABELS)
    ax1.legend(fontsize=7, markerscale=0.5, handlelength=4, handletextpad=0.8)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Memory Usage (train + inference)
    ax2 = axes[0, 1]
    for scale in SCALES:
        train_mem = extract_metric_by_scale(func_results, "training_memory_mb", scale)
        infer_mem = extract_metric_by_scale(func_results, "inference_memory_mb", scale)
        ax2.plot(x, train_mem, marker=markers[scale], color=colors[scale],
                 label=f"Train ({scale})", linewidth=2, markersize=8)
        ax2.plot(x, infer_mem, marker=markers[scale], color=colors[scale],
                 label=f"Infer ({scale})", linewidth=2, markersize=8, linestyle="--")
    ax2.set_xlabel("Dataset Size")
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Memory Usage vs Dataset Size")
    ax2.set_xticks(x)
    ax2.set_xticklabels(DATASET_SIZE_LABELS)
    ax2.legend(fontsize=7, markerscale=0.5, handlelength=4, handletextpad=0.8)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: MSE (train + test)
    ax3 = axes[1, 0]
    for scale in SCALES:
        train_mse = extract_metric_by_scale(func_results, "train_mse", scale)
        test_mse = extract_metric_by_scale(func_results, "test_mse", scale)
        ax3.plot(x, train_mse, marker=markers[scale], color=colors[scale],
                 label=f"Train ({scale})", linewidth=2, markersize=8)
        ax3.plot(x, test_mse, marker=markers[scale], color=colors[scale],
                 label=f"Test ({scale})", linewidth=2, markersize=8, linestyle="--")
    ax3.set_xlabel("Dataset Size")
    ax3.set_ylabel("MSE")
    ax3.set_title("MSE vs Dataset Size")
    ax3.set_xticks(x)
    ax3.set_xticklabels(DATASET_SIZE_LABELS)
    ax3.legend(fontsize=7, markerscale=0.5, handlelength=4, handletextpad=0.8)
    ax3.grid(True, alpha=0.3)
    # Use log scale if values vary widely
    mse_values = []
    for scale in SCALES:
        mse_values.extend(extract_metric_by_scale(func_results, "train_mse", scale))
        mse_values.extend(extract_metric_by_scale(func_results, "test_mse", scale))
    if max(mse_values) / (min(mse_values) + 1e-10) > 100:
        ax3.set_yscale("log")

    # Subplot 4: R² (train + test)
    ax4 = axes[1, 1]
    for scale in SCALES:
        train_r2 = extract_metric_by_scale(func_results, "train_r2", scale)
        test_r2 = extract_metric_by_scale(func_results, "test_r2", scale)
        ax4.plot(x, train_r2, marker=markers[scale], color=colors[scale],
                 label=f"Train ({scale})", linewidth=2, markersize=8)
        ax4.plot(x, test_r2, marker=markers[scale], color=colors[scale],
                 label=f"Test ({scale})", linewidth=2, markersize=8, linestyle="--")
    ax4.set_xlabel("Dataset Size")
    ax4.set_ylabel("R²")
    ax4.set_title("R² vs Dataset Size")
    ax4.set_xticks(x)
    ax4.set_xticklabels(DATASET_SIZE_LABELS)
    ax4.legend(fontsize=7, markerscale=0.5, handlelength=4, handletextpad=0.8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.05)  # R² typically between 0 and 1

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"{func_type}_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_results(json_path: str, output_dir: Path = None) -> None:
    """
    Plot all profiling results from a JSON file.

    Creates one figure per function type with 4 subplots each.
    """
    data = load_profiling_results(json_path)
    timestamp = data["timestamp"]
    results = data["results"]

    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting results from: {json_path}")
    print(f"Output directory: {output_dir}")

    for func_type, func_results in results.items():
        print(f"\nPlotting: {func_type}")
        plot_function_type(func_type, func_results, output_dir, timestamp)

    print("\nPlotting complete!")


def find_latest_results() -> str:
    """Find the most recent profiling results JSON file."""
    json_files = list(RESULTS_DIR.glob("profiling_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No profiling results found in {RESULTS_DIR}")

    # Sort by modification time, newest first
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(json_files[0])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot profiling results")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to profiling results JSON file. If not provided, uses the latest.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to profiling/results/",
    )

    args = parser.parse_args()

    # Find input file
    if args.input:
        json_path = args.input
    else:
        json_path = find_latest_results()
        print(f"Using latest results: {json_path}")

    # Set output directory
    output_dir = Path(args.output) if args.output else RESULTS_DIR

    plot_all_results(json_path, output_dir)


if __name__ == "__main__":
    main()
