"""
CandorFlow Visualization Demo
==============================

This script creates visualizations of the λ(t) stability metric and
controller actions from a training run.

NOTE: This demonstrates only the simplified public metric.
Advanced visualizations from the proprietary system are not included.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_lambda_curve(lambda_values, actions, threshold, save_path="plots/lambda_curve.png"):
    """
    Plot the λ(t) curve with threshold and intervention markers.
    
    Args:
        lambda_values: List of lambda values over time
        actions: List of action dictionaries from controller
        threshold: Stability threshold value
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    steps = range(len(lambda_values))
    
    # Plot lambda curve
    plt.plot(steps, lambda_values, linewidth=2, label='λ(t) - Stability Metric', color='#2E86AB')
    
    # Plot threshold
    plt.axhline(y=threshold, color='#A23B72', linestyle='--', linewidth=2, label='Threshold')
    
    # Mark intervention points
    rollback_steps = [i for i, a in enumerate(actions) if a["action"] == "rollback"]
    warning_steps = [i for i, a in enumerate(actions) if a["action"] == "warning"]
    
    if rollback_steps:
        plt.scatter(
            rollback_steps,
            [lambda_values[i] for i in rollback_steps],
            color='#F18F01',
            s=100,
            marker='o',
            label='Rollback + LR Reduction',
            zorder=5
        )
    
    if warning_steps:
        plt.scatter(
            warning_steps,
            [lambda_values[i] for i in warning_steps],
            color='#C73E1D',
            s=100,
            marker='^',
            label='Warning',
            zorder=5
        )
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('λ(t) - Instability Metric', fontsize=12)
    plt.title('CandorFlow Training Stability Monitor (Simplified Demo)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {save_path}")
    plt.close()


def plot_stability_phases(lambda_values, threshold, save_path="plots/stability_phases.png"):
    """
    Plot training phases: stable, warning, unstable.
    
    Args:
        lambda_values: List of lambda values over time
        threshold: Stability threshold value
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    steps = np.arange(len(lambda_values))
    
    # Create phase colors
    stable = np.array(lambda_values) < threshold * 0.5
    warning = (np.array(lambda_values) >= threshold * 0.5) & (np.array(lambda_values) < threshold)
    unstable = np.array(lambda_values) >= threshold
    
    # Plot with color-coded background
    plt.fill_between(steps, 0, threshold * 0.5, alpha=0.2, color='green', label='Stable Zone')
    plt.fill_between(steps, threshold * 0.5, threshold, alpha=0.2, color='orange', label='Warning Zone')
    plt.fill_between(steps, threshold, max(lambda_values) * 1.1, alpha=0.2, color='red', label='Unstable Zone')
    
    # Plot lambda curve
    plt.plot(steps, lambda_values, linewidth=2, color='black', label='λ(t)', zorder=3)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('λ(t)', fontsize=12)
    plt.title('Training Stability Phases', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.ylim(0, max(lambda_values) * 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {save_path}")
    plt.close()


def main():
    """Generate all visualization plots."""
    
    print("=" * 60)
    print("CandorFlow Visualization Demo")
    print("=" * 60)
    print("NOTE: These plots show simplified metrics only.")
    print("The full proprietary system includes additional visualizations.")
    print("=" * 60 + "\n")
    
    # Load results
    results_path = "plots/training_results.pt"
    
    if not os.path.exists(results_path):
        print(f"❌ Error: {results_path} not found.")
        print("Please run examples/demo_training_loop.py first.")
        return
    
    print(f"📊 Loading results from {results_path}...")
    results = torch.load(results_path)
    
    lambda_values = results["lambda_values"]
    actions = results["actions"]
    summary = results["summary"]
    threshold = summary["threshold"]
    
    print(f"✓ Loaded {len(lambda_values)} data points\n")
    
    # Generate plots
    print("🎨 Generating plots...")
    plot_lambda_curve(lambda_values, actions, threshold)
    plot_stability_phases(lambda_values, threshold)
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total Interventions: {summary['total_interventions']}")
    print(f"Max λ(t): {summary['max_lambda']:.4f}")
    print(f"Mean λ(t): {summary['mean_lambda']:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("=" * 60)
    print("\n✓ All plots generated successfully!")
    print("✓ Check the plots/ directory for output files.")


if __name__ == "__main__":
    main()

