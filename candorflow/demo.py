"""
CandorFlow Demo: Complete Training Stability Demonstration
==========================================================

This module provides a clean, stable training demo that showcases
CandorFlow's stability monitoring and intervention capabilities.

All training logic is contained here for easy Colab integration.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from .lambda_metric import compute_lambda_metric
from .stability_controller import StabilityController
from .utils import set_seed


class TinyModel(nn.Module):
    """Simple MLP for demo purposes."""
    
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def run_demo(
    steps: int = 50,
    spike_step: int = 30,
    threshold: float = 2.0,
    lr: float = 1e-3
) -> Dict:
    """
    Runs the full CandorFlow public training demo.
    
    Args:
        steps: Number of training steps
        spike_step: Step at which to inject synthetic instability spike
        threshold: Lambda threshold for intervention
        lr: Initial learning rate
    
    Returns:
        Dictionary containing:
            - 'lambda_history': List of lambda values
            - 'loss_history': List of loss values
            - 'interventions': List of intervention events
            - 'threshold': Threshold value used
    """
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create stability controller
    controller = StabilityController(
        threshold=threshold,
        checkpoint_dir="./checkpoints",
        lr_reduction_factor=0.5
    )
    
    # Training state
    lambda_history: List[float] = []
    loss_history: List[float] = []
    interventions: List[Dict] = []
    gradient_history: List[float] = []
    
    # Training loop
    for step in range(steps):
        # Generate synthetic batch
        batch_size = 16
        inputs = torch.randn(batch_size, 32).to(device)
        
        # Forward pass for lambda computation
        model.train()
        outputs_lambda = model(inputs)
        loss_lambda = outputs_lambda.mean()
        
        # Compute lambda BEFORE optimizer step
        lambda_value = compute_lambda_metric(
            model=model,
            loss=loss_lambda,
            history_window=10,
            gradient_history=gradient_history
        )
        
        # Synthetic instability spike
        if step == spike_step:
            lambda_value = 3.0  # Force spike above threshold
        
        lambda_history.append(lambda_value)
        
        # Update controller
        action = controller.update(
            lambda_value=lambda_value,
            model=model,
            optimizer=optimizer,
            step=step
        )
        
        # Track interventions
        if action["action"] != "none":
            interventions.append({
                "step": step,
                "action": action["action"],
                "lambda_value": lambda_value
            })
        
        # Handle rollback - rebuild computation graph
        if action["action"] == "rollback":
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
            for p in model.parameters():
                p.grad = None
        
        # Fresh forward pass for optimization (needed after compute_lambda_metric calls backward)
        fresh_inputs = torch.randn(batch_size, 32).to(device)
        fresh_outputs = model(fresh_inputs)
        loss = fresh_outputs.mean()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return {
        "lambda_history": lambda_history,
        "loss_history": loss_history,
        "interventions": interventions,
        "threshold": threshold
    }


def plot_results(results: Dict, output_dir: str = "plots"):
    """
    Generates visualization plots from demo results.
    
    Args:
        results: Dictionary returned by run_demo()
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    lambda_history = results["lambda_history"]
    loss_history = results["loss_history"]
    threshold = results["threshold"]
    interventions = results["interventions"]
    
    # Plot 1: Lambda curve
    plt.figure(figsize=(12, 6))
    steps = range(len(lambda_history))
    plt.plot(steps, lambda_history, linewidth=2, label='λ(t) - Stability Metric', color='#2E86AB')
    plt.axhline(y=threshold, color='#A23B72', linestyle='--', linewidth=2, label='Threshold')
    
    # Mark interventions
    rollback_steps = [i["step"] for i in interventions if i["action"] == "rollback"]
    if rollback_steps:
        plt.scatter(
            rollback_steps,
            [lambda_history[i] for i in rollback_steps],
            color='#F18F01',
            s=100,
            marker='o',
            label='Rollback + LR Reduction',
            zorder=5
        )
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('λ(t) - Instability Metric', fontsize=12)
    plt.title('CandorFlow Training Stability Monitor (Simplified Demo)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lambda_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Stability phases
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(lambda_history))
    
    # Create phase colors
    stable = np.array(lambda_history) < threshold * 0.5
    warning = (np.array(lambda_history) >= threshold * 0.5) & (np.array(lambda_history) < threshold)
    unstable = np.array(lambda_history) >= threshold
    
    # Plot with color-coded background
    plt.fill_between(steps, 0, threshold * 0.5, alpha=0.2, color='green', label='Stable Zone')
    plt.fill_between(steps, threshold * 0.5, threshold, alpha=0.2, color='orange', label='Warning Zone')
    plt.fill_between(steps, threshold, max(lambda_history) * 1.1, alpha=0.2, color='red', label='Unstable Zone')
    
    # Plot lambda curve
    plt.plot(steps, lambda_history, linewidth=2, color='black', label='λ(t)', zorder=3)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('λ(t)', fontsize=12)
    plt.title('Training Stability Phases', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.ylim(0, max(lambda_history) * 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_phases.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plots to {output_dir}/")
    print(f"  - lambda_curve.png")
    print(f"  - stability_phases.png")

