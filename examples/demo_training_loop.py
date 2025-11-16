"""
CandorFlow Training Demo: Simplified Example
=============================================

This script demonstrates the basic usage of the CandorFlow stability monitoring
system on a toy training task.

IMPORTANT: This is a SIMPLIFIED demonstration using placeholder implementations.
The full proprietary CandorFlow system includes many advanced features not
shown here.

This demo:
- Trains a simple MLP model on synthetic data
- Computes the λ(t) stability metric at each step
- Intentionally causes training instability
- Demonstrates early warning detection
- Shows automatic rollback and learning rate reduction
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from candorflow import (
    compute_lambda_metric,
    StabilityController,
    set_seed,
    get_logger
)


# ============================================
# TinyModel — Safe, simple, Colab-friendly MLP
# ============================================
class TinyModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# Synthetic toy data batch (avoids tokenization + embedding mismatches)
def get_batch(batch_size=16, input_dim=32):
    return torch.randn(batch_size, input_dim)


def main():
    """Run the training demo with stability monitoring."""
    
    # Setup
    set_seed(42)
    logger = get_logger("demo", log_file="logs/training_demo.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info("CandorFlow Training Stability Demo")
    logger.info("=" * 60)
    logger.info("NOTE: This is a simplified demonstration.")
    logger.info("The full proprietary system includes many additional features.")
    logger.info("=" * 60)
    
    # Create model
    logger.info("\n📦 Creating model...")
    model = TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create stability controller
    logger.info("🔧 Initializing stability controller...")
    controller = StabilityController(
        threshold=2.0,  # Lambda threshold for intervention
        checkpoint_dir="checkpoints",
        lr_reduction_factor=0.5,
        logger=logger
    )
    
    # Training loop
    num_steps = 50
    gradient_history = []
    lambda_values = []
    actions = []
    
    logger.info(f"\n🚀 Starting training for {num_steps} steps...\n")
    
    for step in range(num_steps):
        # Generate synthetic inputs
        inputs = get_batch().to(device)
        
        # Forward pass through the tiny model
        model.train()
        outputs = model(inputs)
        
        # Simple differentiable loss (mean output)
        loss = outputs.mean()
        
        # Compute lambda metric
        lambda_value = compute_lambda_metric(
            model=model,
            loss=loss,
            history_window=10,
            gradient_history=gradient_history
        )
        
        # ============================================================
        # 🔥 Synthetic instability spike for demonstration purposes
        # NOTE:
        # This spike is NOT part of the proprietary CandorFlow engine.
        # It exists ONLY to make the public demo visually interesting.
        # ============================================================
        if step == 30:
            lambda_value = lambda_value + 3.0   # force spike above threshold
            logger.info("⚠️ Synthetic instability spike injected at step 30 (demo only)")
        
        lambda_values.append(lambda_value)
        
        # Update controller
        action = controller.update(
            lambda_value=lambda_value,
            model=model,
            optimizer=optimizer,
            step=step
        )
        actions.append(action)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # INTENTIONALLY cause instability after step 30
        if step >= 30:
            # Apply large gradient noise to simulate instability
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * 5.0 * (step - 29)
                    param.grad.data += noise
        
        optimizer.step()
        
        # Logging
        if step % 5 == 0 or action["action"] != "none":
            logger.info(
                f"Step {step:3d} | Loss: {loss.item():.4f} | "
                f"λ(t): {lambda_value:.4f} | Action: {action['action']}"
            )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete - Summary")
    logger.info("=" * 60)
    summary = controller.get_summary()
    logger.info(f"Total interventions: {summary['total_interventions']}")
    logger.info(f"Max λ(t): {summary['max_lambda']:.4f}")
    logger.info(f"Mean λ(t): {summary['mean_lambda']:.4f}")
    logger.info(f"Threshold: {summary['threshold']:.4f}")
    
    # Save results for plotting
    results = {
        "lambda_values": lambda_values,
        "actions": actions,
        "summary": summary
    }
    torch.save(results, "plots/training_results.pt")
    logger.info("\n✓ Results saved to plots/training_results.pt")
    logger.info("✓ Run examples/demo_plots.py to visualize the results")


if __name__ == "__main__":
    main()

