"""
CandorFlow Training Demo: Simplified Example
=============================================

This script demonstrates the basic usage of the CandorFlow stability monitoring
system on a toy training task.

IMPORTANT: This is a SIMPLIFIED demonstration using placeholder implementations.
The full proprietary CandorFlow system includes many advanced features not
shown here.

This demo:
- Trains a small transformer model on dummy data
- Computes the λ(t) stability metric at each step
- Intentionally causes training instability
- Demonstrates early warning detection
- Shows automatic rollback and learning rate reduction
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from candorflow import (
    compute_lambda_metric,
    StabilityController,
    set_seed,
    get_logger
)


class SimpleClassifier(nn.Module):
    """A simple classifier built on a small transformer."""
    
    def __init__(self, hidden_size=256, num_classes=2):
        super().__init__()
        # Use a tiny configuration for demo purposes
        config = AutoConfig.from_pretrained("distilbert-base-uncased")
        config.hidden_size = hidden_size
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        
        try:
            self.encoder = AutoModel.from_config(config)
        except:
            # Fallback to simple MLP if transformer fails
            self.encoder = nn.Sequential(
                nn.Linear(768, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
        
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        if hasattr(self.encoder, 'config'):
            # Transformer encoder
            outputs = self.encoder(inputs_embeds=x)
            if hasattr(outputs, 'last_hidden_state'):
                pooled = outputs.last_hidden_state.mean(dim=1)
            else:
                pooled = outputs[0].mean(dim=1)
        else:
            # Simple MLP
            pooled = self.encoder(x.mean(dim=1))
        
        return self.classifier(pooled)


def generate_dummy_data(num_samples=32, seq_len=16, input_dim=768, num_classes=2):
    """Generate dummy training data."""
    X = torch.randn(num_samples, seq_len, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


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
    model = SimpleClassifier(hidden_size=256, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
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
        # Generate batch
        X, y = generate_dummy_data(num_samples=32)
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Compute lambda metric
        lambda_value = compute_lambda_metric(
            model=model,
            loss=loss,
            history_window=10,
            gradient_history=gradient_history
        )
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

