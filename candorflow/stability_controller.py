"""
Stability Controller: Simplified Training Autopilot
====================================================

This module implements a BASIC stability controller that monitors λ(t) and
takes corrective actions when instability is detected.

This is a TOY IMPLEMENTATION for demonstration purposes only.

WHAT THIS INCLUDES:
- Basic threshold-based monitoring
- Checkpoint rollback on instability
- Learning rate reduction
- Simple logging

WHAT THIS DOES NOT INCLUDE (Proprietary Features):
- Reflexive decay algorithms
- Temporal smoothing with active inference
- Multi-signal fusion
- Dynamic threshold adaptation
- Predictive instability modeling
- Real-time stability engine
- HPC-optimized control loops

For the full proprietary CandorFlow controller, please contact the authors.
"""

import os
import logging
from typing import Dict, Optional, Any
import torch


class StabilityController:
    """
    A simplified stability controller for training loop autopilot.
    
    This controller monitors the λ(t) metric and takes basic corrective actions
    when training becomes unstable.
    
    NOTE: This is a demonstration implementation. The full proprietary system
    includes many advanced features not present here.
    """
    
    def __init__(
        self,
        threshold: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
        lr_reduction_factor: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the stability controller.
        
        Args:
            threshold: Lambda value above which to trigger intervention
            checkpoint_dir: Directory to save checkpoints
            lr_reduction_factor: Factor to reduce learning rate by (0.5 = halve)
            logger: Optional logger instance
        """
        self.threshold = threshold
        self.checkpoint_dir = checkpoint_dir
        self.lr_reduction_factor = lr_reduction_factor
        self.logger = logger or logging.getLogger(__name__)
        
        # State tracking
        self.last_stable_checkpoint: Optional[Dict[str, Any]] = None
        self.interventions_count = 0
        self.lambda_history = []
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.logger.info(f"StabilityController initialized with threshold={threshold}")
    
    def update(
        self,
        lambda_value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int
    ) -> Dict[str, Any]:
        """
        Update controller with current λ(t) value and take action if needed.
        
        Args:
            lambda_value: Current stability metric value
            model: PyTorch model
            optimizer: Optimizer
            step: Current training step
        
        Returns:
            action_dict: Dictionary describing action taken
                - "action": "none" | "warning" | "rollback"
                - "lr_reduced": bool
                - "message": str
        """
        self.lambda_history.append(lambda_value)
        
        # Check if we should save a checkpoint (when stable)
        if lambda_value < self.threshold * 0.5:  # Well below threshold
            self._save_stable_checkpoint(model, optimizer, step)
        
        # Check for instability
        if lambda_value > self.threshold:
            self.interventions_count += 1
            self.logger.warning(
                f"⚠️  INSTABILITY DETECTED at step {step}: λ(t)={lambda_value:.4f} "
                f"(threshold={self.threshold})"
            )
            
            # Attempt rollback if we have a stable checkpoint
            if self.last_stable_checkpoint is not None:
                self._rollback(model, optimizer)
                self._reduce_learning_rate(optimizer)
                
                return {
                    "action": "rollback",
                    "lr_reduced": True,
                    "message": f"Rolled back to stable checkpoint and reduced LR",
                    "lambda_value": lambda_value,
                    "step": step
                }
            else:
                self._reduce_learning_rate(optimizer)
                
                return {
                    "action": "warning",
                    "lr_reduced": True,
                    "message": "No checkpoint available, reduced LR only",
                    "lambda_value": lambda_value,
                    "step": step
                }
        
        return {
            "action": "none",
            "lr_reduced": False,
            "message": "Training stable",
            "lambda_value": lambda_value,
            "step": step
        }
    
    def _save_stable_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int
    ):
        """Save a checkpoint when training is stable."""
        self.last_stable_checkpoint = {
            "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "step": step
        }
        self.logger.debug(f"Saved stable checkpoint at step {step}")
    
    def _rollback(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        """Rollback to last stable checkpoint."""
        if self.last_stable_checkpoint is None:
            self.logger.warning("No checkpoint available for rollback")
            return
        
        # Restore model state
        model_state = {
            k: v.to(next(model.parameters()).device)
            for k, v in self.last_stable_checkpoint["model_state"].items()
        }
        model.load_state_dict(model_state)
        
        # Restore optimizer state
        optimizer.load_state_dict(self.last_stable_checkpoint["optimizer_state"])
        
        step = self.last_stable_checkpoint["step"]
        self.logger.info(f"✓ Rolled back to stable checkpoint from step {step}")
    
    def _reduce_learning_rate(self, optimizer: torch.optim.Optimizer):
        """Reduce learning rate by the specified factor."""
        for param_group in optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = old_lr * self.lr_reduction_factor
            param_group["lr"] = new_lr
            self.logger.info(f"✓ Reduced learning rate: {old_lr:.6f} → {new_lr:.6f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from the controller."""
        return {
            "total_interventions": self.interventions_count,
            "max_lambda": max(self.lambda_history) if self.lambda_history else 0.0,
            "mean_lambda": sum(self.lambda_history) / len(self.lambda_history)
            if self.lambda_history else 0.0,
            "threshold": self.threshold
        }

