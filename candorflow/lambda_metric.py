"""
Lambda Metric: Simplified Training Stability Indicator
=======================================================

This module implements a SIMPLIFIED surrogate metric λ(t) for detecting
training instabilities. This is a toy implementation for demonstration purposes.

WHAT THIS IS:
- A basic gradient norm variance metric
- Simple, interpretable, educational

WHAT THIS IS NOT (Proprietary Features Excluded):
- Universal scaling law
- Reflexive ridge computation
- Cross-domain invariants
- Jacobian-based analysis
- Multi-signal fusion
- Domain-general early warning system

For the full proprietary CandorFlow metric, please contact the authors.
"""

import torch
import numpy as np
from typing import List, Optional


def compute_lambda_metric(
    model: torch.nn.Module,
    loss: torch.Tensor,
    history_window: int = 10,
    gradient_history: Optional[List[float]] = None
) -> float:
    """
    Compute a simplified λ(t) stability metric based on gradient norm variance.
    
    This is a TOY IMPLEMENTATION for demonstration purposes only.
    
    The metric uses gradient norm variance as a proxy for training instability.
    High variance suggests the optimization landscape is becoming unstable.
    
    Args:
        model: PyTorch model
        loss: Current loss tensor (with grad_fn attached)
        history_window: Number of past gradient norms to track
        gradient_history: List of recent gradient norms (modified in-place)
    
    Returns:
        lambda_value: Stability metric (higher = more unstable)
    
    Note:
        This simplified version does NOT include:
        - Jacobian spectral analysis
        - Reflexive decay computation
        - Universal scaling laws
        - Multi-signal fusion
        - Cross-domain invariants
    """
    # Compute gradients
    model.zero_grad()
    if loss.requires_grad:
        loss.backward()
    
    # Calculate gradient norm
    grad_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
            param_count += 1
    
    if param_count > 0:
        grad_norm = np.sqrt(grad_norm)
    else:
        grad_norm = 0.0
    
    # Track gradient history
    if gradient_history is not None:
        gradient_history.append(grad_norm)
        if len(gradient_history) > history_window:
            gradient_history.pop(0)
        
        # Compute variance as instability proxy
        if len(gradient_history) >= 3:
            grad_variance = np.var(gradient_history)
            grad_mean = np.mean(gradient_history)
            
            # Normalized variance (coefficient of variation squared)
            if grad_mean > 1e-8:
                lambda_value = grad_variance / (grad_mean ** 2)
            else:
                lambda_value = 0.0
        else:
            lambda_value = 0.0
    else:
        lambda_value = 0.0
    
    return lambda_value
