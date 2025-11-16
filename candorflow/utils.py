"""
Utility Functions for CandorFlow
=================================

Helper functions for checkpointing, logging, and reproducibility.
"""

import os
import logging
import random
import numpy as np
import torch
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint to disk.
    
    Args:
        filepath: Path to save checkpoint
        model: PyTorch model
        optimizer: Optional optimizer to save
        metadata: Optional dictionary of metadata to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint from disk.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
    
    Returns:
        metadata: Dictionary of metadata from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint.get("metadata", {})


def get_logger(
    name: str = "candorflow",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

