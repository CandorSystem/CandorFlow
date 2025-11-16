"""
CandorFlow: Simplified Public Demo
====================================

This package contains a simplified, illustrative implementation of training
stability monitoring inspired by the CandorFlow framework.

WARNING: This is NOT the full proprietary CandorFlow system. Many advanced
features, algorithms, and optimizations are excluded.

For the complete system, please contact the authors.
"""

from .version import __version__
from .lambda_metric import compute_lambda_metric, compute_lambda_metric_simple
from .stability_controller import StabilityController
from .utils import set_seed, save_checkpoint, load_checkpoint, get_logger

__all__ = [
    "__version__",
    "compute_lambda_metric",
    "compute_lambda_metric_simple",
    "StabilityController",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "get_logger",
]

