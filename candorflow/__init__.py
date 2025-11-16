"""
CandorFlow Lite — Public API
All proprietary components removed.
"""

from .lambda_metric import compute_lambda_metric as compute_lambda
from .stability_controller import StabilityController

__all__ = [
    "compute_lambda",
    "StabilityController",
]
