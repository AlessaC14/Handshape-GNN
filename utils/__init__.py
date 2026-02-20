"""
Utility functions for handshape recognition
"""

from .metrics import (
    compute_classification_metrics,
    calculate_finger_independence,
    calculate_thumb_effort,
    calculate_handshape_distance,
    contrastive_loss
)
from .visualization import (
    plot_confusion_matrix,
    plot_training_history,
    plot_class_distribution,
    plot_metric_comparison
)

__all__ = [
    'compute_classification_metrics',
    'calculate_finger_independence',
    'calculate_thumb_effort',
    'calculate_handshape_distance',
    'contrastive_loss',
    'plot_confusion_matrix',
    'plot_training_history',
    'plot_class_distribution',
    'plot_metric_comparison'
]
