"""Utilities for battle NCA."""

from battle_nca.utils.visualization import (
    render_state,
    render_battle,
    create_animation,
    plot_training_curves,
    visualize_channels
)
from battle_nca.utils.metrics import (
    compute_battle_metrics,
    compute_formation_metrics,
    compute_army_statistics
)

__all__ = [
    "render_state",
    "render_battle",
    "create_animation",
    "plot_training_curves",
    "visualize_channels",
    "compute_battle_metrics",
    "compute_formation_metrics",
    "compute_army_statistics",
]
