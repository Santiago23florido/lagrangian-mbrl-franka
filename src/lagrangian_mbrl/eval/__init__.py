"""Evaluation: sample-efficiency, model-accuracy, and physics-consistency metrics."""

from .metrics import (
    acceleration_mse,
    energy_drift,
    mass_matrix_min_eigenvalue,
    rollout_mse,
    sample_efficiency_curve,
    steps_to_threshold,
)

__all__ = [
    "acceleration_mse",
    "energy_drift",
    "mass_matrix_min_eigenvalue",
    "rollout_mse",
    "sample_efficiency_curve",
    "steps_to_threshold",
]
