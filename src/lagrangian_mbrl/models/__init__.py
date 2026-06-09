"""Dynamics models.

A common interface (:class:`DynamicsModel`) lets the MBRL loop swap between the
structured Lagrangian model and the unstructured MLP baseline without changing
any planning/policy code — this is the core experimental control.
"""

from .deep_lagrangian_network import DeepLagrangianNetwork, DeLaNConfig
from .mlp_dynamics import (
    MLPDynamics,
    MLPDynamicsConfig,
    MLPDynamicsEnsemble,
    MLPEnsembleConfig,
)

__all__ = [
    "DeepLagrangianNetwork",
    "DeLaNConfig",
    "MLPDynamics",
    "MLPDynamicsConfig",
    "MLPDynamicsEnsemble",
    "MLPEnsembleConfig",
]
