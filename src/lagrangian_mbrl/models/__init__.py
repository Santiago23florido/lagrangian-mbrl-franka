"""Dynamics models.

A common interface (:class:`DynamicsModel`) lets the MBRL loop swap between the
structured Lagrangian model and the unstructured MLP baseline without changing
any planning/policy code — this is the core experimental control.
"""

from .deep_lagrangian_network import DeepLagrangianNetwork
from .mlp_dynamics import MLPDynamics, MLPDynamicsEnsemble

__all__ = ["DeepLagrangianNetwork", "MLPDynamics", "MLPDynamicsEnsemble"]
