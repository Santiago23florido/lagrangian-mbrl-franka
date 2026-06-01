"""Unstructured MLP dynamics model — the baseline for the comparison.

This is the control in the central experiment: the *same* MBRL loop, the *same*
planner, only the dynamics class differs. The MLP imposes no physics structure,
so any sample-efficiency gap against :class:`DeepLagrangianNetwork` isolates the
effect of the Lagrangian prior.

We follow standard practice from PETS (Chua et al., NeurIPS 2018) and MBPO
(Janner et al., NeurIPS 2019):
  * predict a Gaussian over the *state delta* (mean + log-variance);
  * use an **ensemble** to capture epistemic uncertainty for planning;
  * normalize inputs/outputs.

References
----------
- Chua et al. "Deep RL in a Handful of Trials using Probabilistic Dynamics
  Models (PETS)." NeurIPS 2018.
- Janner et al. "When to Trust Your Model: Model-Based Policy Optimization
  (MBPO)." NeurIPS 2019.
- Nagabandi et al. "Neural Network Dynamics for Model-Based Deep RL." ICRA 2018.

TODO (Phase 2, see PROJECT_PLAN.md):
  [ ] Implement the probabilistic MLP (mean + log-var heads, var clamping).
  [ ] Implement input/output normalization (fit on the replay buffer).
  [ ] Implement the Gaussian NLL training loss.
  [ ] Implement the ensemble wrapper (per-member init, TS-∞/TS-1 sampling).
  [ ] Match capacity to DeLaN fairly when reporting the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class MLPDynamicsConfig:
    state_dim: int = 14          # e.g. (q, q̇) for the 7-DoF Franka
    action_dim: int = 7          # joint torques / position targets
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: str = "silu"
    probabilistic: bool = True   # predict mean + log-variance (PETS-style)
    ensemble_size: int = 5
    learn_logvar_bounds: bool = True


class MLPDynamics(nn.Module):
    """A single (optionally probabilistic) MLP that predicts the state delta.

    Implements the same ``predict_next_state`` interface as
    :class:`DeepLagrangianNetwork` so the two are swappable in the MBRL loop.
    """

    def __init__(self, config: MLPDynamicsConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPDynamicsConfig()
        # TODO: build the MLP trunk + mean head (+ log-var head if probabilistic),
        # plus registered normalization buffers.
        raise NotImplementedError("MLPDynamics not yet implemented — see TODOs.")

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mean_delta, logvar_delta)`` of the next-state distribution."""
        raise NotImplementedError

    def predict_next_state(self, state: Tensor, action: Tensor) -> Tensor:
        """Return the (mean) predicted next state ``s + Δ``. TODO: implement."""
        raise NotImplementedError

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Gaussian negative-log-likelihood (or MSE if deterministic)."""
        raise NotImplementedError


@dataclass
class MLPEnsembleConfig:
    member: MLPDynamicsConfig = field(default_factory=MLPDynamicsConfig)
    size: int = 5


class MLPDynamicsEnsemble(nn.Module):
    """Ensemble of :class:`MLPDynamics` for epistemic-uncertainty-aware planning.

    TODO: implement member management, trajectory sampling (TS-1 / TS-∞), and a
    ``predict_next_state`` that samples a member per step for rollouts.
    """

    def __init__(self, config: MLPEnsembleConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPEnsembleConfig()
        raise NotImplementedError("MLPDynamicsEnsemble not yet implemented.")
