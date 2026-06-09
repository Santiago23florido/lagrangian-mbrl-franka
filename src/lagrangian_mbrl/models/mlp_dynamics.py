"""Unstructured MLP dynamics model — the baseline for the comparison.

This is the control in the central experiment: the *same* MBRL loop, the *same*
planner, only the dynamics class differs. The MLP imposes no physics structure,
so any sample-efficiency gap against :class:`DeepLagrangianNetwork` isolates the
effect of the Lagrangian prior.

We follow standard practice from PETS (Chua et al., NeurIPS 2018) and MBPO
(Janner et al., NeurIPS 2019):
  * predict a Gaussian over the target (mean + log-variance);
  * use an **ensemble** to capture epistemic uncertainty for planning;
  * normalize inputs/outputs.

Two prediction targets are supported behind one class:
  * ``"acceleration"`` — map ``(q, q̇, τ) -> q̈``. Used for the Phase-0 offline
    one-step acceleration-MSE comparison against DeLaN (apples to apples: both
    predict ``q̈`` from the same inputs).
  * ``"delta"`` — map ``(state, action) -> Δstate``. The Phase-2 form used
    inside the MBRL rollout (``predict_next_state``).

References
----------
- Chua et al. "Deep RL in a Handful of Trials using Probabilistic Dynamics
  Models (PETS)." NeurIPS 2018.
- Janner et al. "When to Trust Your Model: Model-Based Policy Optimization
  (MBPO)." NeurIPS 2019.
- Nagabandi et al. "Neural Network Dynamics for Model-Based Deep RL." ICRA 2018.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
}


@dataclass
class MLPDynamicsConfig:
    dof: int = 7                 # generalized coordinates (7-DoF Franka)
    state_dim: int = 14          # e.g. (q, q̇) for the 7-DoF Franka
    action_dim: int = 7          # joint torques / position targets
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: str = "silu"
    probabilistic: bool = True   # predict mean + log-variance (PETS-style)
    ensemble_size: int = 5
    learn_logvar_bounds: bool = True
    target: str = "acceleration"  # "acceleration" | "delta"

    @property
    def in_dim(self) -> int:
        if self.target == "acceleration":
            return 2 * self.dof + self.action_dim  # (q, q̇, τ)
        return self.state_dim + self.action_dim    # (state, action)

    @property
    def out_dim(self) -> int:
        return self.dof if self.target == "acceleration" else self.state_dim


def _build_mlp(
    in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int, activation: str
) -> nn.Sequential:
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden_sizes:
        layers += [nn.Linear(prev, width), act()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MLPDynamics(nn.Module):
    """A single (optionally probabilistic) MLP regressor over the chosen target.

    Implements ``predict_acceleration`` / ``predict_next_state`` so it is
    swappable with :class:`DeepLagrangianNetwork` in the MBRL loop.
    """

    def __init__(self, config: MLPDynamicsConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPDynamicsConfig()
        cfg = self.config
        out = cfg.out_dim * (2 if cfg.probabilistic else 1)
        self.net = _build_mlp(cfg.in_dim, cfg.hidden_sizes, out, cfg.activation)

        # Input/output normalization buffers (fit on the replay/offline buffer).
        self.register_buffer("in_mean", torch.zeros(cfg.in_dim))
        self.register_buffer("in_std", torch.ones(cfg.in_dim))
        self.register_buffer("out_mean", torch.zeros(cfg.out_dim))
        self.register_buffer("out_std", torch.ones(cfg.out_dim))

        if cfg.probabilistic:
            # PETS-style learnable, soft log-variance bounds.
            self.max_logvar = nn.Parameter(torch.full((cfg.out_dim,), 0.5))
            self.min_logvar = nn.Parameter(torch.full((cfg.out_dim,), -10.0))

    # --- normalization --------------------------------------------------------
    def fit_normalization(self, inputs: Tensor, targets: Tensor) -> None:
        """Set normalization stats from data (call before training)."""
        self.in_mean.copy_(inputs.mean(0))
        self.in_std.copy_(inputs.std(0).clamp_min(1e-6))
        self.out_mean.copy_(targets.mean(0))
        self.out_std.copy_(targets.std(0).clamp_min(1e-6))

    def _assemble_input(self, *parts: Tensor) -> Tensor:
        return torch.cat(parts, dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Return ``(mean, logvar)`` of the (de-normalized) target distribution.

        ``logvar`` is ``None`` for a deterministic model.
        """
        xn = (x - self.in_mean) / self.in_std
        out = self.net(xn)
        if not self.config.probabilistic:
            mean = out * self.out_std + self.out_mean
            return mean, None
        mean_n, logvar = out.chunk(2, dim=-1)
        mean = mean_n * self.out_std + self.out_mean
        # Soft-clamp log-variance (in normalized-output space) à la PETS.
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    # --- prediction APIs ------------------------------------------------------
    def predict_acceleration(self, q: Tensor, qd: Tensor, tau: Tensor) -> Tensor:
        assert self.config.target == "acceleration"
        mean, _ = self.forward(self._assemble_input(q, qd, tau))
        return mean

    def predict_next_state(self, state: Tensor, action: Tensor) -> Tensor:
        assert self.config.target == "delta"
        mean, _ = self.forward(self._assemble_input(state, action))
        return state + mean

    # --- training -------------------------------------------------------------
    def _target_and_input(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if self.config.target == "acceleration":
            x = self._assemble_input(batch["q"], batch["qd"], batch["tau"])
            y = batch["qdd"]
        else:
            x = self._assemble_input(batch["state"], batch["action"])
            y = batch["next_state"] - batch["state"]
        return x, y

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Gaussian NLL (probabilistic) or MSE (deterministic) on the target."""
        x, y = self._target_and_input(batch)
        mean, logvar = self.forward(x)
        if logvar is None:
            return F.mse_loss(mean, y)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * ((mean - y) ** 2 * inv_var + logvar).sum(-1).mean()
        # Small penalty pulling the soft log-variance bounds together (PETS).
        nll = nll + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll


@dataclass
class MLPEnsembleConfig:
    member: MLPDynamicsConfig = field(default_factory=MLPDynamicsConfig)
    size: int = 5


class MLPDynamicsEnsemble(nn.Module):
    """Ensemble of :class:`MLPDynamics` for epistemic-uncertainty-aware planning.

    Each member is trained on the same loss (optionally over bootstrapped
    batches); predictions average the member means. For trajectory sampling
    during planning, :meth:`sample_member` selects one member per rollout
    (TS-1 / TS-∞ style).
    """

    def __init__(self, config: MLPEnsembleConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPEnsembleConfig()
        self.members = nn.ModuleList(
            [MLPDynamics(self.config.member) for _ in range(self.config.size)]
        )

    @property
    def target(self) -> str:
        return self.config.member.target

    def fit_normalization(self, inputs: Tensor, targets: Tensor) -> None:
        for m in self.members:
            m.fit_normalization(inputs, targets)

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.stack([m.loss(batch) for m in self.members]).mean()

    def predict_acceleration(self, q: Tensor, qd: Tensor, tau: Tensor) -> Tensor:
        return torch.stack(
            [m.predict_acceleration(q, qd, tau) for m in self.members]
        ).mean(0)

    def predict_next_state(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.stack(
            [m.predict_next_state(state, action) for m in self.members]
        ).mean(0)

    def sample_member(self) -> MLPDynamics:
        idx = int(torch.randint(len(self.members), ()).item())
        return self.members[idx]
