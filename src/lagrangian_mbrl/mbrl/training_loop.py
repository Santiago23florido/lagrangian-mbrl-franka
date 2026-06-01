"""Model-based RL training loop.

Implements the generic MBRL outer loop that is shared by the method and the
unstructured baseline — only the injected ``dynamics_model`` differs, which is
exactly the experimental control described in ``PROJECT_PLAN.md`` §5.

The loop (Dyna / MBPO-style, with an MPC option):

    1.  Collect transitions from the (Isaac Lab) environment under the current
        policy / planner; add to the replay buffer.
    2.  Fit the dynamics model on the buffer (DeLaN or MLP-ensemble).
    3.  Improve the policy *inside the learned model*:
          (a) MPC  : plan with MPPI/CEM over short horizons (good first target,
                     no policy network needed), or
          (b) Dyna : generate short model rollouts and run an off-policy RL
                     update (e.g. SAC) on the mixed real+model data (MBPO).
    4.  Evaluate; log sample-efficiency metrics; repeat.

References
----------
- Janner et al. "MBPO." NeurIPS 2019.
- Chua et al. "PETS." NeurIPS 2018.
- Williams et al. "Model-Predictive Path Integral Control (MPPI)." 2017.
- Luo et al. "Algorithmic Framework for MBRL with Theoretical Guarantees." 2019.

TODO (Phase 2–3, see PROJECT_PLAN.md):
  [ ] Replay buffer with normalization stats.
  [ ] Dynamics fit step (early stopping on held-out model MSE).
  [ ] MPC planner (MPPI/CEM) operating on the dynamics-model interface.
  [ ] Optional Dyna/MBPO branch with SAC.
  [ ] Per-iteration eval + W&B/TensorBoard logging (git SHA, seed, config hash).
  [ ] Checkpointing / resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor, nn


class DynamicsModel(Protocol):
    """Structural interface every dynamics model must satisfy.

    Both :class:`~lagrangian_mbrl.models.DeepLagrangianNetwork` and
    :class:`~lagrangian_mbrl.models.MLPDynamics` conform to this, so the loop is
    agnostic to which one it is given.
    """

    def predict_next_state(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def loss(self, batch: dict[str, Tensor]) -> Tensor: ...


@dataclass
class MBRLConfig:
    total_env_steps: int = 1_000_000
    warmup_random_steps: int = 5_000
    model_train_freq: int = 1_000      # env steps between dynamics refits
    model_train_epochs: int = 50
    planner: str = "mppi"              # "mppi" | "cem" | "dyna_sac"
    planning_horizon: int = 15
    num_eval_episodes: int = 10
    eval_freq: int = 10_000
    seed: int = 0
    device: str = "cuda"
    log: dict[str, Any] = field(default_factory=lambda: {"wandb": True, "tensorboard": True})


class MBRLTrainer:
    """Drives the model-based RL loop for a Franka task.

    Parameters
    ----------
    env:
        An Isaac Lab Franka environment wrapper (see
        :mod:`lagrangian_mbrl.envs`).
    dynamics_model:
        Any object satisfying :class:`DynamicsModel` (DeLaN or MLP ensemble).
    config:
        :class:`MBRLConfig`.
    """

    def __init__(self, env: Any, dynamics_model: nn.Module, config: MBRLConfig | None = None) -> None:
        self.env = env
        self.dynamics_model = dynamics_model
        self.config = config or MBRLConfig()
        # TODO: build replay buffer, optimizer, planner, logger.
        raise NotImplementedError("MBRLTrainer not yet implemented — see TODOs.")

    def collect(self, num_steps: int) -> None:
        """Step the environment and append transitions to the buffer. TODO."""
        raise NotImplementedError

    def fit_dynamics(self) -> dict[str, float]:
        """Train ``self.dynamics_model`` on the buffer; return metrics. TODO."""
        raise NotImplementedError

    def improve_policy(self) -> dict[str, float]:
        """Plan/optimize in the learned model (MPC or Dyna). TODO."""
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        """Roll out the greedy policy in the real env; return eval metrics. TODO."""
        raise NotImplementedError

    def train(self) -> None:
        """Run the full outer loop until ``total_env_steps``. TODO."""
        raise NotImplementedError
