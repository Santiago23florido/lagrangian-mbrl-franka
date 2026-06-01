"""Experiment logging to Weights & Biases and TensorBoard.

Every run must record enough to be reproduced and aggregated: git SHA, resolved
config, seed, and hardware (see ``docs/experiments_protocol.md``). This module
provides a thin facade so the training code logs once and fans out to both
backends.

TODO (Phase 0–2, see PROJECT_PLAN.md):
  [ ] Lazy-init W&B and/or a TensorBoard SummaryWriter from config.
  [ ] Capture git SHA, dirty flag, resolved config (OmegaConf), seed, GPU name.
  [ ] `log_scalars(step, **values)` fanning out to both backends.
  [ ] `save_checkpoint(...)` / `log_artifact(...)`.
  [ ] `finish()` to flush/close cleanly.
"""

from __future__ import annotations

from typing import Any


class ExperimentLogger:
    """Facade over W&B + TensorBoard with run-metadata capture."""

    def __init__(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        log_dir: str = "logs",
    ) -> None:
        self.run_name = run_name
        self.config = config or {}
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        # TODO: capture git SHA / config / hardware and init backends.
        raise NotImplementedError("ExperimentLogger not yet implemented — see TODOs.")

    def log_scalars(self, step: int, **values: float) -> None:
        """Log scalar metrics at ``step`` to all active backends. TODO."""
        raise NotImplementedError

    def save_checkpoint(self, state: dict[str, Any], name: str) -> str:
        """Persist a checkpoint under the run dir; return its path. TODO."""
        raise NotImplementedError

    def finish(self) -> None:
        """Flush and close all backends. TODO."""
        raise NotImplementedError
