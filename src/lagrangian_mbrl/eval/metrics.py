"""Evaluation and sample-efficiency metrics.

These quantify the paper's central claims (``PROJECT_PLAN.md`` §5.2):

Sample efficiency (headline)
    - :func:`sample_efficiency_curve` : return vs. environment steps.
    - :func:`steps_to_threshold`      : env steps to reach X% success.

Model accuracy
    - :func:`rollout_mse` : 1-step and H-step prediction error on held-out data.

Physical consistency (does the structure deliver physics?)
    - :func:`energy_drift`            : energy change over an unforced rollout.
    - :func:`mass_matrix_min_eigenvalue` : check ``M(q) ≻ 0`` for DeLaN.

Aggregate across seeds with stratified bootstrap CIs (``rliable``) — never
report a single seed (see ``docs/experiments_protocol.md``).

TODO (Phase 2–4, see PROJECT_PLAN.md):
  [ ] Implement each metric below.
  [ ] Add an `aggregate_seeds(...)` helper wrapping rliable for mean ± 95% CI.
  [ ] Add a `make_figures(...)` that turns raw logs into the paper figures so
      every figure is regenerable from disk.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_efficiency_curve(
    env_steps: np.ndarray, returns: np.ndarray
) -> dict[str, np.ndarray]:
    """Return a (monotone-binned) learning curve of return vs. env steps.

    Parameters
    ----------
    env_steps, returns:
        Aligned 1-D arrays of cumulative environment steps and evaluation
        returns (one entry per eval).

    Returns
    -------
    dict with keys ``"steps"`` and ``"return"`` ready for plotting. TODO.
    """
    raise NotImplementedError


def steps_to_threshold(
    env_steps: np.ndarray, success_rate: np.ndarray, threshold: float = 0.9
) -> float | None:
    """Environment steps at which ``success_rate`` first reaches ``threshold``.

    Returns ``None`` if the threshold is never met. This is the primary
    head-to-head sample-efficiency number. TODO.
    """
    raise NotImplementedError


def rollout_mse(
    model: Any, dataset: Any, horizons: tuple[int, ...] = (1, 5, 15)
) -> dict[int, float]:
    """Mean squared error of model rollouts at the given horizons on held-out
    transitions. Reports 1-step and multi-step (compounding) error. TODO."""
    raise NotImplementedError


def energy_drift(model: Any, q0: np.ndarray, qd0: np.ndarray, steps: int) -> np.ndarray:
    """Total-energy change ``|E_t − E_0|`` over an *unforced* rollout.

    For the true Lagrangian system this is ~0; DeLaN should stay near zero while
    an unstructured MLP typically drifts. TODO."""
    raise NotImplementedError


def mass_matrix_min_eigenvalue(model: Any, q: np.ndarray) -> np.ndarray:
    """Smallest eigenvalue of the predicted ``M(q)`` over a batch of configs.

    Must be strictly positive for a valid DeLaN model; a useful unit-test and
    diagnostic. TODO."""
    raise NotImplementedError
