"""Evaluation and sample-efficiency metrics.

These quantify the paper's central claims (``PROJECT_PLAN.md`` §5.2):

Sample efficiency (headline)
    - :func:`sample_efficiency_curve` : return vs. environment steps.
    - :func:`steps_to_threshold`      : env steps to reach X% success.

Model accuracy
    - :func:`acceleration_mse` : one-step ``q̈`` prediction error (Phase-0 metric).
    - :func:`rollout_mse`      : H-step rollout error on held-out trajectories.

Physical consistency (does the structure deliver physics?)
    - :func:`energy_drift`               : energy change over an unforced rollout.
    - :func:`mass_matrix_min_eigenvalue` : check ``M(q) ≻ 0`` for DeLaN.

Aggregate across seeds with stratified bootstrap CIs (``rliable``) — never
report a single seed (see ``docs/experiments_protocol.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def _as_tensor(x: Any, dtype: torch.dtype = torch.float64) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _model_dtype(model: Any) -> torch.dtype:
    for p in getattr(model, "parameters", lambda: [])():
        return p.dtype
    return torch.float64


# --- sample efficiency -------------------------------------------------------
def sample_efficiency_curve(
    env_steps: np.ndarray, returns: np.ndarray
) -> dict[str, np.ndarray]:
    """Return a learning curve of return vs. env steps, sorted by steps.

    Aligns and sorts the (steps, return) pairs so they are ready for plotting.
    """
    env_steps = np.asarray(env_steps)
    returns = np.asarray(returns)
    order = np.argsort(env_steps)
    return {"steps": env_steps[order], "return": returns[order]}


def steps_to_threshold(
    env_steps: np.ndarray, success_rate: np.ndarray, threshold: float = 0.9
) -> float | None:
    """Environment steps at which ``success_rate`` first reaches ``threshold``.

    Returns ``None`` if the threshold is never met. This is the primary
    head-to-head sample-efficiency number.
    """
    env_steps = np.asarray(env_steps)
    success_rate = np.asarray(success_rate)
    order = np.argsort(env_steps)
    env_steps, success_rate = env_steps[order], success_rate[order]
    hit = np.nonzero(success_rate >= threshold)[0]
    return float(env_steps[hit[0]]) if hit.size else None


# --- model accuracy ----------------------------------------------------------
@torch.no_grad()
def _predict_acceleration(model: Any, q: Tensor, qd: Tensor, tau: Tensor) -> Tensor:
    # DeLaN needs grad enabled internally even at eval time; it manages that via
    # torch.enable_grad inside forward_dynamics, so no_grad here is safe.
    if hasattr(model, "predict_acceleration"):
        return model.predict_acceleration(q, qd, tau)
    return model.forward_dynamics(q, qd, tau)


def acceleration_mse(model: Any, dataset: dict[str, Any]) -> float:
    """Mean squared error of one-step predicted ``q̈`` on held-out transitions.

    ``dataset`` has keys ``q``, ``qd``, ``tau``, ``qdd``. This is the Phase-0
    exit-criterion metric (DeLaN vs MLP).
    """
    dt = _model_dtype(model)
    q = _as_tensor(dataset["q"], dt)
    qd = _as_tensor(dataset["qd"], dt)
    tau = _as_tensor(dataset["tau"], dt)
    qdd = _as_tensor(dataset["qdd"], dt)
    pred = _predict_acceleration(model, q, qd, tau)
    return float(torch.mean((pred - qdd) ** 2).item())


def rollout_mse(
    model: Any, dataset: dict[str, Any], horizons: tuple[int, ...] = (1, 5, 15)
) -> dict[int, float]:
    """H-step rollout MSE on held-out *trajectories*.

    ``dataset`` holds trajectories: ``q``, ``qd`` shaped ``(N, T, d)`` and the
    applied torque sequence ``tau`` shaped ``(N, T, d)``. We integrate the model
    forward from each trajectory's start under the recorded torques and compare
    the predicted ``(q, q̇)`` to the truth at each requested horizon, reporting
    the compounding error. Requires ``model.predict_next_state(q, qd, tau, dt)``.
    """
    dt_model = _model_dtype(model)
    q = _as_tensor(dataset["q"], dt_model)
    qd = _as_tensor(dataset["qd"], dt_model)
    tau = _as_tensor(dataset["tau"], dt_model)
    dt = float(dataset.get("dt", 1e-2))
    T = q.shape[1]
    max_h = min(max(horizons), T - 1)

    qc, qdc = q[:, 0], qd[:, 0]
    sq_err: dict[int, float] = {}
    for t in range(1, max_h + 1):
        qc, qdc = model.predict_next_state(qc, qdc, tau[:, t - 1], dt)
        if t in horizons:
            err = (qc - q[:, t]) ** 2 + (qdc - qd[:, t]) ** 2
            sq_err[t] = float(torch.mean(err).item())
    return sq_err


# --- physical consistency ----------------------------------------------------
def energy_drift(
    model: Any,
    q0: np.ndarray,
    qd0: np.ndarray,
    steps: int,
    dt: float = 1e-3,
) -> np.ndarray:
    """Total-energy change ``|E_t − E_0|`` over an *unforced* rollout.

    For the true Lagrangian system this is ~0; DeLaN (a conservative vector
    field integrated symplectically) should stay near zero while an unstructured
    MLP typically drifts. Requires ``model.energy`` and ``predict_next_state``.
    """
    dt_model = _model_dtype(model)
    q = _as_tensor(q0, dt_model)
    qd = _as_tensor(qd0, dt_model)
    if q.ndim == 1:
        q, qd = q.unsqueeze(0), qd.unsqueeze(0)
    zero_tau = torch.zeros_like(q)
    with torch.no_grad():
        e0 = model.energy(q, qd)
        drift = [torch.zeros_like(e0)]
        for _ in range(steps):
            q, qd = model.predict_next_state(q, qd, zero_tau, dt)
            drift.append((model.energy(q, qd) - e0).abs())
    return torch.stack(drift, dim=0).cpu().numpy()


def mass_matrix_min_eigenvalue(model: Any, q: np.ndarray) -> np.ndarray:
    """Smallest eigenvalue of the predicted ``M(q)`` over a batch of configs.

    Must be strictly positive for a valid DeLaN model; a useful unit-test and
    diagnostic.
    """
    dt_model = _model_dtype(model)
    qt = _as_tensor(q, dt_model)
    with torch.no_grad():
        M = model.mass_matrix(qt)
        eig = torch.linalg.eigvalsh(M)
    return eig.min(dim=-1).values.cpu().numpy()
