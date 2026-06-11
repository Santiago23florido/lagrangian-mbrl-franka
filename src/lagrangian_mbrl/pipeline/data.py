"""Data sources for the dynamics benchmark — one schema, swappable origin.

Every source yields a flat dataset of one-step transitions as a dict of tensors
with keys ``q``, ``qd``, ``tau``, ``qdd`` (each shaped ``(N, dof)``). This is the
exact schema the models and :func:`acceleration_mse` consume, so the *same*
benchmark runs on either origin:

- **analytic** (``"pendulum"``, ``"two_link"``): closed-form rigid-body systems
  with exact ``(q, q̇, τ, q̈)`` — no simulator, runs on CPU tonight.
- **franka** (``"franka:/path/to/log.npz"``): transitions logged from the
  Isaac Lab Franka env (same keys), available once the simulator is up. The
  ``.npz`` must contain arrays ``q, qd, tau, qdd``.

The split helper draws independent train/val sets from one seeded generator, so
every model in a benchmark sees the *same* data for a given seed — the control
that makes the comparison fair.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from lagrangian_mbrl.envs.analytic_systems import make_system

ANALYTIC_SOURCES = ("pendulum", "two_link")


def _load_franka_npz(path: str | Path) -> dict[str, Tensor]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Franka transition log not found: {p}. Log one from Isaac Lab first "
            "(FrankaReachEnv.log_transitions) with arrays q, qd, tau, qdd."
        )
    import numpy as np

    npz = np.load(p)
    missing = {"q", "qd", "tau", "qdd"} - set(npz.files)
    if missing:
        raise KeyError(f"{p} is missing arrays {sorted(missing)} (need q, qd, tau, qdd).")
    return {k: torch.as_tensor(npz[k]) for k in ("q", "qd", "tau", "qdd")}


def _slice(data: dict[str, Tensor], idx: Tensor, dtype: torch.dtype) -> dict[str, Tensor]:
    return {k: v[idx].to(dtype) for k, v in data.items()}


def make_dataset(
    source: str,
    n: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Draw ``n`` transitions from ``source`` (analytic) — ``(q, q̇, τ, q̈)``."""
    if source in ANALYTIC_SOURCES:
        return make_system(source).sample_dataset(n, generator=generator, dtype=dtype)
    raise ValueError(
        f"Unknown analytic source '{source}'. Use one of {ANALYTIC_SOURCES} or a "
        "'franka:<path.npz>' source via train_val_datasets()."
    )


def train_val_datasets(
    source: str,
    n_train: int,
    n_val: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[dict[str, Tensor], dict[str, Tensor], int]:
    """Build seeded, independent train/val sets and return ``(train, val, dof)``.

    For analytic sources both sets are freshly sampled. For a ``franka:<path>``
    source the logged transitions are shuffled and split ``n_train``/``n_val``.
    """
    gen = torch.Generator().manual_seed(seed)

    if source.startswith("franka:"):
        data = _load_franka_npz(source.split(":", 1)[1])
        total = data["q"].shape[0]
        if n_train + n_val > total:
            raise ValueError(
                f"Requested {n_train}+{n_val} transitions but log has only {total}."
            )
        perm = torch.randperm(total, generator=gen)
        train = _slice(data, perm[:n_train], dtype)
        val = _slice(data, perm[n_train : n_train + n_val], dtype)
    else:
        train = make_dataset(source, n_train, generator=gen, dtype=dtype)
        val = make_dataset(source, n_val, generator=gen, dtype=dtype)

    dof = int(train["q"].shape[-1])
    return train, val, dof
