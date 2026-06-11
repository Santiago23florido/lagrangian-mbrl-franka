"""Hyper-parameter search for a dynamics model.

Backends
--------
- **Optuna** (TPE sampler) if ``optuna`` is importable — smarter, pruning-ready.
- **random search** otherwise — a dependency-free fallback so the benchmark runs
  tonight without installing anything.

Both minimize the held-out one-step **acceleration MSE** on a single tuning seed
(cheap), over the per-model space from :func:`model_hp_space`. The winning
hyper-parameters are then retrained over many seeds by the benchmark.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch

from lagrangian_mbrl.pipeline.experiment import run_single
from lagrangian_mbrl.pipeline.registry import model_hp_space


def _sample_one(spec: tuple, rng: random.Random) -> Any:
    kind = spec[0]
    if kind == "categorical":
        return rng.choice(spec[1])
    if kind == "int":
        return rng.randint(spec[1], spec[2])
    if kind == "loguniform":
        lo, hi = spec[1], spec[2]
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    raise ValueError(f"Unknown hp spec kind: {kind!r}")


def _sample_space(space: dict[str, tuple], rng: random.Random) -> dict[str, Any]:
    return {name: _sample_one(spec, rng) for name, spec in space.items()}


def _objective_value(
    model_name: str,
    hp: dict[str, Any],
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    dof: int,
    *,
    epochs: int,
    seed: int,
    dtype: torch.dtype,
) -> float:
    res = run_single(model_name, hp, train, val, dof, epochs=epochs, seed=seed, dtype=dtype)
    mse = res["best_val_accel_mse"]
    # Guard against NaN/inf so search keeps making progress.
    return mse if math.isfinite(mse) else float("1e9")


def tune(
    model_name: str,
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    dof: int,
    *,
    n_trials: int = 20,
    epochs: int = 150,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Search hyper-parameters for ``model_name``; return best hp + trace.

    Returns ``{"model", "best_hp", "best_value", "n_trials", "backend",
    "trials": [{"hp", "value"}, ...]}``.
    """
    space = model_hp_space(model_name)

    try:
        import optuna

        backend = "optuna"
    except Exception:  # pragma: no cover - exercised only when optuna is absent
        optuna = None
        backend = "random"

    trials: list[dict[str, Any]] = []

    if backend == "optuna":
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            hp: dict[str, Any] = {}
            for name, spec in space.items():
                if spec[0] == "categorical":
                    hp[name] = trial.suggest_categorical(name, spec[1])
                elif spec[0] == "int":
                    hp[name] = trial.suggest_int(name, spec[1], spec[2])
                elif spec[0] == "loguniform":
                    hp[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            value = _objective_value(
                model_name, hp, train, val, dof, epochs=epochs, seed=seed, dtype=dtype
            )
            trials.append({"hp": hp, "value": value})
            return value

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_hp, best_value = dict(study.best_params), float(study.best_value)
    else:
        rng = random.Random(seed)
        best_hp, best_value = {}, float("inf")
        for _ in range(n_trials):
            hp = _sample_space(space, rng)
            value = _objective_value(
                model_name, hp, train, val, dof, epochs=epochs, seed=seed, dtype=dtype
            )
            trials.append({"hp": hp, "value": value})
            if value < best_value:
                best_hp, best_value = hp, value

    return {
        "model": model_name,
        "best_hp": best_hp,
        "best_value": best_value,
        "n_trials": n_trials,
        "backend": backend,
        "trials": trials,
    }
