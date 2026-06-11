"""Run and score a single (model, hyper-parameters, seed) configuration.

This is the atom the HPO and the multi-seed benchmark are built from: it
instantiates a model from the registry, trains it on a given train set, and
reports held-out metrics. Training reuses the Phase-0 :func:`fit_model` loop so
the offline reproduction and the benchmark share one code path.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from lagrangian_mbrl.eval.metrics import acceleration_mse
from lagrangian_mbrl.offline import fit_model
from lagrangian_mbrl.pipeline.registry import OPTIM_HP_KEYS, build_model
from lagrangian_mbrl.utils.seeding import seed_everything


def _split_hp(hp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate optimizer knobs from architecture knobs."""
    optim = {k: hp[k] for k in OPTIM_HP_KEYS if k in hp}
    arch = {k: v for k, v in hp.items() if k not in OPTIM_HP_KEYS}
    return arch, optim


def run_single(
    model_name: str,
    hp: dict[str, Any],
    train: dict[str, Tensor],
    val: dict[str, Tensor],
    dof: int,
    *,
    epochs: int = 400,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    eval_every: int = 10,
) -> dict[str, Any]:
    """Train ``model_name`` with ``hp`` and return held-out metrics.

    Returns a dict with the headline ``best_val_accel_mse`` (held-out one-step
    acceleration MSE — the comparison metric), the final-epoch value, parameter
    count, wall-clock time, and the resolved hyper-parameters.
    """
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed)
    arch_hp, optim_hp = _split_hp(hp)

    model = build_model(model_name, dof, arch_hp).to(dtype)

    t0 = time.time()
    history = fit_model(
        model,
        train,
        val,
        epochs=epochs,
        batch_size=int(optim_hp.get("batch_size", 64)),
        lr=float(optim_hp.get("lr", 3e-3)),
        weight_decay=float(optim_hp.get("weight_decay", 0.0)),
        generator=gen,
        eval_every=eval_every,
    )
    wall = time.time() - t0

    # Final held-out check (also covers models without a logged curve point).
    model.eval()
    final_mse = acceleration_mse(model, val)

    return {
        "model": model_name,
        "seed": seed,
        "hp": dict(hp),
        "best_val_accel_mse": float(history["best_val_accel_mse"]),
        "final_val_accel_mse": float(min(history["final_val_accel_mse"], final_mse)),
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "wall_time_s": float(wall),
        "epochs": int(epochs),
    }
