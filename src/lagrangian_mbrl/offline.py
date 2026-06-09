"""Offline dynamics fitting — the runnable Phase-0 deliverable.

Fits the structured Deep Lagrangian Network and the unstructured MLP baseline on
the *same* offline dataset of ``(q, q̇, τ, q̈)`` transitions and compares their
held-out one-step **acceleration MSE**. This is the Phase-0 exit criterion in
``PROJECT_PLAN.md`` §0:

    "DeLaN trains on offline data and beats an MLP on held-out one-step
     acceleration MSE."

The dataset comes either from an analytic rigid-body system
(:mod:`lagrangian_mbrl.envs.analytic_systems`, no simulator needed) or from
logged Isaac Lab Franka transitions with the same schema. Because rigid-body
dynamics *lie in* the Lagrangian hypothesis class (assumption A1), the structural
prior should win the comparison, especially in the low-data regime — the
empirical face of the sample-complexity argument in §2.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .envs.analytic_systems import make_system
from .eval.metrics import acceleration_mse
from .models import (
    DeepLagrangianNetwork,
    DeLaNConfig,
    MLPDynamics,
    MLPDynamicsConfig,
)
from .utils.seeding import seed_everything


@dataclass
class OfflineFitConfig:
    system: str = "two_link"        # analytic system name (or "franka" via logs)
    n_train: int = 256              # small N highlights the sample-efficiency gap
    n_val: int = 2048
    epochs: int = 800
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"          # float64 conditions the mass-matrix solves
    hidden_sizes: tuple[int, ...] = (128, 128)
    mlp_hidden_sizes: tuple[int, ...] = (256, 256, 256)


def _make_dataset(
    cfg: OfflineFitConfig, n: int, generator: torch.Generator, dtype: torch.dtype
) -> dict[str, Tensor]:
    system = make_system(cfg.system)
    return system.sample_dataset(n, generator=generator, dtype=dtype)


def _iterate_minibatches(
    data: dict[str, Tensor], batch_size: int, generator: torch.Generator
):
    n = data["q"].shape[0]
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        yield {k: v[idx] for k, v in data.items()}


def fit_model(
    model: torch.nn.Module,
    train: dict[str, Tensor],
    val: dict[str, Tensor],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    generator: torch.Generator | None = None,
    eval_every: int = 10,
) -> dict[str, Any]:
    """Train ``model`` on ``train``; track held-out acceleration MSE on ``val``.

    Returns a history dict with ``epoch`` / ``train_loss`` / ``val_accel_mse``
    and the best (lowest) validation acceleration MSE seen.
    """
    generator = generator or torch.Generator()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Fit input/output normalization on the training set.
    if hasattr(model, "fit_normalization"):  # MLP baseline
        x = torch.cat([train["q"], train["qd"], train["tau"]], dim=-1)
        model.fit_normalization(x, train["qdd"])
    if hasattr(model, "fit_input_normalization"):  # DeLaN energy heads
        model.fit_input_normalization(train["q"])

    history = {"epoch": [], "train_loss": [], "val_accel_mse": []}
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        nb = 0
        for batch in _iterate_minibatches(train, batch_size, generator):
            opt.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            epoch_loss += float(loss.item())
            nb += 1
        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            vmse = acceleration_mse(model, val)
            best = min(best, vmse)
            history["epoch"].append(epoch)
            history["train_loss"].append(epoch_loss / max(nb, 1))
            history["val_accel_mse"].append(vmse)
    history["best_val_accel_mse"] = best
    history["final_val_accel_mse"] = history["val_accel_mse"][-1]
    return history


def run_phase0_comparison(
    cfg: OfflineFitConfig | None = None,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit DeLaN and the MLP baseline; return and (optionally) save the results.

    Returns a dict with each model's training history, the head-to-head held-out
    acceleration MSE, and ``delan_wins`` (the Phase-0 exit criterion).
    """
    cfg = cfg or OfflineFitConfig()
    seed_everything(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    torch.set_default_dtype(dtype)
    gen = torch.Generator().manual_seed(cfg.seed)

    train = _make_dataset(cfg, cfg.n_train, gen, dtype)
    val = _make_dataset(cfg, cfg.n_val, gen, dtype)
    dof = train["q"].shape[-1]

    delan = DeepLagrangianNetwork(
        DeLaNConfig(dof=dof, hidden_sizes=cfg.hidden_sizes)
    ).to(dtype)
    mlp = MLPDynamics(
        MLPDynamicsConfig(
            dof=dof,
            state_dim=2 * dof,
            action_dim=dof,
            hidden_sizes=cfg.mlp_hidden_sizes,
            probabilistic=False,
            target="acceleration",
        )
    ).to(dtype)

    results: dict[str, Any] = {"config": asdict(cfg), "dof": dof}
    for name, model in (("delan", delan), ("mlp", mlp)):
        t0 = time.time()
        hist = fit_model(
            model,
            train,
            val,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            generator=gen,
        )
        hist["wall_time_s"] = time.time() - t0
        hist["num_params"] = sum(p.numel() for p in model.parameters())
        results[name] = hist

    delan_mse = results["delan"]["best_val_accel_mse"]
    mlp_mse = results["mlp"]["best_val_accel_mse"]
    results["delan_val_accel_mse"] = delan_mse
    results["mlp_val_accel_mse"] = mlp_mse
    results["improvement_ratio"] = mlp_mse / delan_mse if delan_mse > 0 else float("inf")
    results["delan_wins"] = delan_mse < mlp_mse

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "phase0_offline_results.json").write_text(json.dumps(results, indent=2))
        _maybe_plot(results, out / "phase0_accel_mse.png")
    return results


def _maybe_plot(results: dict[str, Any], path: Path) -> None:
    """Save a val-acceleration-MSE learning curve if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in (("delan", "C0"), ("mlp", "C1")):
        h = results[name]
        ax.plot(h["epoch"], h["val_accel_mse"], label=name.upper(), color=color)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out acceleration MSE")
    ax.set_title(
        f"Phase 0: DeLaN vs MLP ({results['config']['system']}, "
        f"N={results['config']['n_train']})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
