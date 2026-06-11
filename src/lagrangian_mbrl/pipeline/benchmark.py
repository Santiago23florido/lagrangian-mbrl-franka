"""The benchmark orchestrator: HPO → multi-seed retrain → comparable report.

For every model in the config this:

1. **tunes** hyper-parameters on one tuning seed (:mod:`.hpo`),
2. **retrains** the winning configuration across many seeds (the same seeds and
   the same per-seed data for every model — the fairness control), and
3. **aggregates** held-out acceleration MSE across seeds with a bootstrap
   confidence interval, writing a comparable JSON + CSV summary and a bar plot.

Run it once and leave it overnight; everything is CPU-friendly on the analytic
sources. Swap ``source="franka:<log.npz>"`` once Isaac Lab data exists — no other
change is needed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lagrangian_mbrl.pipeline.data import train_val_datasets
from lagrangian_mbrl.pipeline.experiment import run_single
from lagrangian_mbrl.pipeline.hpo import tune


@dataclass
class BenchmarkConfig:
    source: str = "two_link"                       # analytic name or "franka:<path.npz>"
    models: tuple[str, ...] = ("delan", "mlp_ensemble")
    n_train: int = 256                             # small N highlights the prior's edge
    n_val: int = 2048
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)       # >= 5 for credible CIs
    hpo_trials: int = 20
    hpo_epochs: int = 150                          # short epochs while searching
    train_epochs: int = 600                        # full epochs for the final retrain
    dtype: str = "float64"
    n_bootstrap: int = 10_000
    out_dir: str = "logs/benchmarks"


def _bootstrap_ci(
    values: np.ndarray, n_boot: int, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    if values.size <= 1:
        v = float(values.mean()) if values.size else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_benchmark(
    cfg: BenchmarkConfig | None = None, *, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run the full HPO + multi-seed comparison; save artifacts; return results."""
    cfg = cfg or BenchmarkConfig()
    dtype = getattr(torch, cfg.dtype)
    out = Path(out_dir or cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"config": asdict(cfg), "models": {}}

    for model_name in cfg.models:
        # --- 1. HPO on the first seed's data ---
        tr0, va0, dof = train_val_datasets(
            cfg.source, cfg.n_train, cfg.n_val, seed=cfg.seeds[0], dtype=dtype
        )
        hpo = tune(
            model_name,
            tr0,
            va0,
            dof,
            n_trials=cfg.hpo_trials,
            epochs=cfg.hpo_epochs,
            seed=cfg.seeds[0],
            dtype=dtype,
        )
        best_hp = hpo["best_hp"]

        # --- 2. retrain the winner over all seeds (fresh per-seed data) ---
        per_seed: list[dict[str, Any]] = []
        for seed in cfg.seeds:
            tr, va, dof = train_val_datasets(
                cfg.source, cfg.n_train, cfg.n_val, seed=seed, dtype=dtype
            )
            res = run_single(
                model_name, best_hp, tr, va, dof,
                epochs=cfg.train_epochs, seed=seed, dtype=dtype,
            )
            per_seed.append(res)

        # --- 3. aggregate with a bootstrap CI ---
        mses = np.array([r["best_val_accel_mse"] for r in per_seed], dtype=float)
        ci_lo, ci_hi = _bootstrap_ci(mses, cfg.n_bootstrap)
        results["models"][model_name] = {
            "best_hp": best_hp,
            "hpo_backend": hpo["backend"],
            "hpo_best_value": hpo["best_value"],
            "num_params": per_seed[0]["num_params"],
            "accel_mse_mean": float(mses.mean()),
            "accel_mse_std": float(mses.std(ddof=1)) if mses.size > 1 else 0.0,
            "accel_mse_ci95": [ci_lo, ci_hi],
            "per_seed_accel_mse": mses.tolist(),
            "seeds": list(cfg.seeds),
        }

    _write_summary_csv(results, out / "summary.csv")
    (out / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    _maybe_plot(results, out / "accel_mse_comparison.png")
    return results


def _write_summary_csv(results: dict[str, Any], path: Path) -> None:
    rows = []
    for name, m in results["models"].items():
        rows.append(
            {
                "model": name,
                "accel_mse_mean": m["accel_mse_mean"],
                "accel_mse_std": m["accel_mse_std"],
                "ci95_lo": m["accel_mse_ci95"][0],
                "ci95_hi": m["accel_mse_ci95"][1],
                "num_params": m["num_params"],
            }
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _maybe_plot(results: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return
    names = list(results["models"].keys())
    means = [results["models"][n]["accel_mse_mean"] for n in names]
    cis = [results["models"][n]["accel_mse_ci95"] for n in names]
    yerr = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis)]).T
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=yerr, capsize=6, color=["C0", "C1", "C2", "C3"][: len(names)])
    ax.set_yscale("log")
    ax.set_ylabel("held-out acceleration MSE (mean ± 95% CI)")
    ax.set_title(f"Dynamics benchmark — {results['config']['source']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
