#!/usr/bin/env python
"""Run the dynamics-learning benchmark — leave it running overnight.

For every model: tune hyper-parameters, retrain across seeds, and write a
comparable report (JSON + CSV + plot) under a timestamped log directory.

Examples
--------
    # Default config (analytic two-link arm, DeLaN vs MLP-ensemble, 5 seeds):
    python scripts/run_benchmark.py

    # Use a config file and override a few fields:
    python scripts/run_benchmark.py --config configs/benchmark/dynamics_offline.yaml \
        --seeds 0 1 2 3 4 5 6 7 --hpo-trials 40

    # Once Isaac Lab logs exist, point at them (no other change needed):
    python scripts/run_benchmark.py --source franka:logs/franka_transitions.npz

CPU-only; no Isaac Sim required. Install Optuna for smarter HPO
(`pip install optuna`); otherwise a built-in random search is used.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from lagrangian_mbrl.pipeline.benchmark import BenchmarkConfig, run_benchmark


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamics-learning benchmark (HPO + multi-seed).")
    p.add_argument("--config", type=str, default=None, help="YAML config (overrides defaults).")
    p.add_argument("--source", type=str, default=None, help="Data source, e.g. two_link or franka:<path>.")
    p.add_argument("--models", nargs="+", default=None, help="Models to compare.")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds for the multi-seed retrain.")
    p.add_argument("--n-train", type=int, default=None, help="Train-set size.")
    p.add_argument("--hpo-trials", type=int, default=None, help="HPO trials per model.")
    p.add_argument("--train-epochs", type=int, default=None, help="Epochs for the final retrain.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: timestamped).")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> BenchmarkConfig:
    fields = {f.name for f in dataclasses.fields(BenchmarkConfig)}
    values: dict[str, Any] = {}
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        unknown = set(loaded) - fields
        if unknown:
            raise KeyError(f"Unknown config keys {sorted(unknown)}; valid: {sorted(fields)}")
        values.update(loaded)
    # CLI overrides win over the file.
    for key, val in {
        "source": args.source,
        "models": args.models,
        "seeds": args.seeds,
        "n_train": args.n_train,
        "hpo_trials": args.hpo_trials,
        "train_epochs": args.train_epochs,
    }.items():
        if val is not None:
            values[key] = val
    return BenchmarkConfig(**values)


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    out_dir = args.out or str(Path(cfg.out_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("[INFO] Benchmark config:")
    print(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False, default_flow_style=False))
    print(f"[INFO] Writing results to: {out_dir}")

    results = run_benchmark(cfg, out_dir=out_dir)

    print("\n=== Comparable summary (held-out acceleration MSE) ===")
    for name, m in results["models"].items():
        lo, hi = m["accel_mse_ci95"]
        print(
            f"  {name:>14s}: {m['accel_mse_mean']:.3e} "
            f"[95% CI {lo:.3e}, {hi:.3e}]  ({m['num_params']:,} params, HPO={m['hpo_backend']})"
        )
    print(f"\n[INFO] Full report + plot under: {out_dir}")


if __name__ == "__main__":
    main()
