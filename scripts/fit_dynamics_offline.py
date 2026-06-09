#!/usr/bin/env python
"""Phase-0 offline dynamics fit: DeLaN vs MLP on held-out acceleration MSE.

Runnable without a simulator — fits both models on an analytic rigid-body
system and reports the head-to-head one-step acceleration MSE (the Phase-0 exit
criterion in PROJECT_PLAN.md §0).

Examples
--------
    # default: 2-link arm, small training set (structure should win clearly)
    python scripts/fit_dynamics_offline.py

    # simple pendulum, save artifacts to logs/
    python scripts/fit_dynamics_offline.py --system pendulum --out-dir logs/phase0

    # sweep the data regime to trace the empirical sample-complexity curve
    python scripts/fit_dynamics_offline.py --n-train 64 --epochs 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lagrangian_mbrl.offline import OfflineFitConfig, run_phase0_comparison


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", default="two_link", choices=["two_link", "pendulum"])
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-val", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="logs/phase0", type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = OfflineFitConfig(
        system=args.system,
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    results = run_phase0_comparison(cfg, out_dir=args.out_dir)

    delan = results["delan_val_accel_mse"]
    mlp = results["mlp_val_accel_mse"]
    print("\n=== Phase 0 — offline dynamics fit ===")
    print(f"system            : {cfg.system} (dof={results['dof']})")
    print(f"train / val size  : {cfg.n_train} / {cfg.n_val}")
    print(f"DeLaN params      : {results['delan']['num_params']:>8d}")
    print(f"MLP   params      : {results['mlp']['num_params']:>8d}")
    print(f"DeLaN val accel MSE: {delan:.6e}")
    print(f"MLP   val accel MSE: {mlp:.6e}")
    print(f"improvement (MLP/DeLaN): {results['improvement_ratio']:.2f}x")
    verdict = "PASS ✅" if results["delan_wins"] else "FAIL ❌"
    print(f"exit criterion (DeLaN < MLP): {verdict}")
    if args.out_dir:
        print(f"artifacts saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
