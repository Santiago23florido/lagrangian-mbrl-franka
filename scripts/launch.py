#!/usr/bin/env python
"""Fast experiment launcher for the dynamics benchmark.

Pick a named preset from ``configs/benchmark/experiments/``, optionally tweak a
few fields inline, optionally sweep one axis, and run. No Hydra: presets are
plain YAML, overrides are ``key=value`` pairs.

Examples
--------
    # Run a preset as-is:
    python scripts/launch.py --preset quick

    # Preset + inline overrides (yaml-typed):
    python scripts/launch.py --preset full models='[delan,mlp_ensemble]' seeds='[0,1,2]'

    # Sweep one field -> one run per value (e.g. the sample-efficiency curve):
    python scripts/launch.py --preset sample_efficiency --sweep n_train=64,128,256,512,1024

    # Queue several presets back-to-back overnight:
    python scripts/launch.py --preset quick --preset full

All results land under ``logs/benchmarks/<preset>/<timestamp>[__<sweep>]/``.
CPU-only; no Isaac Sim required.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from lagrangian_mbrl.pipeline.benchmark import BenchmarkConfig, run_benchmark

PRESET_DIR = Path("configs/benchmark/experiments")
_FIELDS = {f.name for f in dataclasses.fields(BenchmarkConfig)}


def _load_preset(name: str) -> dict[str, Any]:
    path = name if name.endswith((".yaml", ".yml")) else str(PRESET_DIR / f"{name}.yaml")
    p = Path(path)
    if not p.exists():
        avail = sorted(q.stem for q in PRESET_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"Preset '{name}' not found ({p}). Available: {avail}")
    return yaml.safe_load(p.read_text()) or {}


def _apply_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be key=value.")
        key, raw = item.split("=", 1)
        if key not in _FIELDS:
            raise KeyError(f"Unknown field '{key}'. Valid: {sorted(_FIELDS)}")
        values[key] = yaml.safe_load(raw)  # yaml types ints/lists/floats for us
    return values


def _run_one(values: dict[str, Any], out_dir: Path, tag: str) -> dict[str, Any]:
    unknown = set(values) - _FIELDS
    if unknown:
        raise KeyError(f"Unknown config keys {sorted(unknown)}.")
    cfg = BenchmarkConfig(**values)
    print(f"\n{'='*70}\n[RUN] {tag}  ->  {out_dir}\n{'='*70}")
    results = run_benchmark(cfg, out_dir=out_dir)
    for name, m in results["models"].items():
        lo, hi = m["accel_mse_ci95"]
        print(f"  {name:>14s}: {m['accel_mse_mean']:.3e} [95% CI {lo:.3e}, {hi:.3e}]")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Launch dynamics-benchmark experiments.")
    p.add_argument("--preset", action="append", default=[], help="Preset name (repeatable to queue).")
    p.add_argument("--sweep", default=None, help="Sweep one field: field=v1,v2,v3")
    p.add_argument("--out", default="logs/benchmarks", help="Base output directory.")
    p.add_argument("overrides", nargs="*", help="Inline key=value overrides.")
    args = p.parse_args()

    if not args.preset:
        p.error("give at least one --preset (e.g. --preset quick)")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(args.out)
    summary: list[tuple[str, dict[str, Any]]] = []

    for preset in args.preset:
        values = _apply_overrides(_load_preset(preset), args.overrides)

        if args.sweep:
            field, raw_vals = args.sweep.split("=", 1)
            if field not in _FIELDS:
                p.error(f"--sweep field '{field}' is not a config field.")
            for raw in raw_vals.split(","):
                val = yaml.safe_load(raw)
                sweep_vals = dict(values, **{field: val})
                tag = f"{preset}__{field}{val}"
                out_dir = base / preset / f"{stamp}__{field}{val}"
                summary.append((tag, _run_one(sweep_vals, out_dir, tag)))
        else:
            out_dir = base / preset / stamp
            summary.append((preset, _run_one(values, out_dir, preset)))

    print(f"\n{'#'*70}\n# Launched {len(summary)} run(s). Results under: {base}\n{'#'*70}")
    for tag, results in summary:
        winners = sorted(results["models"].items(), key=lambda kv: kv[1]["accel_mse_mean"])
        best = winners[0][0]
        print(f"  {tag:<32s} best: {best}")


if __name__ == "__main__":
    main()
