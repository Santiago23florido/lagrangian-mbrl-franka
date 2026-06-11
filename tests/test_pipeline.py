"""Tests for the dynamics benchmark pipeline.

These run **without Isaac Sim** on the analytic two-link system (CPU, tiny
budgets), so they exercise the real data → train → HPO → benchmark path end to
end and are safe in CI.
"""

import torch
from torch import nn

from lagrangian_mbrl.pipeline import build_model, model_hp_space
from lagrangian_mbrl.pipeline.benchmark import BenchmarkConfig, run_benchmark
from lagrangian_mbrl.pipeline.data import train_val_datasets
from lagrangian_mbrl.pipeline.experiment import run_single
from lagrangian_mbrl.pipeline.hpo import tune

MODELS = ("delan", "mlp", "mlp_ensemble")


def test_registry_builds_all_models():
    for name in MODELS:
        model = build_model(name, dof=2, hp={"hidden_width": 16, "hidden_depth": 2})
        assert isinstance(model, nn.Module)
        assert sum(p.numel() for p in model.parameters()) > 0
        assert isinstance(model_hp_space(name), dict)


def test_data_source_shapes():
    train, val, dof = train_val_datasets("two_link", n_train=32, n_val=48, seed=0)
    assert dof == 2
    for key in ("q", "qd", "tau", "qdd"):
        assert train[key].shape == (32, 2)
        assert val[key].shape == (48, 2)


def test_same_seed_gives_same_data():
    a, _, _ = train_val_datasets("two_link", 16, 16, seed=3)
    b, _, _ = train_val_datasets("two_link", 16, 16, seed=3)
    assert torch.allclose(a["q"], b["q"])  # fairness control across models


def test_run_single_returns_finite_metric():
    train, val, dof = train_val_datasets("two_link", 64, 128, seed=0)
    res = run_single(
        "delan",
        {"hidden_width": 32, "hidden_depth": 2, "lr": 3e-3, "batch_size": 32},
        train,
        val,
        dof,
        epochs=5,
        seed=0,
    )
    assert torch.isfinite(torch.tensor(res["best_val_accel_mse"]))
    assert res["num_params"] > 0


def test_hpo_random_search_runs():
    train, val, dof = train_val_datasets("two_link", 64, 128, seed=0)
    out = tune("mlp", train, val, dof, n_trials=2, epochs=3, seed=0)
    assert out["best_hp"]
    assert len(out["trials"]) == 2
    assert out["backend"] in ("optuna", "random")


def test_benchmark_smoke(tmp_path):
    cfg = BenchmarkConfig(
        source="two_link",
        models=("mlp",),
        n_train=48,
        n_val=64,
        seeds=(0, 1),
        hpo_trials=2,
        hpo_epochs=3,
        train_epochs=5,
        n_bootstrap=200,
    )
    results = run_benchmark(cfg, out_dir=tmp_path)
    m = results["models"]["mlp"]
    assert len(m["per_seed_accel_mse"]) == 2
    assert m["accel_mse_ci95"][0] <= m["accel_mse_mean"] <= m["accel_mse_ci95"][1]
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "summary.csv").exists()
