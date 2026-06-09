"""Tests for the evaluation metrics (no simulator needed)."""

import numpy as np
import torch

from lagrangian_mbrl.eval.metrics import (
    acceleration_mse,
    energy_drift,
    mass_matrix_min_eigenvalue,
    sample_efficiency_curve,
    steps_to_threshold,
)
from lagrangian_mbrl.envs.analytic_systems import TwoLinkArm
from lagrangian_mbrl.models import DeepLagrangianNetwork, DeLaNConfig

torch.set_default_dtype(torch.float64)


def test_steps_to_threshold():
    steps = np.array([0, 100, 200, 300])
    success = np.array([0.0, 0.5, 0.92, 0.99])
    assert steps_to_threshold(steps, success, 0.9) == 200.0
    assert steps_to_threshold(steps, success, 1.0) is None


def test_sample_efficiency_curve_sorts_by_steps():
    out = sample_efficiency_curve(np.array([200, 0, 100]), np.array([2.0, 0.0, 1.0]))
    assert list(out["steps"]) == [0, 100, 200]
    assert list(out["return"]) == [0.0, 1.0, 2.0]


def test_acceleration_mse_zero_on_perfect_model():
    arm = TwoLinkArm()
    gen = torch.Generator().manual_seed(0)
    data = arm.sample_dataset(64, generator=gen)
    # The analytic system predicts itself exactly -> MSE == 0.
    assert acceleration_mse(arm, data) < 1e-18


def test_mass_matrix_min_eigenvalue_positive():
    model = DeepLagrangianNetwork(DeLaNConfig(dof=3, hidden_sizes=(16, 16)))
    q = np.random.randn(20, 3)
    eig = mass_matrix_min_eigenvalue(model, q)
    assert eig.shape == (20,)
    assert (eig > 0).all()


def test_energy_drift_bounded_for_symplectic_rollout():
    model = DeepLagrangianNetwork(DeLaNConfig(dof=2, hidden_sizes=(16, 16)))
    drift = energy_drift(model, np.zeros((4, 2)) + 0.3, np.zeros((4, 2)) + 0.3, steps=50, dt=1e-3)
    assert drift.shape == (51, 4)
    assert np.isfinite(drift).all()
    assert drift[0].max() == 0.0
