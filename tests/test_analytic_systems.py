"""Tests for the analytic rigid-body systems used as the Phase-0 oracle."""

import torch

from lagrangian_mbrl.envs.analytic_systems import Pendulum, TwoLinkArm, make_system

torch.set_default_dtype(torch.float64)


def test_two_link_mass_matrix_is_symmetric_pd():
    arm = TwoLinkArm()
    q = torch.randn(64, 2)
    M = arm.mass_matrix(q)
    assert M.shape == (64, 2, 2)
    assert torch.allclose(M, M.transpose(-1, -2))
    assert (torch.linalg.eigvalsh(M) > 0).all()


def test_forward_inverse_dynamics_roundtrip():
    for system in (Pendulum(), TwoLinkArm()):
        d = system.dof
        q, qd, qdd = torch.randn(32, d), torch.randn(32, d), torch.randn(32, d)
        tau = system.inverse_dynamics(q, qd, qdd)
        qdd_rec = system.forward_dynamics(q, qd, tau)
        assert torch.allclose(qdd, qdd_rec, atol=1e-9)


def test_sample_dataset_schema_and_consistency():
    arm = make_system("two_link")
    gen = torch.Generator().manual_seed(0)
    data = arm.sample_dataset(128, generator=gen)
    assert set(data) == {"q", "qd", "tau", "qdd"}
    for v in data.values():
        assert v.shape == (128, 2)
        assert torch.isfinite(v).all()
    # qdd must be the exact forward dynamics of (q, qd, tau)
    qdd = arm.forward_dynamics(data["q"], data["qd"], data["tau"])
    assert torch.allclose(qdd, data["qdd"], atol=1e-9)
