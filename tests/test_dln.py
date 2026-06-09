"""Tests for the Deep Lagrangian Network — the physics guarantees we rely on.

These encode the *correctness contract* of the structured model (see
PROJECT_PLAN.md §2, §7):
  * M(q) is symmetric and positive-definite for all sampled q.
  * Forward and inverse dynamics are mutual inverses.
  * Energy is (approximately) conserved over an unforced symplectic rollout.
  * Gradients flow through the (second-order) physics for training.
"""

import torch

from lagrangian_mbrl.models import DeepLagrangianNetwork, DeLaNConfig

torch.set_default_dtype(torch.float64)


def _model(dof: int = 3) -> DeepLagrangianNetwork:
    torch.manual_seed(0)
    return DeepLagrangianNetwork(DeLaNConfig(dof=dof, hidden_sizes=(32, 32)))


def test_mass_matrix_is_symmetric_positive_definite():
    model = _model(dof=4)
    q = torch.randn(32, model.config.dof)
    M = model.mass_matrix(q)
    assert M.shape == (32, 4, 4)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-9)
    eigvals = torch.linalg.eigvalsh(M)
    assert (eigvals > 0).all()
    # epsilon floor keeps it strictly away from singular
    assert eigvals.min() >= model.config.epsilon - 1e-9


def test_forward_inverse_dynamics_are_consistent():
    model = _model(dof=3)
    q = torch.randn(16, 3)
    qd = torch.randn(16, 3)
    qdd = torch.randn(16, 3)
    tau = model.inverse_dynamics(q, qd, qdd)
    qdd_rec = model.forward_dynamics(q, qd, tau)
    assert torch.allclose(qdd, qdd_rec, atol=1e-6)


def test_energy_conservation_unforced_rollout():
    # A conservative field integrated with semi-implicit (symplectic) Euler at a
    # small step should not drift in energy, even for an untrained net.
    model = _model(dof=2)
    q = 0.4 * torch.randn(8, 2)
    qd = 0.4 * torch.randn(8, 2)
    e0 = model.energy(q, qd)
    dt, steps = 1e-3, 200
    zero = torch.zeros_like(q)
    for _ in range(steps):
        q, qd = model.predict_next_state(q, qd, zero, dt)
    drift = (model.energy(q, qd) - e0).abs()
    rel = drift / e0.abs().clamp_min(1e-6)
    assert torch.isfinite(rel).all()
    assert rel.max() < 5e-2


def test_loss_is_differentiable_through_physics():
    model = _model(dof=3)
    batch = {
        "q": torch.randn(16, 3),
        "qd": torch.randn(16, 3),
        "qdd": torch.randn(16, 3),
        "tau": torch.randn(16, 3),
    }
    loss = model.loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads)


def test_predict_acceleration_matches_forward_dynamics():
    model = _model(dof=2)
    q, qd, tau = torch.randn(5, 2), torch.randn(5, 2), torch.randn(5, 2)
    assert torch.allclose(
        model.predict_acceleration(q, qd, tau),
        model.forward_dynamics(q, qd, tau),
    )
