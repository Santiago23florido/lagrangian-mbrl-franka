"""Tests for the Deep Lagrangian Network — the physics guarantees we rely on.

These encode the *correctness contract* of the structured model. They are
expected to FAIL until the model is implemented (Phase 0–1), hence xfail.

Once implemented, the key invariants to verify (see PROJECT_PLAN.md §2, §7):
  * M(q) is symmetric and positive-definite for all sampled q.
  * Energy is (approximately) conserved over an unforced rollout.
  * On a known analytic system (pendulum / 2-link arm), inverse-dynamics torques
    match the analytic ground truth within tolerance.
"""

import pytest


@pytest.mark.xfail(reason="DeLaN not yet implemented (Phase 0–1).", strict=False)
def test_mass_matrix_is_positive_definite():
    import torch

    from lagrangian_mbrl.models import DeepLagrangianNetwork

    model = DeepLagrangianNetwork()
    q = torch.randn(32, model.config.dof)
    M = model.mass_matrix(q)
    # symmetric
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-5)
    # positive-definite -> all eigenvalues > 0
    eigvals = torch.linalg.eigvalsh(M)
    assert (eigvals > 0).all()


@pytest.mark.xfail(reason="DeLaN not yet implemented (Phase 0–1).", strict=False)
def test_energy_conservation_unforced_pendulum():
    # TODO: instantiate a DeLaN fit to a pendulum, roll out with tau=0, assert
    # |E_t - E_0| stays below tolerance.
    raise NotImplementedError
