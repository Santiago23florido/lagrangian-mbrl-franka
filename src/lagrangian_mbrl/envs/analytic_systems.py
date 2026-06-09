"""Analytic rigid-body systems with *known* Lagrangian dynamics.

Phase 0 needs an offline dataset of ``(q, q̇, τ, q̈)`` transitions to fit the
dynamics models on. The plan sources this from logged Isaac Lab Franka rollouts
(see :mod:`lagrangian_mbrl.envs.franka_reach_wrapper`), but Isaac Sim is a heavy,
GPU-only dependency. To make Phase 0 *runnable and unit-testable anywhere*, we
also provide closed-form rigid-body systems whose exact equations of motion

    M(q) q̈ + c(q, q̇) + g(q) = τ

are known analytically. These serve three purposes:

  1. Ground-truth supervision for the offline DeLaN-vs-MLP acceleration-MSE
     comparison (the Phase-0 exit criterion).
  2. A correctness oracle for the DeLaN unit tests (inverse dynamics must match
     the analytic torques on a known system).
  3. A clean A1-satisfying regime (pure Lagrangian, no contact) where the
     structural prior is *expected* to help — exactly the setting the theory in
     ``PROJECT_PLAN.md`` §2 targets.

When Isaac Lab is available, ``FrankaReachEnv.log_transitions`` dumps a dataset
with the same schema, so the offline-fit script is source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class RigidBodySystem:
    """Interface for an analytic ``M(q) q̈ + c(q,q̇) + g(q) = τ`` system."""

    dof: int

    def mass_matrix(self, q: Tensor) -> Tensor:  # (B, d, d)
        raise NotImplementedError

    def coriolis_forces(self, q: Tensor, qd: Tensor) -> Tensor:  # (B, d)
        raise NotImplementedError

    def gravity_forces(self, q: Tensor) -> Tensor:  # (B, d)
        raise NotImplementedError

    def inverse_dynamics(self, q: Tensor, qd: Tensor, qdd: Tensor) -> Tensor:
        """Analytic torques ``τ = M q̈ + c + g``."""
        M = self.mass_matrix(q)
        return (
            torch.einsum("...ij,...j->...i", M, qdd)
            + self.coriolis_forces(q, qd)
            + self.gravity_forces(q)
        )

    def forward_dynamics(self, q: Tensor, qd: Tensor, tau: Tensor) -> Tensor:
        """Analytic accelerations ``q̈ = M⁻¹ (τ − c − g)``."""
        M = self.mass_matrix(q)
        rhs = (tau - self.coriolis_forces(q, qd) - self.gravity_forces(q)).unsqueeze(-1)
        return torch.linalg.solve(M, rhs).squeeze(-1)

    def sample_dataset(
        self,
        n: int,
        *,
        q_scale: float = 3.14159,
        qd_scale: float = 2.0,
        qdd_scale: float = 5.0,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> dict[str, Tensor]:
        """Sample ``n`` i.i.d. transitions ``(q, q̇, q̈, τ)``.

        We sample ``(q, q̇, q̈)`` uniformly on a compact set (assumption A2 in
        ``PROJECT_PLAN.md``) and compute the *exact* torque ``τ`` from inverse
        dynamics — mirroring a logged robot transition, where ``(q, q̇, q̈)`` are
        observed and ``τ`` is the applied command. Sampling accelerations (rather
        than torques) keeps the regression target ``q̈`` well-scaled and avoids
        the ill-conditioning of inverting near-singular mass matrices.

        Uses double precision by default — the mass-matrix solves and second
        derivatives are better conditioned in float64 (see the risk table,
        "Cholesky / 2nd derivatives").
        """
        d = self.dof
        g = generator

        def _u(scale: float) -> Tensor:
            return (torch.rand(n, d, generator=g, dtype=dtype) * 2 - 1) * scale

        q, qd, qdd = _u(q_scale), _u(qd_scale), _u(qdd_scale)
        tau = self.inverse_dynamics(q, qd, qdd)
        return {"q": q, "qd": qd, "tau": tau, "qdd": qdd}


@dataclass
class PendulumParams:
    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81


class Pendulum(RigidBodySystem):
    """Simple 1-DoF pendulum: ``m l² θ̈ + m g l sin θ = τ`` (no Coriolis)."""

    dof = 1

    def __init__(self, params: PendulumParams | None = None) -> None:
        self.p = params or PendulumParams()

    def mass_matrix(self, q: Tensor) -> Tensor:
        ml2 = self.p.mass * self.p.length**2
        return torch.full((*q.shape[:-1], 1, 1), ml2, dtype=q.dtype, device=q.device)

    def coriolis_forces(self, q: Tensor, qd: Tensor) -> Tensor:
        return torch.zeros_like(q)

    def gravity_forces(self, q: Tensor) -> Tensor:
        p = self.p
        return p.mass * p.gravity * p.length * torch.sin(q)


@dataclass
class TwoLinkArmParams:
    """Planar 2-link arm parameters (textbook acrobot, angles from horizontal)."""

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    lc1: float = 0.5  # COM distance along link 1
    lc2: float = 0.5  # COM distance along link 2
    i1: float = 0.083  # link inertia about COM (~ m l²/12 for a rod)
    i2: float = 0.083
    gravity: float = 9.81


class TwoLinkArm(RigidBodySystem):
    """Planar 2-link manipulator with full configuration-dependent dynamics.

    Standard rigid-body model (Spong & Vidyasagar). Unlike the pendulum, ``M(q)``
    depends on ``q`` and the Coriolis/centrifugal forces are non-trivial, so it
    genuinely exercises the structured terms a DeLaN must learn.
    """

    dof = 2

    def __init__(self, params: TwoLinkArmParams | None = None) -> None:
        self.p = params or TwoLinkArmParams()

    def mass_matrix(self, q: Tensor) -> Tensor:
        p = self.p
        q2 = q[..., 1]
        c2 = torch.cos(q2)
        d1 = (
            p.m1 * p.lc1**2
            + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * c2)
            + p.i1
            + p.i2
        )
        d2 = p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2) + p.i2
        d3 = torch.full_like(q2, p.m2 * p.lc2**2 + p.i2)
        M = torch.stack(
            [torch.stack([d1, d2], -1), torch.stack([d2, d3], -1)], dim=-2
        )
        return M

    def coriolis_forces(self, q: Tensor, qd: Tensor) -> Tensor:
        p = self.p
        q2 = q[..., 1]
        qd1, qd2 = qd[..., 0], qd[..., 1]
        h = p.m2 * p.l1 * p.lc2 * torch.sin(q2)
        c1 = -h * (2 * qd1 * qd2 + qd2**2)
        c2 = h * qd1**2
        return torch.stack([c1, c2], dim=-1)

    def gravity_forces(self, q: Tensor) -> Tensor:
        p = self.p
        q1, q2 = q[..., 0], q[..., 1]
        g = p.gravity
        g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * g * torch.cos(q1) + p.m2 * p.lc2 * g * torch.cos(
            q1 + q2
        )
        g2 = p.m2 * p.lc2 * g * torch.cos(q1 + q2)
        return torch.stack([g1, g2], dim=-1)


def make_system(name: str) -> RigidBodySystem:
    """Factory used by the offline-fit script (``--system pendulum|two_link``)."""
    systems = {"pendulum": Pendulum, "two_link": TwoLinkArm}
    try:
        return systems[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown system {name!r}; choose from {sorted(systems)}") from exc
