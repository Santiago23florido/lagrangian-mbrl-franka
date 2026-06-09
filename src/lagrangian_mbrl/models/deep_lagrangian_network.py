"""Deep Lagrangian Network (DeLaN) — energy-conserving structured dynamics.

This is the core methodological contribution of the project. Instead of letting
an unstructured network predict accelerations directly, we parameterize the
*Lagrangian* of the system and recover the dynamics through the Euler–Lagrange
equations, which guarantees a physically consistent inductive bias.

Background
----------
For a rigid-body manipulator the equations of motion are

    M(q) q̈  +  c(q, q̇)  +  g(q)  =  τ                                    (EoM)

where
    q, q̇, q̈ ∈ R^d   generalized positions / velocities / accelerations,
    M(q) ≻ 0         symmetric positive-definite mass (inertia) matrix,
    c(q, q̇)         Coriolis / centrifugal forces,
    g(q)            gravitational forces,
    τ               applied generalized torques.

DeLaN (Lutter, Ritter & Peters, ICLR 2019) learns the kinetic-energy mass
matrix ``M(q)`` and the potential energy ``V(q)`` with neural networks, then
derives ``c`` and ``g`` analytically from derivatives of the Lagrangian
``L = T - V = ½ q̇ᵀ M(q) q̇ - V(q)``. This guarantees:
  * ``M(q)`` is symmetric positive-definite (parameterized via a Cholesky
    factor ``L_θ(q)`` so that ``M = L_θ L_θᵀ + ε I``);
  * forces are conservative / energy is conserved in the unforced case;
  * far fewer effective degrees of freedom than a free-form MLP (this is what
    the sample-complexity argument in ``PROJECT_PLAN.md`` §2 exploits).

Implementation notes
--------------------
We recover the generalized forces ``c(q,q̇) + g(q)`` from the learned
Lagrangian using the compact Euler–Lagrange identity

    c(q,q̇) + g(q)  =  (∂²L/∂q̇∂q) q̇  −  ∂L/∂q                            (FORCES)

(derived in ``theory/derivations.md``). The first term is a Jacobian–vector
product of the generalized momentum ``p = ∂L/∂q̇`` along ``q̇``; we compute it
with a double-backward trick so the whole thing stays differentiable for
training. Because ``L`` is quadratic in ``q̇`` we additionally have the exact
identity ``∂²L/∂q̇∂q̇ = M(q)``, so we use the *structured* Cholesky ``M(q)`` for
the linear solves in forward dynamics (more stable than differentiating twice).

References
----------
- Lutter, Ritter, Peters. "Deep Lagrangian Networks." ICLR 2019.
- Lutter & Peters. "Combining Physics and Deep Learning for Continuous-Time
  Dynamics Models." 2023.
- Cranmer et al. "Lagrangian Neural Networks." ICLR 2020 (DeepDiffEq workshop).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


@dataclass
class DeLaNConfig:
    """Configuration for :class:`DeepLagrangianNetwork`.

    Attributes
    ----------
    dof:
        Number of generalized coordinates (7 for the Franka arm).
    hidden_sizes:
        Hidden layer widths for the energy heads.
    activation:
        Activation name; must be twice differentiable (e.g. ``softplus``,
        ``tanh``) because the dynamics use second derivatives of the energy.
    epsilon:
        Diagonal offset added to ``L Lᵀ`` to keep ``M(q)`` strictly PD.
    diag_softplus_shift:
        Constant added to the (softplus) Cholesky diagonal to keep it well away
        from zero, improving conditioning of ``M(q)``.
    learn_dissipation:
        If True, add a (positive-semidefinite) diagonal damping term for
        non-conservative / contact-influenced regimes (relaxes assumption A1 in
        PROJECT_PLAN.md). The damping enters forces as ``D(θ) q̇``.
    """

    dof: int = 7
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "softplus"
    epsilon: float = 1e-4
    diag_softplus_shift: float = 1e-3
    learn_dissipation: bool = False
    loss_type: str = "inverse"  # "inverse" (torque MSE, canonical) | "forward" (accel MSE)


def _build_mlp(
    in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int, activation: str
) -> nn.Sequential:
    """A plain MLP with a twice-differentiable activation and a linear head."""
    try:
        act = _ACTIVATIONS[activation]
    except KeyError as exc:  # pragma: no cover - guarded by config
        raise ValueError(
            f"Unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}"
        ) from exc
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden_sizes:
        layers += [nn.Linear(prev, width), act()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DeepLagrangianNetwork(nn.Module):
    """Energy-conserving dynamics model via the Euler–Lagrange equations.

    Implements the ``predict_next_state`` / ``loss`` interface (see
    :class:`lagrangian_mbrl.mbrl.training_loop.DynamicsModel`) so it is
    interchangeable with :class:`~lagrangian_mbrl.models.MLPDynamics` inside the
    MBRL loop.
    """

    def __init__(self, config: DeLaNConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeLaNConfig()
        d = self.config.dof
        self._n_tril = d * (d + 1) // 2
        # lower-triangular indices for assembling the Cholesky factor L_θ(q)
        tril_rows, tril_cols = torch.tril_indices(d, d)
        self.register_buffer("_tril_rows", tril_rows, persistent=False)
        self.register_buffer("_tril_cols", tril_cols, persistent=False)
        self.register_buffer("_diag_mask", (tril_rows == tril_cols), persistent=False)

        self.l_head = _build_mlp(
            d, self.config.hidden_sizes, self._n_tril, self.config.activation
        )
        self.v_head = _build_mlp(d, self.config.hidden_sizes, 1, self.config.activation)
        self._diag_act = nn.Softplus()

        # Input normalization for the energy heads (an affine reparameterization
        # of q; the autograd chain rule keeps the physics terms exact). Fit on
        # data via :meth:`fit_input_normalization` before training.
        self.register_buffer("q_mean", torch.zeros(d))
        self.register_buffer("q_std", torch.ones(d))

        if self.config.learn_dissipation:
            # raw -> softplus -> non-negative diagonal damping coefficients
            self.log_damping = nn.Parameter(torch.full((d,), -3.0))
        else:
            self.log_damping = None

    # --- structured internals -------------------------------------------------
    def fit_input_normalization(self, q: Tensor) -> None:
        """Set the energy-head input normalization stats from data (q only)."""
        self.q_mean.copy_(q.mean(0))
        self.q_std.copy_(q.std(0).clamp_min(1e-6))

    def _normalize_q(self, q: Tensor) -> Tensor:
        return (q - self.q_mean) / self.q_std

    def cholesky_factor(self, q: Tensor) -> Tensor:
        """Return the lower-triangular Cholesky factor ``L_θ(q)`` of ``M(q)``.

        The diagonal is passed through ``softplus`` (plus a shift) so it is
        strictly positive, which makes ``L Lᵀ`` positive-definite.
        """
        raw = self.l_head(self._normalize_q(q))  # (B, n_tril)
        diag_raw = raw[..., self._diag_mask]
        diag = self._diag_act(diag_raw) + self.config.diag_softplus_shift
        entries = raw.clone()
        entries[..., self._diag_mask] = diag
        d = self.config.dof
        L = q.new_zeros(*q.shape[:-1], d, d)
        L[..., self._tril_rows, self._tril_cols] = entries
        return L

    def mass_matrix(self, q: Tensor) -> Tensor:
        """Return the symmetric positive-definite mass matrix ``M(q)``.

        Parameters
        ----------
        q: (B, dof) batch of generalized positions.

        Returns
        -------
        (B, dof, dof) tensor with ``M = L Lᵀ + ε I`` (guaranteed PD).
        """
        L = self.cholesky_factor(q)
        d = self.config.dof
        eye = torch.eye(d, dtype=q.dtype, device=q.device)
        return L @ L.transpose(-1, -2) + self.config.epsilon * eye

    def potential_energy(self, q: Tensor) -> Tensor:
        """Return the scalar potential energy ``V(q)`` per batch element, shape (B,)."""
        return self.v_head(self._normalize_q(q)).squeeze(-1)

    def kinetic_energy(self, q: Tensor, qd: Tensor) -> Tensor:
        """Return ``T = ½ q̇ᵀ M(q) q̇`` per batch element, shape (B,)."""
        M = self.mass_matrix(q)
        return 0.5 * torch.einsum("...i,...ij,...j->...", qd, M, qd)

    def lagrangian(self, q: Tensor, qd: Tensor) -> Tensor:
        """Return ``L = ½ q̇ᵀ M(q) q̇ - V(q)`` per batch element, shape (B,)."""
        return self.kinetic_energy(q, qd) - self.potential_energy(q)

    def energy(self, q: Tensor, qd: Tensor) -> Tensor:
        """Total mechanical energy ``E = T + V`` (conserved when unforced)."""
        return self.kinetic_energy(q, qd) + self.potential_energy(q)

    # --- Euler–Lagrange force terms ------------------------------------------
    def generalized_forces(self, q: Tensor, qd: Tensor) -> Tensor:
        """Return ``c(q, q̇) + g(q)`` via the Euler–Lagrange identity (FORCES).

        Uses ``c + g = (∂²L/∂q̇∂q) q̇ − ∂L/∂q``. The mixed-partial term is a
        Jacobian–vector product of the generalized momentum ``p = ∂L/∂q̇`` along
        ``q̇``, evaluated with a double-backward trick so the result remains
        differentiable w.r.t. the network parameters (needed for training).
        """
        with torch.enable_grad():
            q = q.clone().requires_grad_(True)
            qd = qd.clone().requires_grad_(True)
            L = self.lagrangian(q, qd).sum()
            dL_dq, dL_dqd = torch.autograd.grad(L, (q, qd), create_graph=True)
            # JVP: (∂(dL_dqd)/∂q) @ qd, contracted over the q index.
            w = torch.zeros_like(dL_dqd, requires_grad=True)
            # g_j = Σ_i w_i ∂(dL_dqd)_i/∂q_j
            (g_of_w,) = torch.autograd.grad(dL_dqd, q, grad_outputs=w, create_graph=True)
            # jvp_i = Σ_j qd_j ∂g_j/∂w_i = Σ_j (∂(dL_dqd)_i/∂q_j) qd_j
            (jvp,) = torch.autograd.grad(g_of_w, w, grad_outputs=qd, create_graph=True)
        forces = jvp - dL_dq
        if self.log_damping is not None:
            forces = forces + torch.nn.functional.softplus(self.log_damping) * qd
        return forces

    # --- DynamicsModel interface ---------------------------------------------
    def forward_dynamics(self, q: Tensor, qd: Tensor, tau: Tensor) -> Tensor:
        """Predict accelerations ``q̈`` by solving the Euler–Lagrange equations.

        Solves ``M(q) q̈ = τ − (c(q, q̇) + g(q))`` for ``q̈`` (a batched linear
        solve). Returns a tensor shaped like ``q``.
        """
        M = self.mass_matrix(q)
        rhs = (tau - self.generalized_forces(q, qd)).unsqueeze(-1)
        qdd = torch.linalg.solve(M, rhs).squeeze(-1)
        return qdd

    def inverse_dynamics(self, q: Tensor, qd: Tensor, qdd: Tensor) -> Tensor:
        """Predict torques ``τ = M(q) q̈ + c(q, q̇) + g(q)`` — used for training."""
        M = self.mass_matrix(q)
        return torch.einsum("...ij,...j->...i", M, qdd) + self.generalized_forces(q, qd)

    def predict_next_state(
        self, q: Tensor, qd: Tensor, tau: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        """Integrate one step with semi-implicit (symplectic) Euler.

        Semi-implicit Euler updates the velocity first and then uses it for the
        position update, which (approximately) preserves the energy-conservation
        guarantee far better than explicit Euler. Returns ``(q_next, qd_next)``.
        """
        qdd = self.forward_dynamics(q, qd, tau)
        qd_next = qd + dt * qdd
        q_next = q + dt * qd_next
        return q_next, qd_next

    # --- training -------------------------------------------------------------
    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Training loss, selected by ``config.loss_type``.

        - ``"inverse"`` (default, canonical DeLaN): torque MSE between
          ``inverse_dynamics(q, q̇, q̈)`` and ``τ``. Avoids backpropagating
          through the ``M⁻¹`` solve, so it is markedly more stable than the
          forward objective and gives lower held-out acceleration error in
          practice (Lutter et al., 2019).
        - ``"forward"``: acceleration MSE between ``forward_dynamics(q, q̇, τ)``
          and the target ``q̈`` — matches the eval metric exactly but is
          sensitive to mass-matrix conditioning during early training.

        Batch keys: ``q``, ``qd``, ``qdd``, ``tau``.
        """
        q, qd = batch["q"], batch["qd"]
        if self.config.loss_type == "inverse":
            tau_pred = self.inverse_dynamics(q, qd, batch["qdd"])
            return torch.nn.functional.mse_loss(tau_pred, batch["tau"])
        qdd_pred = self.forward_dynamics(q, qd, batch["tau"])
        return torch.nn.functional.mse_loss(qdd_pred, batch["qdd"])

    def predict_acceleration(self, q: Tensor, qd: Tensor, tau: Tensor) -> Tensor:
        """Alias for :meth:`forward_dynamics` (matches the MLP baseline API)."""
        return self.forward_dynamics(q, qd, tau)
