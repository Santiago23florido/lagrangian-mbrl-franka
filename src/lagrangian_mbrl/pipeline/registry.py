"""Recommended dynamics-model architectures + their hyper-parameter spaces.

Which architectures, and why
----------------------------
The problem is *learning the forward dynamics* ``(q, q̇, τ) → q̈`` of an
articulated rigid body (the 7-DoF Franka) to use inside a model-based RL loop.
The recommended model classes for this problem are **physics-structured**
networks that embed the Euler–Lagrange equations, compared against a strong
**unstructured** baseline:

- ``"delan"`` — **Deep Lagrangian Network** (Lutter et al., 2019): the
  recommended structured model. It parameterizes the Lagrangian
  ``L(q, q̇) = ½ q̇ᵀ M(q) q̇ − V(q)`` with a Cholesky-factored, guaranteed-SPD
  mass matrix ``M(q) = L(q)L(q)ᵀ + εI`` and a potential ``V(q)``; the dynamics
  follow from autodiff of the Euler–Lagrange equation. Requires **C²
  (twice-differentiable) activations** — ``softplus``/``tanh`` — because the
  forces use second derivatives of the energy. This is the architecture we put
  forward as the solution.
- ``"mlp_ensemble"`` — **probabilistic MLP ensemble** (PETS-style; Chua et al.,
  2018): the recommended *unstructured* baseline. Each member predicts a
  Gaussian over ``q̈``; the ensemble captures epistemic uncertainty for planning.
- ``"mlp"`` — a single deterministic MLP: a minimal unstructured reference.

Related structured models worth a future ablation — Lagrangian Neural Networks
(Cranmer et al., 2020), Hamiltonian NNs (Greydanus et al., 2019) and Lagrangian
Graph NNs (Bhattoo et al., 2022) — are described in ``theory/dynamics_models.tex``.
New architectures plug in by adding a builder to ``MODEL_REGISTRY``.

Hyper-parameter spaces
----------------------
:func:`model_hp_space` returns, per model, a dict of ``name -> spec`` where each
spec is one of:

- ``("categorical", [choices...])``
- ``("int", low, high)``
- ``("loguniform", low, high)``

consumed by both the Optuna and the random-search backends in
:mod:`lagrangian_mbrl.pipeline.hpo`.
"""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from lagrangian_mbrl.models import (
    DeepLagrangianNetwork,
    DeLaNConfig,
    MLPDynamics,
    MLPDynamicsConfig,
    MLPDynamicsEnsemble,
    MLPEnsembleConfig,
)

# Hyper-parameters that configure *training* rather than the network itself.
# The benchmark/HPO treat these uniformly with architecture knobs but the
# experiment harness routes them to the optimizer, not to ``build_model``.
OPTIM_HP_KEYS = ("lr", "weight_decay", "batch_size")


def _hidden_sizes(hp: dict[str, Any], default_width: int, default_depth: int) -> tuple[int, ...]:
    width = int(hp.get("hidden_width", default_width))
    depth = int(hp.get("hidden_depth", default_depth))
    return tuple([width] * depth)


def _build_delan(dof: int, hp: dict[str, Any]) -> nn.Module:
    return DeepLagrangianNetwork(
        DeLaNConfig(
            dof=dof,
            hidden_sizes=_hidden_sizes(hp, 128, 2),
            activation=str(hp.get("activation", "softplus")),  # must be C²
            epsilon=float(hp.get("epsilon", 1e-3)),
            diag_softplus_shift=float(hp.get("diag_softplus_shift", 1e-3)),
        )
    )


def _build_mlp(dof: int, hp: dict[str, Any]) -> nn.Module:
    return MLPDynamics(
        MLPDynamicsConfig(
            dof=dof,
            state_dim=2 * dof,
            action_dim=dof,
            hidden_sizes=_hidden_sizes(hp, 256, 3),
            activation=str(hp.get("activation", "silu")),
            probabilistic=False,
            target="acceleration",
        )
    )


def _build_mlp_ensemble(dof: int, hp: dict[str, Any]) -> nn.Module:
    member = MLPDynamicsConfig(
        dof=dof,
        state_dim=2 * dof,
        action_dim=dof,
        hidden_sizes=_hidden_sizes(hp, 256, 3),
        activation=str(hp.get("activation", "silu")),
        probabilistic=True,
        target="acceleration",
    )
    return MLPDynamicsEnsemble(MLPEnsembleConfig(member=member, size=int(hp.get("ensemble_size", 5))))


# name -> builder(dof, hp) -> nn.Module
MODEL_REGISTRY: dict[str, Callable[[int, dict[str, Any]], nn.Module]] = {
    "delan": _build_delan,
    "mlp": _build_mlp,
    "mlp_ensemble": _build_mlp_ensemble,
}


def build_model(name: str, dof: int, hp: dict[str, Any] | None = None) -> nn.Module:
    """Instantiate a registered dynamics model with hyper-parameters ``hp``."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](dof, hp or {})


# Per-model hyper-parameter search spaces. ``softplus``/``tanh`` only for DeLaN
# (C² requirement); the optimizer knobs are shared.
_OPTIM_SPACE: dict[str, tuple] = {
    "lr": ("loguniform", 3e-4, 1e-2),
    "weight_decay": ("loguniform", 1e-7, 1e-3),
    "batch_size": ("categorical", [32, 64, 128]),
}

_HP_SPACES: dict[str, dict[str, tuple]] = {
    "delan": {
        "hidden_width": ("categorical", [64, 128, 256]),
        "hidden_depth": ("int", 2, 3),
        "activation": ("categorical", ["softplus", "tanh"]),
        "epsilon": ("loguniform", 1e-4, 1e-2),
        **_OPTIM_SPACE,
    },
    "mlp": {
        "hidden_width": ("categorical", [128, 256, 512]),
        "hidden_depth": ("int", 2, 4),
        "activation": ("categorical", ["silu", "relu", "tanh"]),
        **_OPTIM_SPACE,
    },
    "mlp_ensemble": {
        "hidden_width": ("categorical", [128, 256, 512]),
        "hidden_depth": ("int", 2, 4),
        "activation": ("categorical", ["silu", "relu", "tanh"]),
        "ensemble_size": ("categorical", [3, 5, 7]),
        **_OPTIM_SPACE,
    },
}


def model_hp_space(name: str) -> dict[str, tuple]:
    """Return the hyper-parameter search space for a registered model."""
    if name not in _HP_SPACES:
        raise KeyError(f"No HP space for '{name}'. Known: {sorted(_HP_SPACES)}")
    return dict(_HP_SPACES[name])
