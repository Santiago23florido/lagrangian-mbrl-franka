# Architecture & Design

This document explains how the code is organized and *why*, so the abstractions
hold up as the project grows from a reach prototype to a multi-task study.

## Design goals

1. **Swappable dynamics models.** The single most important comparison in the
   paper is "structured vs. unstructured dynamics inside the *same* loop." The
   architecture makes the dynamics model a plug-in behind one interface so the
   MBRL loop, planner, and configs are identical across the method and the
   baseline. This *is* the experimental control.
2. **Simulator isolation.** Isaac Sim is heavy and not always available (CI,
   theory work). Everything except `envs/` imports without a simulator; Isaac
   Lab imports are lazy (inside methods).
3. **Reproducibility by construction.** Config-driven runs, captured metadata,
   seed-averaged metrics, regenerable figures.

## Module map

```
src/lagrangian_mbrl/
├── models/
│   ├── deep_lagrangian_network.py   # M(q), V(q) -> Euler–Lagrange dynamics
│   └── mlp_dynamics.py              # probabilistic MLP + ensemble (PETS/MBPO)
├── mbrl/
│   └── training_loop.py             # collect -> fit model -> plan/learn -> eval
├── envs/
│   └── franka_reach_wrapper.py      # Isaac Lab Franka task -> (q, q̇, τ, q̈)
├── eval/
│   └── metrics.py                   # sample-eff, rollout MSE, energy drift
└── utils/
    ├── seeding.py                   # cross-library reproducible seeding
    └── logging.py                   # W&B + TensorBoard facade + metadata
```

## The key interface: `DynamicsModel`

Defined structurally as a `Protocol` in
[`mbrl/training_loop.py`](../src/lagrangian_mbrl/mbrl/training_loop.py):

```python
class DynamicsModel(Protocol):
    def predict_next_state(self, *args, **kwargs) -> Tensor: ...
    def loss(self, batch: dict[str, Tensor]) -> Tensor: ...
```

Both `DeepLagrangianNetwork` and `MLPDynamics`/`MLPDynamicsEnsemble` satisfy it.
The `MBRLTrainer` only ever talks to this interface, so switching models is a
config change (`model=dln` ↔ `model=mlp`).

### Why DeLaN gets extra structured methods
`DeepLagrangianNetwork` additionally exposes `mass_matrix(q)`,
`potential_energy(q)`, `lagrangian(q, q̇)`, and `inverse_dynamics(...)`. These
are *not* part of the shared interface — they're used by physics-consistency
metrics (`eval/metrics.py`) and the DeLaN training loss. The MLP simply doesn't
have them, and nothing in the loop assumes it does.

## Data flow (MBRL loop)

```
            ┌─────────────────────────── MBRLTrainer.train() ───────────────────────────┐
            │                                                                            │
 FrankaReachEnv ──(q, q̇, τ, q̈, r, done)──▶ ReplayBuffer ──▶ DynamicsModel.loss/fit       │
      ▲                                                          │                       │
      │                                                          ▼                       │
   action ◀──── Planner (MPPI/CEM) / Policy ◀──── DynamicsModel.predict_next_state ──────┘
                              │
                              └──▶ eval/metrics ──▶ ExperimentLogger (W&B + TB) ──▶ logs/
```

- **MPC branch (Phase 2):** the planner queries `predict_next_state` to optimize
  an action sequence each control step; no policy network required.
- **Dyna branch (Phase 3):** short model rollouts feed an off-policy RL update
  (SAC), MBPO-style, for speed at scale.

## Configuration

Hydra composes `configs/config.yaml` from `env/`, `model/`, and `experiment/`
groups. Overrides on the CLI (`model=mlp seed=3`) select the comparison cell.
Every run dumps its resolved config alongside metadata (git SHA, seed, GPU).

## Testing strategy

- **CI-safe:** package import, seeding, metric math (no simulator).
- **Physics contracts:** DeLaN PD-mass-matrix and energy-conservation tests
  (xfail until implemented) on analytic pendulum / 2-link systems — these guard
  the property the whole thesis rests on.
- **Integration (manual/GPU):** short end-to-end MBRL run on reach.

## Extension points

- New task (push / peg-insertion): subclass the env wrapper; add a `configs/env/`
  file. Contact tasks motivate `learn_dissipation=true` in the DeLaN config.
- New dynamics model (HNN/LNN): implement the `DynamicsModel` interface; add a
  `configs/model/` file. No loop changes.
