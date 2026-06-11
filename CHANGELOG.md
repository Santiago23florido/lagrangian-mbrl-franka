# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Dynamics-learning benchmark pipeline (data → train → test, multi-seed + HPO).**
  - `pipeline/registry.py`: recommended architectures behind one `build_model`
    factory — structured **DeLaN** (C²-activation, SPD mass matrix) as the
    proposed model, plus the PETS-style probabilistic **MLP ensemble** and a
    plain MLP as unstructured baselines — each with a hyper-parameter space.
  - `pipeline/data.py`: data-source abstraction (analytic `pendulum`/`two_link`
    now; `franka:<log.npz>` Isaac Lab logs later) with seeded, model-shared
    train/val splits.
  - `pipeline/experiment.py`: train+test one `(model, hp, seed)` configuration.
  - `pipeline/hpo.py`: hyper-parameter search — Optuna (TPE) if installed, with
    a dependency-free random-search fallback.
  - `pipeline/benchmark.py`: HPO → multi-seed retrain → bootstrap-CI comparison,
    writing `benchmark_results.json`, `summary.csv`, and a bar plot.
  - `scripts/run_benchmark.py` + `configs/benchmark/dynamics_offline.yaml`:
    one-command, CPU-only overnight entry point.
  - `tests/test_pipeline.py`: end-to-end pipeline tests on analytic data.
  - `theory/dynamics_models.tex`: Lagrangian equations, where the network enters
    DeLaN, per-model architectures, and the MPPI controller.
  - Optional `optuna` dependency (`pip install -e ".[hpo]"`).
- **Phase 0 — DeLaN implementation & offline reproduction (simulator-free).**
  - `models/deep_lagrangian_network.py`: full DeLaN forward pass — Cholesky-
    parameterized PD mass matrix `M(q) = L Lᵀ + εI`, potential-energy head
    `V(q)`, Coriolis/gravity via the autograd Euler–Lagrange identity
    `c + g = (∂²L/∂q̇∂q) q̇ − ∂L/∂q` (double-backward JVP), forward/inverse
    dynamics, a symplectic one-step integrator, input normalization, and an
    inverse-/forward-dynamics training loss.
  - `models/mlp_dynamics.py`: the unstructured baseline (probabilistic head,
    normalization, NLL/MSE loss, ensemble) with `acceleration`/`delta` targets.
  - `envs/analytic_systems.py`: closed-form `Pendulum` and `TwoLinkArm`
    rigid-body systems with exact `M, c, g` — the simulator-free source of
    `(q, q̇, τ, q̈)` offline data and the unit-test oracle.
  - `eval/metrics.py`: implemented `acceleration_mse`, `rollout_mse`,
    `energy_drift`, `mass_matrix_min_eigenvalue`, and the sample-efficiency
    helpers.
  - `offline.py` + `scripts/fit_dynamics_offline.py`: fits DeLaN and the MLP on
    the same offline data and reports held-out one-step acceleration MSE — the
    Phase-0 exit criterion. DeLaN beats the MLP by ~5–11× on the 2-link arm.
  - Tests: real (non-xfail) DeLaN physics tests (PD mass matrix, forward/inverse
    consistency, energy conservation, differentiability), analytic-system and
    metric tests, and the Phase-0 DeLaN-vs-MLP comparison test.
- Initial repository scaffold: `src/` package layout, configs, tests, docs,
  theory, and CI-friendly tooling.
- `PROJECT_PLAN.md` — full theory→implementation→publication research plan.
- `README.md` — overview, motivation, method, install, usage, reproduction.
- Placeholder modules with docstrings/TODOs:
  - `models/deep_lagrangian_network.py` (energy-conserving dynamics).
  - `models/mlp_dynamics.py` (unstructured baseline + ensemble).
  - `mbrl/training_loop.py` (model-based RL loop).
  - `envs/franka_reach_wrapper.py` (Isaac Lab Franka reach task wrapper).
  - `eval/metrics.py` (sample-efficiency & physics-consistency metrics).
- `docs/`: setup guide, architecture/design doc, experiments protocol.
- MIT `LICENSE`, `.gitignore`, `pyproject.toml`, `requirements.txt`.

## [0.1.0] - 2026-06-01

### Added
- Project inception.

[Unreleased]: https://github.com/<user>/lagrangian-mbrl-franka/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/<user>/lagrangian-mbrl-franka/releases/tag/v0.1.0
