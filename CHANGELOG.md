# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
