#!/usr/bin/env python
"""Training entry point.

Composes a Hydra config from ``configs/`` and runs the requested experiment:

    # Lagrangian model-based RL (the method):
    python scripts/train.py experiment=dln_mbrl

    # Unstructured MLP-ensemble MBRL baseline (swap only the dynamics model):
    python scripts/train.py experiment=dln_mbrl model=mlp

    # Model-free PPO baseline (RSL-RL / SKRL):
    python scripts/train.py experiment=ppo_baseline

TODO (Phase 0–2, see PROJECT_PLAN.md):
  [ ] @hydra.main over configs/; resolve + dump the config.
  [ ] seed_everything(cfg.seed); build env, dynamics model, logger.
  [ ] For MBRL: MBRLTrainer(env, model, cfg).train().
  [ ] For PPO baseline: dispatch to RSL-RL/SKRL runner.
"""

from __future__ import annotations


def main() -> None:
    # TODO: replace with a real Hydra-decorated entry point.
    raise NotImplementedError(
        "Training not yet wired up. See scripts/train.py TODOs and PROJECT_PLAN.md Phase 0."
    )


if __name__ == "__main__":
    main()
