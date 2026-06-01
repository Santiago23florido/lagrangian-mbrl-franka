#!/usr/bin/env python
"""Evaluation entry point.

Loads a checkpoint and computes the metrics in
:mod:`lagrangian_mbrl.eval.metrics`:

    python scripts/evaluate.py --checkpoint logs/<run_id>/checkpoints/best.pt \
                               --metrics sample_efficiency rollout_mse energy_drift

TODO (Phase 2–4, see PROJECT_PLAN.md):
  [ ] argparse: --checkpoint, --metrics, --num-episodes, --seeds.
  [ ] Load env + dynamics model + policy from the checkpoint.
  [ ] Run requested metrics; aggregate across seeds (rliable, 95% CI).
  [ ] Write a results table + regenerate figures into figures/.
"""

from __future__ import annotations


def main() -> None:
    # TODO: replace with a real argparse-based entry point.
    raise NotImplementedError(
        "Evaluation not yet wired up. See scripts/evaluate.py TODOs and PROJECT_PLAN.md §5."
    )


if __name__ == "__main__":
    main()
