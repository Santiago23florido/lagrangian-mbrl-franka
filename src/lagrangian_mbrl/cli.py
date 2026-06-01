"""Console-script entry points (see ``[project.scripts]`` in pyproject.toml).

These thin wrappers let ``lmbrl-train`` / ``lmbrl-eval`` be installed on PATH.
They defer to the same logic as ``scripts/train.py`` and ``scripts/evaluate.py``.

TODO: parse Hydra config and dispatch to MBRLTrainer / evaluation routines.
"""

from __future__ import annotations


def train_entrypoint() -> None:
    """Entry point for the ``lmbrl-train`` console script. TODO."""
    raise NotImplementedError("Wire up Hydra config -> MBRLTrainer.train().")


def eval_entrypoint() -> None:
    """Entry point for the ``lmbrl-eval`` console script. TODO."""
    raise NotImplementedError("Wire up checkpoint loading -> evaluation metrics.")
