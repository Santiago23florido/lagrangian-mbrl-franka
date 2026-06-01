"""Reproducible seeding across Python, NumPy, and PyTorch.

Reproducibility is a first-class requirement here (see
``docs/experiments_protocol.md``). Note that Isaac Sim is not fully
deterministic; we control for that by seed-averaging over >=5 seeds rather than
expecting bit-exact runs.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch RNGs; return the seed used.

    Parameters
    ----------
    seed:
        The integer seed to apply everywhere.
    deterministic_torch:
        If True, request cuDNN deterministic algorithms (slower, more
        reproducible). TODO: wire up torch.use_deterministic_algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is a hard dependency in practice; guard so utils import cleanly.
        pass
    return seed
