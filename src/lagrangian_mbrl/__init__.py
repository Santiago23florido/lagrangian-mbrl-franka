"""lagrangian_mbrl — physics-guided model-based RL for the Franka Panda.

Subpackages
-----------
- :mod:`lagrangian_mbrl.models` : dynamics models (Deep Lagrangian Network +
  unstructured MLP baseline).
- :mod:`lagrangian_mbrl.mbrl`   : the model-based RL training loop.
- :mod:`lagrangian_mbrl.envs`   : Isaac Lab Franka task wrappers.
- :mod:`lagrangian_mbrl.eval`   : sample-efficiency and physics metrics.
- :mod:`lagrangian_mbrl.utils`  : seeding, logging, and config helpers.

See ``PROJECT_PLAN.md`` for the research plan this package implements.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
