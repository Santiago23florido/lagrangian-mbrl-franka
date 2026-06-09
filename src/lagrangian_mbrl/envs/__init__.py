"""Environment wrappers and analytic systems for the Franka manipulation tasks.

:mod:`analytic_systems` provides closed-form rigid-body dynamics that need no
simulator (used for Phase-0 offline fitting and the unit tests). The Isaac Lab
Franka wrapper is imported lazily — importing it pulls in Isaac Sim — so it is
*not* re-exported here; import it directly when a simulator is running:

    from lagrangian_mbrl.envs.franka_reach_wrapper import FrankaReachEnv
"""

from .analytic_systems import (
    Pendulum,
    PendulumParams,
    RigidBodySystem,
    TwoLinkArm,
    TwoLinkArmParams,
    make_system,
)

__all__ = [
    "RigidBodySystem",
    "Pendulum",
    "PendulumParams",
    "TwoLinkArm",
    "TwoLinkArmParams",
    "make_system",
]
