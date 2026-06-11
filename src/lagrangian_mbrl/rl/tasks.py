"""Gymnasium registration for the project's Franka reach RL tasks.

Importing this module registers the environment IDs below. The training and play
scripts import it (after the Isaac Sim app is launched) so that
``gym.make("LMBRL-Franka-Reach-v0", cfg=...)`` resolves.

Registered IDs
--------------
- ``LMBRL-Franka-Reach-v0``      — full task (joint-position control).
- ``LMBRL-Franka-Reach-Play-v0`` — small-scene variant for eval / video.
"""

from __future__ import annotations

import gymnasium as gym

from lagrangian_mbrl.rl.agents.rsl_rl_ppo_cfg import FrankaReachPPORunnerCfg
from lagrangian_mbrl.rl.franka_reach_env_cfg import (
    FrankaReachEnvCfg,
    FrankaReachEnvCfg_PLAY,
)

gym.register(
    id="LMBRL-Franka-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaReachEnvCfg,
        "rsl_rl_cfg_entry_point": FrankaReachPPORunnerCfg,
    },
)

gym.register(
    id="LMBRL-Franka-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": FrankaReachPPORunnerCfg,
    },
)
