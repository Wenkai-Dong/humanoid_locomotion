# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg
from humanoid_locomotion.tasks.velocity.dual_gate import mdp


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        # no height scan
        self.scene.actor_height_scanner = None
        self.scene.critic_height_scanner = None
        self.observations.actor_map = None
        self.observations.critic_map = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # Increase the length_s of episode
        self.episode_length_s = 40.0
        # make a smaller scene for play
        self.scene.num_envs = 64
        self.scene.terrain.max_init_terrain_level = None


@configclass
class RecorderManagerCfg(mdp.TrackingErrorRecorderManagerCfg):
    """Recorder configurations for recording actions and states."""

    dataset_export_dir_path = "logs/rsl_rl/"


@configclass
class G1FlatEnvCfg_EVAL(G1FlatEnvCfg):
    # Recoder Settings
    recorders: RecorderManagerCfg = RecorderManagerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Basic settings
        self.commands.base_velocity.resampling_time_range = (30, 30)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_x = (0, 0)
        self.commands.base_velocity.ranges.heading = (0, 0)
        # MDP settings
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi/4, math.pi/4)
        # Recoder Settings
        self.terminations.success = DoneTerm(func=mdp.subterrain_out_of_bounds, params={"distance_buffer": 0.0})


