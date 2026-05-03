# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import G1VelocityRoughEnvCfg
from humanoid_locomotion.tasks.velocity.dual_gate import mdp
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config import ATTENTION_TERRAINS_CFG

@configclass
class G1AttentionEnvCfg(G1VelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to attention
        self.scene.terrain.terrain_generator = ATTENTION_TERRAINS_CFG


@configclass
class G1AttentionEnvCfg_PLAY(G1AttentionEnvCfg):
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
class G1AttentionEnvCfg_EVAL(G1AttentionEnvCfg):
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