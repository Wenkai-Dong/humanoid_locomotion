# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import math
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

from .rough_env_cfg import G1VelocityRoughEnvCfg
from humanoid_locomotion.tasks.velocity.dual_gate import mdp
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config.attention import ATTENTION_TERRAINS_CFG
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config.attention_v1 import ATTENTION_TERRAINS_CFGv1
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config.attention_v2 import ATTENTION_TERRAINS_CFGv2
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config.attention_v3 import ATTENTION_TERRAINS_CFGv3

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
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = 9


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

        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.difficulty_range = (0.9, 1.0)
        # Basic settings
        self.commands.base_velocity.resampling_time_range = (30, 30)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.heading = (0, 0)
        # MDP settings
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi/4, math.pi/4)
        # Recoder Settings
        self.terminations.success = DoneTerm(func=mdp.subterrain_out_of_bounds, params={"distance_buffer": 0.0})


@configclass
class G1AttentionEnvCfgv1(G1VelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to attention
        self.scene.terrain.terrain_generator = ATTENTION_TERRAINS_CFGv1

@configclass
class G1AttentionEnvCfgv1_PLAY(G1AttentionEnvCfgv1):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # Increase the length_s of episode
        self.episode_length_s = 40.0
        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = 9

@configclass
class G1AttentionEnvCfgv1_EVAL(G1AttentionEnvCfgv1):
    # Recoder Settings
    recorders: RecorderManagerCfg = RecorderManagerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.difficulty_range = (0.9, 1.0)
        # Basic settings
        self.commands.base_velocity.resampling_time_range = (30, 30)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.heading = (0, 0)
        # MDP settings
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi/4, math.pi/4)
        # Recoder Settings
        self.terminations.success = DoneTerm(func=mdp.subterrain_out_of_bounds, params={"distance_buffer": 0.0})

@configclass
class G1AttentionEnvCfgv2(G1VelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to attention
        self.scene.terrain.terrain_generator = ATTENTION_TERRAINS_CFGv2

@configclass
class G1AttentionEnvCfgv3(G1VelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to attention
        self.scene.terrain.terrain_generator = ATTENTION_TERRAINS_CFGv3
        self.rewards.standing_joint_velocity_penalty.weight = -0.02
        self.rewards.standing_joint_velocity_penalty.params["threshold"] = 0.01

@configclass
class G1AttentionEnvCfgv3_PLAY(G1AttentionEnvCfgv3):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # Increase the length_s of episode
        self.episode_length_s = 40.0
        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = 9

@configclass
class G1AttentionEnvCfgv3_EVAL(G1AttentionEnvCfgv3):
    # Recoder Settings
    recorders: RecorderManagerCfg = RecorderManagerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.difficulty_range = (0.9, 1.0)
        # Basic settings
        self.commands.base_velocity.resampling_time_range = (30, 30)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        # MDP settings
        self.events.reset_base.params["pose_range"]["x"] = (-0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (-0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.0, 0.0)
        # Recoder Settings
        self.terminations.success = DoneTerm(func=mdp.subterrain_out_of_bounds, params={"distance_buffer": 0.0})