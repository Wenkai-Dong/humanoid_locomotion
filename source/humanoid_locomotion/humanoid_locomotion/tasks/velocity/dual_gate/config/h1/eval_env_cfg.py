# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from humanoid_locomotion.tasks.velocity.ame_1.config.h1.stage2_env_cfg import H1Stage2EnvCfg
from humanoid_locomotion.tasks.velocity.ame_1.config.h1.stage1_env_cfg import CommandsCfg, TerminationsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from humanoid_locomotion.tasks.velocity.ame_1.terrains.config.ame_1_eval import AME1_STAGE2_EVALUATE_TERRAINS_CFG
from humanoid_locomotion.tasks.velocity.ame_1 import mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


class EvalCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=1.,
        rel_standing_envs=0.00,
        rel_heading_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 0.4), lin_vel_y=(-0., 0.), ang_vel_z=(-1., 1.), heading=(-math.pi, math.pi)
        ),
        resampling_time_range=(30.0, 30.0),
        debug_vis=True,
    )


class EvalTerminationsCfg(TerminationsCfg):
    """Termination terms for the MDP."""
    success = DoneTerm(func=mdp.subterrain_out_of_bounds, params={"distance_buffer": 0.0})


@configclass
class RecorderManagerCfg(mdp.TrackingErrorRecorderManagerCfg):
    """Recorder configurations for recording actions and states."""

    dataset_export_dir_path = "logs/rsl_rl/ame1_eval_h1_v0"


@configclass
class H1EvalEnvCfg(H1Stage2EnvCfg):
    # Basic Settings
    commands: EvalCommandsCfg = EvalCommandsCfg()
    # MDP Settings
    events: EvalTerminationsCfg = EvalTerminationsCfg()
    # Recoder Settings
    recorders: RecorderManagerCfg = RecorderManagerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # scene
        self.scene.terrain.terrain_generator = AME1_STAGE2_EVALUATE_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 1
        # events
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)

@configclass
class H1EvalEnvCfg_PLAY(H1EvalEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 32