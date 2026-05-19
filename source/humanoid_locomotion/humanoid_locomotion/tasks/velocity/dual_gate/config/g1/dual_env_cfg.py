# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import math
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from .attention_env_cfg import G1AttentionEnvCfgv3
from humanoid_locomotion.tasks.velocity.dual_gate import mdp

@configclass
class G1DualEnvCfg(G1AttentionEnvCfgv3):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # privilege observation
        self.observations.critic.joint_effort_l = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["left_hip_.*_joint", "left_knee_joint", "left_ankle_.*_joint"],
                )
            }
        )
        self.observations.critic.body_incoming_wrench_l = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
            }
        )
        self.observations.critic.contact_forces_l = ObsTerm(
            func=mdp.body_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            }
        )
        self.observations.critic.contact_time_l = ObsTerm(
            func=mdp.body_contact_time,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            }
        )
        self.observations.critic.pose_root_l = ObsTerm(
            func=mdp.body_pose_root,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
            }
        )
        self.observations.critic.body_vel_w_root_l = ObsTerm(
            func=mdp.body_vel_w_root,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
            }
        )
        self.observations.critic.joint_effort_r = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["right_hip_.*_joint", "right_knee_joint", "right_ankle_.*_joint"],
                )
            }
        )
        self.observations.critic.body_incoming_wrench_r = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
            }
        )
        self.observations.critic.contact_forces_r = ObsTerm(
            func=mdp.body_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
            }
        )
        self.observations.critic.contact_time_r = ObsTerm(
            func=mdp.body_contact_time,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
            }
        )
        self.observations.critic.pose_root_r = ObsTerm(
            func=mdp.body_pose_root,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
            }
        )
        self.observations.critic.body_vel_w_root_r = ObsTerm(
            func=mdp.body_vel_w_root,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
            }
        )


@configclass
class G1DualEnvCfg_PLAY(G1DualEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # Increase the length_s of episode
        self.episode_length_s = 40.0
        # make a smaller scene for play
        self.scene.num_envs = 32


@configclass
class RecorderManagerCfg(mdp.TrackingErrorRecorderManagerCfg):
    """Recorder configurations for recording actions and states."""

    dataset_export_dir_path = "logs/rsl_rl/"

@configclass
class G1DualEnvCfg_EVAL(G1DualEnvCfg):
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
