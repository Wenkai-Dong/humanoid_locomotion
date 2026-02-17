# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from humanoid_locomotion.tasks.velocity.ame_1.config.h1.stage1_env_cfg import H1Stage1EnvCfg
from humanoid_locomotion.tasks.velocity.ame_1.terrains.config.ame_1 import AME1_STAGE2_TERRAINS_CFG
from humanoid_locomotion.tasks.velocity.ame_1 import mdp
from humanoid_locomotion.tasks.velocity.ame_1.config.h1.stage1_env_cfg import RewardsCfg



@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg_MapScans(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        height_scanner = ObsTerm(func=mdp.elevation_mapping,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
            noise=Gnoise(mean=0.0, std=0.02),
        )

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    policy_map: PolicyCfg_MapScans = PolicyCfg_MapScans()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, )
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg_MapScans(ObsGroup):
        """Observations for critic group."""
        height_scanner = ObsTerm(func=mdp.elevation_mapping,
                                 params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                                 clip=(-5.0, 5.0),
                                 )

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    # privileged observations
    critic: CriticCfg = CriticCfg()
    critic_map: CriticCfg_MapScans = CriticCfg_MapScans()


@configclass
class S2RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # -- Style
    standing_joint_velocity_penalty = RewTerm(
        func=mdp.stand_still_velocity,
        weight=-0.2,
    )


@configclass
class H1Stage2EnvCfg(H1Stage1EnvCfg):
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: S2RewardsCfg = S2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.terrain.terrain_generator = AME1_STAGE2_TERRAINS_CFG
        self.scene.height_scanner.drift_range = (-0.1, 0.1)
        # terminations
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                    "pelvis",
                    ".*shoulder.*",
                ],
            ),
            "threshold": 1.0
        }


@configclass
class H1Stage2EnvCfg_PLAY(H1Stage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 64
