import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from humanoid_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_DelayPD_CFG as ROBOT_CFG
from humanoid_locomotion.tasks.velocity.dual_gate import mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    actor_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.32, 0.0, 20)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=[1.36, 0.96]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
        drift_range=(-0.03, 0.03),
        ray_cast_drift_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
    )
    critic_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.32, 0.0, 20)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=[1.36, 0.96]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
        drift_range=(-0.0, 0.0),
        ray_cast_drift_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
    )
    base_height = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
        drift_range=(-0.0, 0.0),
        ray_cast_drift_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
        debug_vis=True
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ActorCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ActorMapCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        height_scanner = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("actor_height_scanner"), "size": (13, 18), "offset": 0.825, "z_noise": 0.05},
            clip=(-3.0, 3.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    actor: ActorCfg = ActorCfg()
    actor_map: ActorMapCfg = ActorMapCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # The first term in the critic observation must be velocity. when calculates the MSE loss of velocity estimator,
        # The PPO will use the true velocity obtained from the first term of the critic observation.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticMapCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        height_scanner = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("critic_height_scanner"), "size": (13, 18), "offset": 0.825, "z_noise": 0.0},
            clip=(-3.0, 3.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # privileged observations
    critic: CriticCfg = CriticCfg()
    critic_map: CriticMapCfg = CriticMapCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "distribution": "uniform",
        },
    )

    armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "armature_distribution_params": (0.85, 1.15),
        }
    )

    payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)},
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task
    linear_velocity_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity", "std": math.sqrt(1.)
        },
    )
    angular_velocity_tracking = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(1.)}
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-3)
    joint_acceleration_penalty = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"])
        },
    )
    # shoulder_pitch_joint_acceleration_penalty = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-1.0e-4,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"]),
    #     },
    # )
    joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint"])
        },
    )
    joint_position_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    joint_velocity_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.1, params={"soft_ratio": 0.9})
    joint_torque_limits = RewTerm(func=mdp.joint_torque_limits, weight=-2.0e-3, params={"soft_ratio": 0.8})

    # -- Style
    angular_velocity_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-5.0e-2)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    joint_deviation_penalty_l2 = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]),
            "threshold": 0.25,
        },
    )
    no_fly = RewTerm(
        func=mdp.no_fly,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "grace_period_steps": 5,
        },
    )
    straight_body = RewTerm(
        func=mdp.straight_orientation_l2,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis", ".*_ankle_roll_link" ]
            )
        }
    )
    standing_joint_velocity_penalty = RewTerm(
        func=mdp.stand_still_velocity,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "threshold": 0.05,
        },
    )
    # g1-29dof else joint
    wrist_joint_deviation_penalty_l1 = RewTerm(    # 6
        func=mdp.joint_deviation_l1,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*_joint"]),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.001,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]), "threshold": 1.0}
    )
    waist_yaw_joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    hip_yaw_joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    base_height_terrain = DoneTerm(
        func=mdp.root_height_below_minimum_terrain,
        params={"minimum_height": 0.2, "sensor_cfg": SceneEntityCfg("base_height")}
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -3.0})
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_knee_link"]), "threshold": 1.0}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.attention_terrain_levels)


@configclass
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the velocity velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**28
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.actor_height_scanner is not None:
            self.scene.actor_height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.critic_height_scanner is not None:
            self.scene.critic_height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0
        self.scene.num_envs = 64
        self.scene.terrain.max_init_terrain_level = None


@configclass
class RecorderManagerCfg(mdp.TrackingErrorRecorderManagerCfg):
    """Recorder configurations for recording actions and states."""

    dataset_export_dir_path = "logs/rsl_rl/"


@configclass
class G1RoughEnvCfg_EVAL(G1RoughEnvCfg):
    # Recoder Settings
    recorders: RecorderManagerCfg = RecorderManagerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene settings
        self.scene.terrain.max_init_terrain_level = None
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

class G1VelocityRoughEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.actor.history_length = 50
        self.observations.actor.flatten_history_dim = False

class G1VelocityRoughEnvCfg_PLAY(G1VelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0
        self.scene.num_envs = 64
        self.scene.terrain.max_init_terrain_level = None

class G1VelocityRoughEnvCfg_EVAL(G1VelocityRoughEnvCfg):
    def __post_init__(self):

        super().__post_init__()
        # Scene settings
        self.scene.terrain.max_init_terrain_level = None
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