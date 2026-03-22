from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat
from isaaclab.sensors import RayCaster, ContactSensor
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def elevation_mapping(
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        offset: float = 0.5,
) -> torch.Tensor:
    """Elevation mapping from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    device = sensor.device
    resolution = sensor.cfg.pattern_cfg.resolution
    size = sensor.cfg.pattern_cfg.size
    num_x = int(size[0] / resolution) + 1
    num_y = int(size[1] / resolution) + 1
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    height = height.view(env.num_envs, 1, num_y, num_x)
    sensor_offset = sensor.cfg.offset.pos
    # define grid pattern
    x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution, device=device) + sensor_offset[0]
    y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution, device=device) + sensor_offset[1]
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    # store into ray starts
    position = torch.stack([grid_x, grid_y], dim=0)
    position = position.unsqueeze(0).expand(env.num_envs, -1, -1, -1)
    return torch.cat([position, height], dim=1)

def contact_state(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    ).float()
    return contacts

def contact_forces(env, sensor_cfg: SceneEntityCfg, robot_mass: float) -> torch.Tensor:
    """Penalize if none of the desired contacts are present.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return contacts/(robot_mass*9.81)

def body_pose_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Computes the relative body velocity and pose for an articulated object in the
    simulation environment.

    This function processes the positional and velocity data of a specified asset
    within a simulation environment. It transforms and computes the relative
    pose and twist of a body in the local frame of reference. Various quantities
    such as linear and angular velocities in both world and local frames are merged
    into a single tensor output.

    Parameters:
        env (ManagerBasedEnv): An instance of the simulation environment managing
            assets and physical entities.
        asset_cfg (SceneEntityCfg, optional): Configuration object for the asset
            to be processed. Contains information like the asset's name and body
            ID list. Defaults to a robot configuration.

    Returns:
        torch.Tensor: A tensor containing concatenated relative pose and velocities
            (linear and angular) in various frames of reference. Dimensions are
            (num_instances, total_features), where total_features includes:
            7 for relative pose, 3 for body-local linear velocity, 3 for body-local
            angular velocity, and additional details (relative body velocity in
            different frames).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    root_pos_w = asset.data.root_pos_w  # (num_instances, 3)
    root_quat_w = math_utils.quat_unique(asset.data.root_quat_w)    # (num_instances, 4)
    root_lin_vel_w = asset.data.root_lin_vel_w  # (num_instances, 3)
    root_ang_vel_w = asset.data.root_ang_vel_w  # (num_instances, 3)

    body_pose_w = asset.data.body_pose_w[:, asset_cfg.body_ids, :7].reshape(env.num_envs, -1)    # (num_instances, num_bodies, 7)
    body_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids, :6].reshape(env.num_envs, -1)   # (num_instances, num_bodies, 6)

    body_pose_rel = math_utils.subtract_frame_transforms(
        t01=root_pos_w, q01=root_quat_w,
        t02=body_pose_w[..., :3], q02=body_pose_w[..., 3:7]
    )    # Tuple[(N, 3), (N, 4)]
    body_pose_rel = torch.cat(body_pose_rel, dim=-1) # (num_instances, 7)

    body_lin_vel_b_o = math_utils.quat_apply_inverse(root_quat_w, body_vel_w[..., :3])  # (num_instances, 3)
    body_ang_vel_b_o = math_utils.quat_apply_inverse(root_quat_w, body_vel_w[..., 3:6]) # (num_instances, 3)

    # access the body poses in world frame
    v_b, w_b = math_utils.rigid_body_twist_transform(
        v0=root_lin_vel_w, w0=root_ang_vel_w,
        t01=body_pose_w[..., :3] - root_pos_w, q01=root_quat_w,
    )
    body_lin_vel_b = body_lin_vel_b_o - v_b # (num_instances, 3)
    body_ang_vel_b = body_ang_vel_b_o - w_b # (num_instances, 3)
    return torch.cat([body_pose_rel, body_lin_vel_b_o, body_ang_vel_b_o/0.2, body_lin_vel_b, body_ang_vel_b/0.2], dim=-1)