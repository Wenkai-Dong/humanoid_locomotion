from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat, quat_apply_inverse
from isaaclab.sensors import RayCaster, ContactSensor
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def elevation_map(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, size: tuple[int, int], offset: float = 0.825, z_noise: float = 0.0) -> torch.Tensor:
    """Elevation Map from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)    # (N, B, 3)
    sensor_quat_expanded = yaw_quat(sensor.data.quat_w).unsqueeze(1).expand(-1, relative_pos_w.shape[1], -1)
    relative_pos_s = quat_apply_inverse(sensor_quat_expanded, relative_pos_w)

    relative_pos_s = torch.nan_to_num(relative_pos_s, nan=0.0, posinf=3.0, neginf=-3.0)
    # Z-axis height: height = hit_point_z - sensor_height + offset + noise
    relative_pos_s[..., 2] += offset + (torch.rand_like(relative_pos_s[..., 2]) - 0.5) * 2 * z_noise
    return relative_pos_s.reshape(relative_pos_w.shape[0], size[0], size[1], 3).permute(0, 3, 1, 2).contiguous()
# TODO
def body_contact_forces(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return contacts

def body_contact_time(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.cat([current_air_time, current_contact_time, last_air_time, last_contact_time], dim=-1)

def body_pose_root(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    root_pos_w = asset.data.root_pos_w  # (num_instances, 3)
    root_quat_w = math_utils.quat_unique(asset.data.root_quat_w)    # (num_instances, 4)
    body_pose_w = asset.data.body_pose_w[:, asset_cfg.body_ids, :7].reshape(env.num_envs, -1)    # (num_instances, num_bodies, 7)

    body_pose_root = math_utils.subtract_frame_transforms(
        t01=root_pos_w, q01=root_quat_w,
        t02=body_pose_w[..., :3], q02=body_pose_w[..., 3:7]
    )    # Tuple[(N, 3), (N, 4)]
    body_pose_root = torch.cat(body_pose_root, dim=-1) # (num_instances, 7)
    return body_pose_root

def body_lin_vel_w_root(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    root_quat_w = math_utils.quat_unique(asset.data.root_quat_w)    # (num_instances, 4)
    body_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids, :6].reshape(env.num_envs, -1)   # (num_instances, num_bodies, 6)

    body_lin_vel_w_root = math_utils.quat_apply_inverse(root_quat_w, body_vel_w[..., :3])  # (num_instances, 3)
    return body_lin_vel_w_root

def body_height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    ray_hits_w = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=-1e6, posinf=-1e6, neginf=-1e6)
    return sensor.data.pos_w[:, 2].unsqueeze(1) - ray_hits_w - offset