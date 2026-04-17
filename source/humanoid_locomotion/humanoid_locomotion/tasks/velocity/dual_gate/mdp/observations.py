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
    # Z-axis height: height = hit_point_z - sensor_height + offset + noise
    relative_pos_s[..., 2] += offset + (torch.rand_like(relative_pos_s[..., 2]) - 0.5) * 2 * z_noise
    return relative_pos_s.reshape(relative_pos_w.shape[0], size[0], size[1], 3)