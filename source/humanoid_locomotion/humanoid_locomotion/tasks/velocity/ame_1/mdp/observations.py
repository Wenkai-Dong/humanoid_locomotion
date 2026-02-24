from __future__ import annotations

import torch
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
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    ).float()
    return contacts