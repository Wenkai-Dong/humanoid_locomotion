from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat
from isaaclab.sensors import RayCaster
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
        device: str = "cuda:0",
        resolution: float = 0.1,
        size: tuple[float, float] = (1.6, 1.0),

) -> torch.Tensor:
    """Elevation mapping from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    height = height.view(env.num_envs, 1, 11, 17)
    # define grid pattern
    x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution, device=device)
    y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    # store into ray starts
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(env.num_envs, 2, num_rays, device=device)
    ray_starts[:, 0, :] = grid_x.flatten()
    ray_starts[:, 1, :] = grid_y.flatten()
    position = ray_starts.view(-1, 2, 11, 17)
    return torch.cat([position, height], dim=1)
