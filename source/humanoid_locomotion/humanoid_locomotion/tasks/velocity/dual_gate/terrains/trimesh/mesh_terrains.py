# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh

from .utils import *  # noqa: F401, F403
from .utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def pallets_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPalletsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # resolve terrain configuration
    pallet_width = cfg.pallet_width_range[1] - difficulty * (cfg.pallet_width_range[1] - cfg.pallet_width_range[0])
    pallet_distance = cfg.pallet_distance_range[0] + difficulty * (
        cfg.pallet_distance_range[1] - cfg.pallet_distance_range[0]
    )
    pallet_height = cfg.pallet_height[0] + difficulty * (cfg.pallet_height[1] - cfg.pallet_height[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    pallet_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2]

    # Generate the outer ring
    outer_size = [cfg.size[0], cfg.size[1]]
    inner_size = [cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width]
    meshes_list += make_border(outer_size, inner_size, terrain_height, terrain_center)
    outer_size[0] -=2 * (cfg.border_width + pallet_distance)
    outer_size[1] -=2 * (cfg.border_width + pallet_distance)
    inner_size[0] -=2 * (pallet_distance + pallet_width)
    inner_size[1] -=2 * (pallet_distance + pallet_width)
    # Generate the pallets
    while outer_size[0] > cfg.platform_width and outer_size[1] > cfg.platform_width:
        pallet_center[2] = -terrain_height / 2 + np.random.uniform(-pallet_height, pallet_height)
        meshes_list += make_border(outer_size, inner_size, terrain_height, pallet_center)
        outer_size[0] -= 2 * (pallet_distance + pallet_width)
        outer_size[1] -= 2 * (pallet_distance + pallet_width)
        inner_size[0] -= 2 * (pallet_distance + pallet_width)
        inner_size[1] -= 2 * (pallet_distance + pallet_width)

    # Generate the inner box
    box_dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin


def stepping_stones_height_terrain(
        difficulty: float, cfg: mesh_terrains_cfg.MeshSteppingStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    stone_height_max = cfg.stone_height_max[0] + difficulty * (cfg.stone_height_max[1] - cfg.stone_height_max[0])
    if cfg.max_yx_angle:
        max_yx_angle = cfg.max_yx_angle[0] + difficulty * (cfg.max_yx_angle[1] - cfg.max_yx_angle[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    ob_height = 1.0
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -ob_height / 2)
    # Generate the outer ring
    outer_size = [cfg.size[0], cfg.size[1]]
    inner_size = [cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width]
    meshes_list += make_border(outer_size, inner_size, ob_height, terrain_center)
    # add the stones
    start_x, start_y = 0 + cfg.border_width, 0 + cfg.border_width
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    while start_y < cfg.size[1] - cfg.border_width:
        # ensure that stone stops along y-axis
        stop_y = min(cfg.size[1] - cfg.border_width, start_y + stone_width)
        # randomly sample x-position
        start_x = np.random.uniform(0 + cfg.border_width, stone_width + cfg.border_width)
        stop_x = max(0 + cfg.border_width, start_x - stone_distance)
        # fill first stone
        length = stop_x - cfg.border_width
        width = stop_y - start_y
        height = ob_height
        center = ((stop_x + cfg.border_width) / 2, (stop_y + start_y) / 2, -ob_height / 2)
        if cfg.max_yx_angle:
            stone = make_box(length=length, width=width, height=height, center=center, max_yx_angle=max_yx_angle,)
        else:
            stone = trimesh.creation.box((length, width, height), trimesh.transformations.translation_matrix(center))
        meshes_list.append(stone)
        # fill row with stones
        while start_x < cfg.size[0] - cfg.border_width:
            stop_x = min(cfg.size[0] - cfg.border_width, start_x + stone_width)
            length = stop_x - start_x
            width = stop_y - start_y
            height = ob_height
            if not (stop_x < 0.5 * (cfg.size[0] - cfg.platform_width) or start_x > 0.5 * (cfg.size[0] + cfg.platform_width)
            or stop_y < 0.5 * (cfg.size[1] - cfg.platform_width) or start_y > 0.5 * (cfg.size[1] + cfg.platform_width)):
                center = ((stop_x + start_x) / 2, (stop_y + start_y) / 2, -ob_height / 2)
            else:
                center = ((stop_x + start_x) / 2, (stop_y + start_y) / 2,
                          -ob_height / 2 + np.random.uniform(-stone_height_max, stone_height_max))
            if stop_x >= cfg.size[0] - cfg.border_width:
                center = ((stop_x + start_x) / 2, (stop_y + start_y) / 2, -ob_height / 2)

            if cfg.max_yx_angle:
                stone = make_box(length=length, width=width, height=height, center=center, max_yx_angle=max_yx_angle, )
            else:
                stone = trimesh.creation.box((length, width, height),trimesh.transformations.translation_matrix(center))
            meshes_list.append(stone)

            start_x += stone_width + stone_distance
        # update y-position
        start_y += stone_width + stone_distance
    # add the platform in the center
    box_dim = (cfg.platform_width, cfg.platform_width, ob_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin
