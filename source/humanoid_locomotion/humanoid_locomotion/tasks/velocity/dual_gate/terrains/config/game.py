# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
import humanoid_locomotion.tasks.velocity.dual_gate.terrains as dual_gate_terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from ..height_field import hf_terrains

BEAM_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "Beams": terrain_gen.MeshStarTerrainCfg(
            num_bars=2,
            bar_width_range=(0.135, 1.0),
            bar_height_range=(10.0, 10.0),
            platform_width=2.0,
        ),
    },
)

GridStones_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "GridStones": dual_gate_terrain_gen.MeshSteppingStonesTerrainCfg(
            stone_height_max=(0, 0.05),
            stone_width_range=(0.18, 0.5),
            stone_distance_range=(0.05, 0.3),
            platform_width=(2.0, 1.0),
            border_width=1.0,
            max_yx_angle=False,
        ),
    },
)

GridStonesNarrow_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "GridStonesNarrow": dual_gate_terrain_gen.MeshPalletsNarrowTerrainCfg(
            num_bars=2,
            bar_width_range=(0.2, 1.0),
            stone_length_range=(0.17, 1.0),
            stone_distance_range=(0.05, 0.25),
            stone_height_range=(0.0, 0.05),
            platform_width=(2.0, 1.0),
        ),
    },
)

GridStonesAngled_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "GridStonesAngle": dual_gate_terrain_gen.MeshSteppingStonesTerrainCfg(
            stone_height_max=(0, 0.05),
            stone_width_range=(0.2, 0.5),
            stone_distance_range=(0.02, 0.25),
            platform_width=(2.0, 1.0),
            border_width=1.0,
            max_yx_angle=(0, 10),
        ),
    },
)

Pallets_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "PalletsMesh": dual_gate_terrain_gen.MeshPalletsTerrainCfg(
            pallet_height=(0.0, 0.05),
            pallet_width_range=(0.15, 0.5),
            pallet_distance_range=(0.05, 0.5),
            platform_width=(2.0, 1.0),
            border_width=1.0,
        ),
    },
)
PalletsNarrow_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "PalletsNarrow": dual_gate_terrain_gen.MeshPalletsNarrowTerrainCfg(
            num_bars=2,
            bar_width_range=(0.5, 1.0),
            stone_length_range=(0.15, 1.0),
            stone_distance_range=(0.05, 0.25),
            stone_height_range=(0.0, 0.05),
            platform_width=(2.0, 1.0),
        )
    },
)
Gap_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "Gaps": terrain_gen.MeshGapTerrainCfg(
            gap_width_range=(0.05,0.7),
            platform_width=2.0,
        ),
    },
)
Boxer_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(100.0, 100.0),
    border_width=10,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=0.45, grid_height_range=(0.05, 0.3), platform_width=2.0
        ),
    },
)
"""Attention terrains configuration."""