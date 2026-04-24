# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
import humanoid_locomotion.tasks.velocity.ame_1.terrains as attention_terrains_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from humanoid_locomotion.tasks.velocity.dual_gate.terrains import hf_terrains
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.hf_terrains_cfg import *

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=10,
    num_rows=1,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "Rough": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1, noise_range=(-0.075, 0.075), noise_step=0.02, downsampled_scale=0.1, border_width=1.0
        ),
        "Stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.3),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "StairsInverted": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.3),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "Gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.1,
            gap_width_range=(0.11,0.8),
            platform_width=2.0,
        ),
        "GridStones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.15,
            stone_width_range=(0.31, 0.5),
            stone_distance_range=(0.05, 0.3),
            platform_width=2.0,
            holes_depth=-10.0,
            border_width=1.0
        ),
        "Pallets": HfConcentricGapTerrainCfg(
            proportion=0.1,
            gap_width_range=(0.1, 0.5),
            platform_width=2.0,
            border_width=0.25,
            gap_depth=-10.0,
            ground_width_range=(0.31, 0.5),
            ground_height_max=0.025
        ),
        "Pits": terrain_gen.MeshPitTerrainCfg(
            proportion=0.1,
            pit_depth_range=(0.1,0.45),
            double_pit=True,
            platform_width=2.,
        ),
        "PitsInverted": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.1,
            box_height_range=(0.1,0.6),
            double_box=True,
            platform_width=2.,
        ),
        "Beams": terrain_gen.MeshStarTerrainCfg(
            proportion=0.2,
            num_bars=4,
            bar_width_range=(0.1,1.),
            bar_height_range=(5.,5.),
            platform_width=2.,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
    },
)
"""Rough terrains configuration."""