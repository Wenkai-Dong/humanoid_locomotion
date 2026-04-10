# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
import humanoid_locomotion.tasks.velocity.ame_1.terrains as attention_terrains_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

AME1_STAGE1_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=10,
    num_rows=10,
    num_cols=9,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "Rough":attention_terrains_gen.HfRandomUniformDifficultyTerrainCfg(
            noise_range=(-0.075, 0.075),
            noise_step=0.02,
            border_width=1.
        ),
        "Stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            step_height_range=(0.01, 0.3),
            step_width=0.35,
            inverted=True,
            platform_width=2,
            border_width=1,
        ),
        "StairsInverted": terrain_gen.HfPyramidStairsTerrainCfg(
            step_height_range=(0.01, 0.3),
            step_width=0.35,
            inverted=False,
            platform_width=2,
            border_width=1,
        ),
        "Gaps": terrain_gen.MeshGapTerrainCfg(
            gap_width_range=(0.1,0.8),
            platform_width=2.,
        ),
        "GridStones": terrain_gen.HfSteppingStonesTerrainCfg(
            stone_height_max=0.2,
            stone_width_range=(0.3,0.8),
            stone_distance_range=(0.05,0.3),
            holes_depth=-10,
            platform_width=1.5,
            border_width=1,
        ),
        "Pallets": attention_terrains_gen.MeshConcentricBeamsTerrainCfg(
            function = attention_terrains_gen.mesh_concentric_beams_terrain,
            # 基础参数
            platform_width=1.5,
            beam_thickness=3.0,
            # 课程难度参数
            step_height_range=(0.0, 0.25),  # 难度越高，下沉越深
            beam_width_range=(0.5, 0.1),  # 难度越高，路越窄
            gap_width_range=(0.1, 0.5),  # 难度越高，缝隙越大
        ),
        "Pits": terrain_gen.MeshPitTerrainCfg(
            pit_depth_range=(0.1,0.5),
            double_pit=True,
            platform_width=2.,
        ),
        "PitsInverted": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(0.1,0.7),
            double_box=True,
            platform_width=2.,
        ),
        "Beams": terrain_gen.MeshStarTerrainCfg(
            num_bars=4,
            bar_width_range=(0.15,1.),
            bar_height_range=(5.,5.),
            platform_width=2.,
        ),
    },
)

AME1_STAGE2_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(14.0, 14.0),
    border_width=10,
    num_rows=10,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "ConsecutiveGaps": attention_terrains_gen.MeshConcentricBeamsTerrainCfg(
            function = attention_terrains_gen.mesh_concentric_beams_terrain,
            # 基础参数
            platform_width=2.,
            beam_thickness=3.0,
            # 课程难度参数
            step_height_range=(0.0, 0.2),  # 难度越高，下沉越深
            beam_width_range=(0.6, 0.4),  # 难度越高，路越窄
            gap_width_range=(0.5, 1.1),  # 难度越高，缝隙越大
        ),
        "NarrowStairs": attention_terrains_gen.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.01, 0.3),
            step_width=0.4,
            platform_width=0.55,
            holes=True,
            border_width=1,
        ),
        "NarrowStairsInverted": attention_terrains_gen.MeshInvertedPyramidStairsTerrainCfg(
            step_height_range=(0.01, 0.3),
            step_width=0.4,
            platform_width=0.55,
            holes=True,
            border_width=1,
        ),
        "stakes1": attention_terrains_gen.HfDoubleColumnStakesTerrainCfg(
            stake_height_max=0.03, stake_side_range=(0.30, 0.40), stake_gap_range=(0.1, 0.3),
            column_gap_range=(0.1, 0.1), column_jitter=0.0, holes_depth=-2.0, platform_width=2.0, border_width=1.
        ),
        "stakes2": attention_terrains_gen.HfAlternateColumnStakesTerrainCfg(
            stake_height_max=0.03, stake_side_range=(0.40, 0.50), stake_gap_range=(0.05, 0.15),
            column_gap_range=(0.0, 0.2), column_jitter=0.0, holes_depth=-2.0, platform_width=2.0, border_width=1.
        ),
        "stakes3": attention_terrains_gen.HfAlternateColumnStakesTerrainCfg(
            stake_height_max=0.03, stake_side_range=(0.30, 0.40), stake_gap_range=(0.05, 0.25),
            column_gap_range=(0.3, 0.2), column_jitter=0.0, holes_depth=-2.0, platform_width=2.0, border_width=1.
        ),
    },
)