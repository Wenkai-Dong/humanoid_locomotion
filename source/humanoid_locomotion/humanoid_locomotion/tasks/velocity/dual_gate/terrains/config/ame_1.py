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
    num_cols=1,
    sub_terrains={
        "Flat": attention_terrains_gen.HfRandomUniformDifficultyTerrainCfg(
            noise_range=(-0.075, 0.075),
            noise_step=0.02,
            border_width=1.
        ),
    },
)