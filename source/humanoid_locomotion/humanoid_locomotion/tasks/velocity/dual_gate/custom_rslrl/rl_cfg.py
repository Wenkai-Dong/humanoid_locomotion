# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.rl_cfg import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg


#########################
# Model configurations #
#########################


@configclass
class RslRlCNNVelocityModelCfg(RslRlCNNModelCfg):
    """Configuration for CNN model."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.cnn_velocity_model:CNNVelocityModel"
    """The model class name. Defaults to VelocityCNNModel."""



############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoVelocityAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_velocity:PPOVelocity"
    """The name of the Velocity Estimator PPO algorithm. Defaults to 'VelocityEstimatorPPOAlgorithm'"""


