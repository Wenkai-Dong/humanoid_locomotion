# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

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

@configclass
class RslRlGatedModelCfgv1(RslRlCNNModelCfg):
    """Configuration for CNN model."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.gated_model_v1:GatedMHAModel"

    gated_position: Literal["key", "value", "sdpa", "dense"] | None = MISSING



############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoVelocityAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_velocity:PPOVelocity"
    """The name of the Velocity Estimator PPO algorithm. Defaults to 'VelocityEstimatorPPOAlgorithm'"""

@configclass
class RslRlPpoAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_ae:PPOAE"
    """The name of the Velocity Estimator PPO algorithm. Defaults to 'VelocityEstimatorPPOAlgorithm'"""

@configclass
class RslRlPpoVAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_vae:PPOVAE"

@configclass
class RslRlPpoSwAVAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_swav:PPOSwAV"

#########################
# Runner configurations #
#########################


@configclass
class RslRlOnPolicyRunnerCfgNew(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    torch_compile_mode: Literal["default", "max-autotune-no-cudagraphs"] | None = None