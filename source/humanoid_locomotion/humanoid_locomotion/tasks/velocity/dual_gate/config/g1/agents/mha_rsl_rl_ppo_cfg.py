# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
    RslRlSymmetryCfg,
    RslRlCNNModelCfg,
)
from humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.rl_cfg import (
    RslRlCNNVelocityModelCfg,
    RslRlPpoVelocityAlgorithmCfg,
    RslRlOnPolicyRunnerCfgNew,
)
from humanoid_locomotion.tasks.velocity.dual_gate.mdp.symmetry import g1, g1_history


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfgNew):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 50
    experiment_name = "dualgate_mha_g1_v0"
    obs_groups = {
        "actor": ["actor", "actor_map"],
        "critic": ["critic", "critic_map"],
    }
    torch_compile_mode = None
    actor = RslRlCNNVelocityModelCfg(
        class_name="humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.mha_model:MHAModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16,61],
            kernel_size=5,
            stride=1,
            dilation=1,
            padding="zeros",
            norm="layer",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=False,
        )
    )
    critic = RslRlCNNVelocityModelCfg(
        class_name="humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.mha_model:MHAModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 61],
            kernel_size=5,
            stride=1,
            dilation=1,
            padding="zeros",
            norm="layer",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=False,
        )
    )
    algorithm = RslRlPpoVelocityAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=4,
        num_mini_batches=3,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=g1_history.compute_symmetric_states
        ),
        share_cnn_encoders=False,
    )