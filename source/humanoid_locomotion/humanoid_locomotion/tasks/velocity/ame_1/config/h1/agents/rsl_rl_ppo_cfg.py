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

from humanoid_locomotion.tasks.velocity.ame_1.mdp.symmetry import h1

@configclass
class H1Stage1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "ame1_stage1_h1_v0"
    obs_groups = {
        "actor": ["policy", "policy_map"],
        "critic": ["policy", "policy_map"],
    }
    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="log",
        state_dependent_std=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16,61],
            kernel_size=5,
            stride=1,
            dilation=1,
            padding="zeros",
            norm="none",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=True,
        )
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 61],
            kernel_size=5,
            stride=1,
            dilation=1,
            padding="zeros",
            norm="none",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=True,
        )
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders = True,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=h1.compute_symmetric_states
        ),
    )


@configclass
class H1Stage2PPORunnerCfg(H1Stage1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "ame1_stage2_h1_v0"
        self.algorithm.entropy_coef=0.002
