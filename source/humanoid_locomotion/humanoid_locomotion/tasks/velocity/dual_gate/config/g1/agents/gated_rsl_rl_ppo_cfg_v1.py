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
    RslRlGatedModelCfgv1,
)
from humanoid_locomotion.tasks.velocity.dual_gate.mdp.symmetry import g1, g1_history


@configclass
class G1RoughPPORunnerKeyCfg(RslRlOnPolicyRunnerCfgNew):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 300
    experiment_name = "Dual-G1-v0/gated_key_v1"
    obs_groups = {
        "actor": ["actor", "actor_map"],
        "critic": ["critic", "critic_map"],
    }
    # wandb_project = "velocity"
    # logger = "wandb"
    torch_compile_mode = None   # "default", "max-autotune-no-cudagraphs"
    actor = RslRlGatedModelCfgv1(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=5,
            stride=1,
            dilation=1,
            padding="zeros",
            norm="layer",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=False,
        ),
        gated_position="key",
    )
    critic = RslRlGatedModelCfgv1(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=5,
            stride=1,
            dilation=1,
            padding="zeros",
            norm="layer",
            activation="elu",
            max_pool=False,
            global_pool="none",
            flatten=False,
        ),
        gated_position="key",
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

@configclass
class G1RoughPPORunnerValueCfg(G1RoughPPORunnerKeyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actor.gated_position = "value"
        self.critic.gated_position = "value"

@configclass
class G1RoughPPORunnerSDPACfg(G1RoughPPORunnerKeyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actor.gated_position = "sdpa"
        self.critic.gated_position = "sdpa"


@configclass
class G1RoughPPORunnerPoolCfg(G1RoughPPORunnerKeyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actor.gated_position = None
        self.critic.gated_position = None