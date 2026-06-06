# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import MLP
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState
from humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.modules import CNN1D


class SwAVModel(MLPModel):

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        self.temperature = 0.1
        # Resolve observation groups and dimensions
        if obs[obs_set].dim() == 3:
            obs_current = obs.clone()
            obs_current[obs_set] = obs_current[obs_set][:, -1]
        else:
            obs_current = obs
        obs_groups_1d, obs_dim_1d = self._get_obs_dim(obs_current, obs_groups, obs_set)

        # Create or validate CNN encoders
        if cnns is not None:
            # Check compatibility if CNNs are provided
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            # Create a cnn config for each 2D observation group in case only one is provided
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
            # Create CNNs for each 2D observation
            cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx] - 2,
                    **cnn_cfg[obs_group],
                )

        # Initialize the parent MLP model
        super().__init__(
            obs_current,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        # Initialize the weights of the MLP
        self.mlp.init_weights([2**0.5, None, 2**0.5, None, 2**0.5, None, 0.01])

        # Register CNN encoders
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)
        # Initialize the weights of the CNN
        for cnn in self.cnns.values():
            cnn.init_weights()

        # Initialize velocity estimator
        if obs_set == "actor":
            self.cnn1d = nn.Sequential(
                nn.Conv1d(
                    in_channels=obs["actor"].shape[-1], out_channels=obs["actor"].shape[-1],
                    kernel_size=3, stride=1, padding=1, groups=obs["actor"].shape[-1]
                ),
                nn.Conv1d(in_channels=obs["actor"].shape[-1], out_channels=128, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=128),
                nn.ELU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(num_groups=8, num_channels=128),
                nn.ELU(),
                nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=32),
                nn.ELU(),
                nn.Flatten(1),
                MLP(800, 256, (256, 256), activation, activation)
            )
            # Initialize the weights of the 1D-CNN
            for idx, module in enumerate(self.cnn1d):
                if isinstance(module, nn.Conv1d):
                    torch.nn.init.kaiming_normal_(module.weight)
                    torch.nn.init.zeros_(module.bias)  # type: ignore
            self.cnn1d[-1].init_weights([2**0.5, None, 2**0.5, None, 2**0.5, None, 2**0.5, None])

            self.encoder_velocity = MLP(256, 3, (64,), activation)
            self.encoder_velocity.init_weights([2**0.5, None, 0.001])
            self.encoder_left = MLP(256, 16, (128, 64), activation)
            self.encoder_left.init_weights([2**0.5, None, 2**0.5, None, 1.0])
            self.encoder_right = MLP(256, 16, (128, 64), activation)
            self.encoder_right.init_weights([2**0.5, None, 2**0.5, None, 1.0])
            self.target = MLP(30, 16, (128, 64), activation)
            self.target.init_weights([2**0.5, None, 2**0.5, None, 1.0])
            self.proto = nn.Embedding(32, 16)

        self.linear = nn.Linear(obs_dim_1d + 3, 64)
        self.linear_z = nn.Linear(16, 64)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)
        nn.init.orthogonal_(self.linear_z.weight, gain=1.0)
        nn.init.zeros_(self.linear_z.bias)

        self.position = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 16),
        )
        nn.init.orthogonal_(self.position[0].weight, gain=1.0)
        nn.init.orthogonal_(self.position[3].weight, gain=2 ** 0.5)
        nn.init.zeros_(self.position[0].bias)
        nn.init.zeros_(self.position[3].bias)

        self.dilated_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.LayerNorm([16, 13, 18]),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=6, dilation=3),
            nn.LayerNorm([16, 13, 18]),
            nn.ELU(),
        )
        for idx, module in enumerate(self.dilated_cnn):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)  # type: ignore

        self.q_norm = nn.LayerNorm(64)
        self.k_norm = nn.LayerNorm(64)
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=16, batch_first=True)
        self.o_norm = nn.LayerNorm(64)
        self.need_weights = False

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by combining normalized 1D and CNN-encoded 2D observation groups."""
        # get history observation
        if obs["actor"].dim() == 3:
            obs_current = obs.copy()
            obs_current["actor"] = obs_current["actor"][:, -1]
        else:
            obs_current = obs
        # Concatenate 1D observation groups and normalize
        latent_1d = super().get_latent(obs_current)
        # get velocity and privilege
        if self.obs_groups[0] == "actor":
            velocity, z_l, z_r = self.swav(obs)
            velocity, z_l, z_r = velocity.detach(), z_l.detach(), z_r.detach()
            z_l_linear = self.linear_z(z_l)    # (N, 64)
            z_r_linear = self.linear_z(z_r)    # (N, 64)
        latent_1d_linear = self.linear(torch.cat([velocity, latent_1d], dim=-1))    # (N, 64)
        latent_1d_encoder = torch.stack([latent_1d_linear, z_l_linear, z_r_linear], dim=1)    # (N, 3, 64)
        # Process 2D observation groups with CNNs
        latent_cnn_list = [self.cnns[obs_group](obs[obs_group][:, 2:3, ...]) for obs_group in self.obs_groups_2d]
        latent_cnn = torch.cat(latent_cnn_list, dim=-1) # (N, 32, 13, 18)
        latent_cnn = latent_cnn.flatten(2).permute(0, 2, 1) # (N, 234, 32)
        latent_dilated = self.dilated_cnn(obs["actor_map"][:, 2:3, ...])   # (N, 16, 13, 18)
        latent_dilated = latent_dilated.flatten(2).permute(0, 2, 1) # (N, 234, 16)
        latent_position = self.position(obs["actor_map"].flatten(2).permute(0, 2, 1))   # (N, 234, 16)
        latent_mapping = torch.cat([latent_cnn, latent_dilated, latent_position], dim=-1)   # (N, 234, 64)
        query = self.q_norm(latent_1d_encoder)
        key = self.k_norm(latent_mapping)
        latent_mha, self.attn_output_weights = self.mha(query, key, latent_mapping, need_weights=self.need_weights) #(N, 3, 64)
        latent_mha = self.o_norm(latent_mha).flatten(1)
        # Concatenate 1D and CNN latents
        return torch.cat([velocity, latent_1d, latent_mha, z_l, z_r], dim=-1)

    def swav(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ):
        """Build the base velocity by concatenating and normalizing selected observation groups."""
        # Normalize hitstory observation groups and normalize
        obs_history = (obs["actor"] - self.obs_normalizer._mean) / (self.obs_normalizer._std + self.obs_normalizer.eps)
        latent_history = self.cnn1d(obs_history.permute(0, 2, 1).contiguous())  # （N, 256)
        self.velocity = self.encoder_velocity(latent_history)   # (N, 3)
        z_l = F.normalize(self.encoder_left(latent_history), dim=-1, p=2)   # (N, 16)
        z_r = F.normalize(self.encoder_right(latent_history), dim=-1, p=2)  # (N, 16)
        self.z = torch.stack((z_l, z_r), dim=1).reshape(-1, 16) # (2N, 16)
        return self.velocity, z_l, z_r

    def update(self, privilege, vel):
        pred_vel = self.velocity.float()
        z_s = self.z.float()    # (2N, 16)
        z_t = F.normalize(self.target(privilege), dim=-1, p=2)  # (2N, 16)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T # (2N, 32)
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = self.sinkhorn(score_s)    # (2N, 32)
            q_t = self.sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1) # (2N, 32)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        # swav_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        swav_loss = -0.5 * ((q_s * log_p_t).sum(-1) + (q_t * log_p_s).sum(-1)).mean()
        velocity_loss = F.mse_loss(pred_vel, vel)

        with torch.no_grad():
            p = F.softmax(score_s / self.temperature, dim=-1)
            # H(C|X)
            H_cond = -(p * (p + 1e-9).log()).sum(-1).mean()
            # H(C)
            marginal = p.mean(0)
            H_marg = -(marginal * (marginal + 1e-9).log()).sum()
            # I(X;C)
            mutual_info = H_marg - H_cond
            # std of z
            z_std = z_s.std(0).mean()
        return velocity_loss, swav_loss, H_cond, H_marg, mutual_info, z_std

    @torch.no_grad()
    def sinkhorn(self, out, eps=0.05, iters=3):
        Q = torch.exp(out / eps).T
        K, B = Q.shape[0], Q.shape[1]
        Q /= Q.sum()

        for it in range(iters):
            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        return (Q * B).T

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchCNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCNNModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        # Iterate through active observation groups and separate 1D and 2D observations
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided. If this is intentional, use the MLP model instead.")

        # Store active 2D observation groups and dimensions directly as attributes
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d
        # Return active 1D observation groups and dimension for parent class
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        if self.obs_groups[0] == "actor":
            return self.obs_dim + 3 + 224
        else:
            return self.obs_dim + 64


class _TorchCNNModel(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: CNNModel) -> None:
        """Create a TorchScript-friendly copy of a CNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and 2D inputs."""
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_cnn_list.append(cnn(obs_2d[i]))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxCNNModel(nn.Module):
    """Exportable CNN model for ONNX."""

    def __init__(self, model: CNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
