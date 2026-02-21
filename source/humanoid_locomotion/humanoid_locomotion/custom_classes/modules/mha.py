# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn as nn
import torch.nn.functional as F
from typing import Optional

from rsl_rl.utils import get_param, resolve_nn_activation


class MHA(nn.Module):
    """Multi-Head Attention Network (MHA).

    The MHA network is a sequence of Multi-Head Attention, Add layers, Norm layers and optional activation functions.
    The final output can be flattened.
    """

    def __init__(
        self,
        input_dim_q: int,
        input_channels_q: int,
        input_dim_kv: int,
        num_heads: int | tuple[int] | list[int],
        dropout: float | tuple[float] | list[float] = 0.0,
        bias: bool | tuple[bool] | list[bool] = True,
        attention_type: str | tuple[str] | list[str] = "cross",
        norm: str | tuple[str] | list[str] = "layer",
        norm_position: str | tuple[str] | list[str] = "pre_norm",
        activation: str | tuple[str] | list[str] = "sigmoid",
        flatten: bool = True,
    ) -> None:
        """Initialize the CNN.

        Args:
            input_dim: Height and width of the input.
            input_channels: Number of input channels.
            output_channels: List of output channels for each convolutional layer.
            kernel_size: List of kernel sizes for each convolutional layer or a single kernel size for all layers.
            stride: List of strides for each convolutional layer or a single stride for all layers.
            dilation: List of dilations for each convolutional layer or a single dilation for all layers.
            padding: Padding type to use. Either 'none', 'zeros', 'reflect', 'replicate', or 'circular'.
            norm: List of normalization types for each convolutional layer or a single type for all layers. Either
                'none', 'batch', or 'layer'.
            activation: Activation function to use.
            max_pool: List of booleans indicating whether to apply max pooling after each convolutional layer or a
                single boolean for all layers.
            global_pool: Global pooling type to apply at the end. Either 'none', 'max', or 'avg'.
            flatten: Whether to flatten the output tensor.
        """
        super().__init__()

        self.attention_type = [attention_type] if isinstance(attention_type, str) else attention_type
        self.norm_position = [get_param(norm_position, i) for i in range(len(self.attention_type))]
        self.input_dim_kv = input_dim_kv
        self.flatten = flatten

        self.mhas = nn.ModuleList()
        self.norm_kvs = nn.ModuleList()
        self.norm_qs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for idx in range(len(self.attention_type)):
            # Get parameters for the current layer
            h = get_param(num_heads, idx)
            d = get_param(dropout, idx)
            b = get_param(bias, idx)
            a = get_param(activation, idx)
            a_type = get_param(attention_type, idx)
            n = get_param(norm, idx)
            p = get_param(norm_position, idx)
            vdim = kdim = input_dim_kv if a_type == "cross" else None

            # Append multihead attention layer
            self.mhas.append(
                MHABlock(embed_dim=input_dim_q, num_heads=h, dropout=d, bias=b, kdim=kdim, vdim=vdim, activation=a)
            )

            # Append normalization layer if specified
            if n == "layer":
                if a_type == "cross" and p == "pre_norm":
                    self.norm_kvs.append(nn.LayerNorm(input_dim_kv))
                    self.norm_qs.append(nn.LayerNorm(input_dim_q))
                    self.norms.append(nn.Identity())
                else:
                    self.norm_kvs.append(nn.Identity())
                    self.norm_qs.append(nn.Identity())
                    self.norms.append(nn.LayerNorm(input_dim_q))
            elif n == "none":
                self.norm_kvs.append(nn.Identity())
                self.norm_qs.append(nn.Identity())
                self.norms.append(nn.Identity())

        # Apply flattening if specified
        self.flatten_layer = nn.Flatten(start_dim=1) if flatten else nn.Identity()

        # Store final output dimension
        self._output_channels = input_channels_q if not flatten else None
        self._output_dim = input_dim_q if not flatten else input_channels_q * input_dim_q

    @property
    def output_channels(self) -> int | None:
        """Get the number of output channels or None if output is flattened."""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int] | int:
        """Get the output height and width or total output dimension if output is flattened."""
        return self._output_dim

    def forward(self, q: torch.Tensor, kv_origin: Optional[torch.Tensor]=None,) -> torch.Tensor:
        """Forward pass of the MHA."""
        for idx, (mha, norm_kv, norm_q, norm) in enumerate(zip(self.mhas, self.norm_kvs, self.norm_qs, self.norms)):
            a_type = self.attention_type[idx]
            n_pos = self.norm_position[idx]

            if a_type == "self":
                kv = q
            else:
                assert kv_origin is not None, "Cross attention requires kv_origin"
                kv = kv_origin
            residual = q

            if n_pos == "pre_norm":
                if a_type == "cross":
                    kv = norm_kv(kv)
                    q = norm_q(q)
                elif a_type == "self":
                    kv = q = norm(q)
                q = mha(q, kv)
                q = q + residual
            elif n_pos == "post_norm":
                q = mha(q, kv)
                q = q + residual
                q = norm(q)
            elif n_pos == "none":
                q = mha(q, kv)
                q = q + residual

        q = self.flatten_layer(q)
        return q

class MHABlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias, kdim=None, vdim=None, activation=None) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.kdim = kdim if kdim is not None else self.embed_dim
        self.vdim = vdim if vdim is not None else self.embed_dim

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_k = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.w_v = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_p = dropout
        self.activation = activation
        if self.activation != "identity":
            self.w_theta = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.w_theta = nn.Identity()
        self.activation_function = resolve_nn_activation(activation)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the MutiHead Attention with xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(
            self,
            query: torch.Tensor,
            key_value: torch.Tensor,
    ):
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key_value.shape
        q = self.w_q(query)
        k = self.w_k(key_value)
        v = self.w_v(key_value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        o = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.dropout_p if self.training else 0.0,
                                           )
        o = o.transpose(1, 2).flatten(2)

        if self.activation != "identity":
            gate = self.activation_function(self.w_theta(query))
            o = gate * o
        else:
            o = self.activation_function(o)

        output = self.w_o(o)
        return output
