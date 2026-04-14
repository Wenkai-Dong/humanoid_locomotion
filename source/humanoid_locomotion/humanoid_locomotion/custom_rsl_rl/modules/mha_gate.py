# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from torch import nn as nn
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter

from .linear import NonDynamicallyQuantizableLinear
from .module import Module

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
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        kdim=None,
        vdim=None,
        device=None,
        dtype=None,
        activation=None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.g_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((4 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
            self.register_parameter("g_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(4 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        self.activation_function = resolve_nn_activation(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
            xavier_uniform_(self.g_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:

        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(
                _arg_requires_grad(x) for x in tensor_args
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> tuple[Optional[Tensor], Optional[int]]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

        self.need_weights: bool = False
        self.attn_weights = torch.zeros(0)


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

        if self.need_weights:
            self.attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)

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
