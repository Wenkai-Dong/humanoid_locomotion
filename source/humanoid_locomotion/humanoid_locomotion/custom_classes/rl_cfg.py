# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.rl_cfg import RslRlMLPModelCfg

#########################
# Model configurations #
#########################


@configclass
class RslRlAME1ModelCfg(RslRlMLPModelCfg):
    """Configuration for CNN model."""

    class_name: str = "humanoid_locomotion.custom_classes.models.ame1_model:AME1Model"
    """The model class name. Default is AME1Model."""

    @configclass
    class CNNCfg:
        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

        kernel_size: int | tuple[int] | list[int] = MISSING
        """The kernel size for the CNN."""

        stride: int | tuple[int] | list[int] = 1
        """The stride for the CNN."""

        dilation: int | tuple[int] | list[int] = 1
        """The dilation for the CNN."""

        padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
        """The padding for the CNN."""

        norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = "none"
        """The normalization for the CNN."""

        activation: str = MISSING
        """The activation function for the CNN."""

        max_pool: bool | tuple[bool] | list[bool] = False
        """Whether to use max pooling for the CNN."""

        global_pool: Literal["none", "max", "avg"] = "none"
        """The global pooling for the CNN."""

        flatten: bool = True
        """Whether to flatten the output of the CNN."""

    @configclass
    class MHACfg:
        num_heads: int | tuple[int] | list[int] = MISSING
        """The number of heads for each convolutional layer for the MHA."""

        dropout: float | tuple[float] | list[float] = 0.0
        """Dropout probability on attn_output_weights. Default: 0.0 (no dropout)."""

        bias: bool | tuple[bool] | list[bool] = True
        """If specified, adds bias to input / output projection layers. Default: True."""

        attention_type: Literal["cross", "self"] | tuple[str] | list[str] = "cross"
        """The attention type for the CNN."""

        norm: Literal["none", "layer"] | tuple[str] | list[str] = "none"
        """The normalization for the MHA."""

        norm_position: Literal["pre_norm", "post_norm", "none"] | tuple[str] | list[str] = "none"
        """The normalization for the CNN."""

        activation: str | tuple[str] | list[str] = "identity"
        """The activation function for the MHA."""

        flatten: bool = True
        """Whether to flatten the output of the MHA."""

    @configclass
    class LinearCfg:
        out_features: int = MISSING
        """The number of output features."""

    cnn_cfg: CNNCfg = MISSING
    """The configuration for the CNN(s)."""

    mha_cfg: MHACfg = MISSING
    """The configuration for the MHA(s)."""

    linear_cfg: LinearCfg = None
    """The configuration for the MHA(s)."""


