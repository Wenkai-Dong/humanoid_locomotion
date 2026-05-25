# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_velocity_model import CNNVelocityModel
from .mha_model import MHAModel
from .gated_model import GatedMHAModel
from .ae_model import AEModel
from .vae_model import VAEModel
from .swav_model import SwAVModel

__all__ = [
    "CNNVelocityModel",
    "MHAModel",
    "GatedMHAModel",
    "AEModel",
    "VAEModel",
    "SwAVModel",
]
