# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_velocity_model import CNNVelocityModel
from .mha_model import MHAModel

__all__ = [
    "CNNVelocityModel",
    "MHAModel",
]
