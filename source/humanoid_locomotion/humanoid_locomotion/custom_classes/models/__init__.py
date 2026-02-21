# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .ame1_model import AME1Model
from .cnn_model import CNNModel

__all__ = [
    "AME1Model",
    "CNNModel"
]
