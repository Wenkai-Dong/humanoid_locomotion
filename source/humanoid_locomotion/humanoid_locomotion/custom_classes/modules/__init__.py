# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .mha import MHA
from .cnn import CNN

__all__ = [
    "MHA",
    "CNN",
]
