# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .ppo_velocity import PPOVelocity
from .ppo_ae import PPOAE

__all__ = ["PPOVelocity", "PPOAE"]
