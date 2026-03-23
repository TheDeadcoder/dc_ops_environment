# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Datacenter operation scenarios."""

from .base import ProcedureRule, Scenario, ScenarioResult
from .registry import (
    get_scenario,
    list_scenarios,
    random_scenario,
    register_scenario,
    registered_scenario_ids,
)

# Import scenario modules to trigger registration
from . import thermal_scenarios  # noqa: F401
from . import power_scenarios    # noqa: F401

__all__ = [
    "ProcedureRule",
    "Scenario",
    "ScenarioResult",
    "get_scenario",
    "list_scenarios",
    "random_scenario",
    "register_scenario",
    "registered_scenario_ids",
]
