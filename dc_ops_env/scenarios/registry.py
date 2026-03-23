# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario registry for selecting scenarios by ID, type, or difficulty.
"""

from __future__ import annotations

import random
from typing import Optional

from .base import Scenario


# Global registry: scenario_id → Scenario class
_REGISTRY: dict[str, type[Scenario]] = {}


def register_scenario(cls: type[Scenario]) -> type[Scenario]:
    """Class decorator to register a scenario.

    Usage:
        @register_scenario
        class MyCoolScenario(Scenario):
            ...
    """
    # Instantiate temporarily to read scenario_id
    instance = cls()
    _REGISTRY[instance.scenario_id] = cls
    return cls


def get_scenario(scenario_id: str) -> Scenario:
    """Get a scenario by its ID (e.g. 'A1', 'B4')."""
    cls = _REGISTRY.get(scenario_id)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {available}")
    return cls()


def list_scenarios(
    *,
    scenario_type: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> list[Scenario]:
    """List registered scenarios, optionally filtered by type or difficulty."""
    result = []
    for cls in _REGISTRY.values():
        instance = cls()
        if scenario_type and instance.scenario_type != scenario_type:
            continue
        if difficulty and instance.difficulty != difficulty:
            continue
        result.append(instance)
    return result


def random_scenario(
    *,
    scenario_type: Optional[str] = None,
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
) -> Scenario:
    """Pick a random scenario from the registry, optionally filtered."""
    candidates = list_scenarios(scenario_type=scenario_type, difficulty=difficulty)
    if not candidates:
        raise ValueError(
            f"No scenarios match type={scenario_type!r}, difficulty={difficulty!r}"
        )
    rng = random.Random(seed)
    return rng.choice(candidates)


def registered_scenario_ids() -> list[str]:
    """Return all registered scenario IDs in sorted order."""
    return sorted(_REGISTRY.keys())
