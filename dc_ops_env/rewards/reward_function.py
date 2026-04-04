# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-objective reward function for DC-Ops environment.

Research-informed design:
  - Softplus barrier functions for safety constraints
    (Google/DeepMind 2017, ICLR 2025 DC Cooling)
  - Delta-based progress rewards for credit assignment
    (process reward model literature)
  - Normalized components in [-1, 1] via tanh
  - Scenario-type-aware weight profiles

All components are bounded to [-1, 1]. Total reward is the weighted sum,
clamped to [-1, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..config import ASHRAE_CLASSES
from ..simulation.thermal import ThermalSimulation
from ..simulation.power import PowerSimulation
from ..simulation.types import UPSMode
from ..actions.parser import CommandResult
from ..scenarios.base import ScenarioResult


# ---------------------------------------------------------------------------
# Numerically stable softplus
# ---------------------------------------------------------------------------
def softplus(x: float) -> float:
    """Numerically stable softplus: ln(1 + exp(x)).

    - x > 20:  returns x        (avoids exp overflow)
    - x < -20: returns 0.0      (avoids underflow noise)
    """
    if x > 20.0:
        return x
    if x < -20.0:
        return 0.0
    return math.log1p(math.exp(x))


# ---------------------------------------------------------------------------
# Reward components dataclass
# ---------------------------------------------------------------------------
@dataclass
class RewardComponents:
    """Individual reward components for logging and analysis."""

    thermal_safety: float = 0.0
    power_safety: float = 0.0
    efficiency: float = 0.0
    scenario_progress: float = 0.0
    procedure: float = 0.0
    action_quality: float = 0.0
    speed_bonus: float = 0.0
    total: float = 0.0


# ---------------------------------------------------------------------------
# Weight profiles
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    """Weights for reward components. Should sum to 1.0."""

    thermal_safety: float = 0.30
    power_safety: float = 0.10
    efficiency: float = 0.15
    scenario_progress: float = 0.25
    procedure: float = 0.15
    action_quality: float = 0.05


WEIGHT_PROFILES: dict[str, RewardWeights] = {
    "thermal": RewardWeights(
        thermal_safety=0.30,
        power_safety=0.05,
        efficiency=0.10,
        scenario_progress=0.30,
        procedure=0.20,
        action_quality=0.05,
    ),
    "power": RewardWeights(
        thermal_safety=0.10,
        power_safety=0.25,
        efficiency=0.05,
        scenario_progress=0.30,
        procedure=0.25,
        action_quality=0.05,
    ),
    "default": RewardWeights(
        thermal_safety=0.30,
        power_safety=0.15,
        efficiency=0.25,
        scenario_progress=0.0,
        procedure=0.0,
        action_quality=0.30,
    ),
}


# ---------------------------------------------------------------------------
# Softplus barrier constants
# ---------------------------------------------------------------------------
# Thermal barriers
_ALPHA_RECOMMENDED = 2.0   # °C transition width at recommended limit
_ALPHA_ALLOWABLE = 1.5     # °C transition width at allowable limit
_ALLOWABLE_WEIGHT = 3.0    # Allowable violations 3x worse per degree
_THERMAL_NORM = 8.0        # Normalization so T=40°C (A2) → R≈-0.97

# Thermal safety positive baseline — small reward for being well within limits
# Based on DCRL-Green (ICLR 2025): agents learn faster with a positive signal
# for maintaining safe state, not just penalties for violations.
_SAFE_MARGIN_C = 3.0       # °C below recommended max to qualify as "safe"
_SAFE_BASELINE = 0.1       # Small positive reward when all zones safe

# Power barriers
_SOC_THRESHOLD = 0.5       # Concern increases below 50% SOC
_SOC_ALPHA = 0.15          # Sharp transition around threshold
_UPS_FAULT_PENALTY = 5.0   # Fixed penalty for UPS fault
_POWER_NORM = 4.0          # Normalization constant

# Efficiency
_PUE_NORM = 2.0            # PUE sensitivity: PUE=3.0 → R≈-0.76

# Action quality
_REPEAT_WHITELIST = frozenset({"wait", "check_status"})


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------
class RewardFunction:
    """Composable, research-informed reward function for DC operations.

    Usage:
        rf = RewardFunction(scenario_type="thermal")
        rf.reset()   # Call at episode start

        # Each step:
        components = rf.compute(thermal_sim, power_sim, cmd_result,
                                action_command, action_history, scenario_result)
        reward = components.total
    """

    def __init__(
        self,
        scenario_type: str = "default",
        weights: Optional[RewardWeights] = None,
    ) -> None:
        self._scenario_type = scenario_type
        self._weights = weights or WEIGHT_PROFILES.get(
            scenario_type, WEIGHT_PROFILES["default"]
        )
        self._prev_progress: float = 0.0

    def reset(self) -> None:
        """Reset state between episodes."""
        self._prev_progress = 0.0

    def compute(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: Optional[PowerSimulation],
        cmd_result: CommandResult,
        action_command: str,
        action_history: list[str],
        scenario_result: Optional[ScenarioResult],
    ) -> RewardComponents:
        """Compute all reward components and weighted total.

        Returns RewardComponents with per-component values and total.
        Total is clamped to [-1, 1].
        """
        r_thermal = self._thermal_safety(thermal_sim)
        r_power = self._power_safety(power_sim)
        r_efficiency = self._efficiency(thermal_sim, power_sim)
        r_progress = self._scenario_progress(scenario_result)
        r_procedure = self._procedure(scenario_result)
        r_action = self._action_quality(
            cmd_result, action_command, action_history, thermal_sim, power_sim,
        )

        w = self._weights
        total = (
            w.thermal_safety * r_thermal
            + w.power_safety * r_power
            + w.efficiency * r_efficiency
            + w.scenario_progress * r_progress
            + w.procedure * r_procedure
            + w.action_quality * r_action
        )

        total = max(-1.0, min(1.0, total))

        return RewardComponents(
            thermal_safety=r_thermal,
            power_safety=r_power,
            efficiency=r_efficiency,
            scenario_progress=r_progress,
            procedure=r_procedure,
            action_quality=r_action,
            total=total,
        )

    # -------------------------------------------------------------------
    # Component implementations
    # -------------------------------------------------------------------

    @staticmethod
    def _thermal_safety(thermal_sim: ThermalSimulation) -> float:
        """ASHRAE compliance via dual softplus barriers.

        Returns value in [-1, _SAFE_BASELINE].
        Two barriers per zone: recommended (gentle) and allowable (steep).
        Averaged across zones so the signal is independent of zone count.

        Positive baseline (+0.1) when ALL zones are well within safe range
        (>= _SAFE_MARGIN_C below recommended max). This provides gradient
        signal for maintaining good state, not just avoiding violations.
        (Informed by DCRL-Green, ICLR 2025.)
        """
        zones = thermal_sim.state.zones
        if not zones:
            return 0.0

        n_zones = len(zones)
        penalty = 0.0
        all_safe = True

        for zone in zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if not ashrae:
                continue

            t = zone.max_inlet_temp_c
            rec_max = ashrae.recommended_max_c
            allow_max = ashrae.allowable_max_c

            # Check if zone is well within safe range
            if t > rec_max - _SAFE_MARGIN_C:
                all_safe = False

            # Soft barrier at recommended limit
            penalty += softplus((t - rec_max) / _ALPHA_RECOMMENDED) / n_zones
            # Harder barrier at allowable limit
            penalty += (
                _ALLOWABLE_WEIGHT
                * softplus((t - allow_max) / _ALPHA_ALLOWABLE)
                / n_zones
            )

        if penalty < 1e-6 and all_safe:
            return _SAFE_BASELINE

        return -math.tanh(penalty / _THERMAL_NORM)

    @staticmethod
    def _power_safety(power_sim: Optional[PowerSimulation]) -> float:
        """UPS battery and fault condition penalty.

        Returns value in [-1, 0].
        Penalty compounds across multiple failing UPS units.
        """
        if power_sim is None:
            return 0.0

        penalty = 0.0
        for ups in power_sim.state.ups_units:
            if ups.mode == UPSMode.ON_BATTERY:
                penalty += softplus((_SOC_THRESHOLD - ups.battery_soc) / _SOC_ALPHA)
            elif ups.mode == UPSMode.FAULT:
                penalty += _UPS_FAULT_PENALTY

        return -math.tanh(penalty / _POWER_NORM)

    @staticmethod
    def _efficiency(
        thermal_sim: ThermalSimulation,
        power_sim: Optional[PowerSimulation],
    ) -> float:
        """PUE-based energy efficiency penalty.

        Returns value in [-1, 0].
        PUE 1.0 (ideal) → 0, PUE 2.0 → -0.46, PUE 3.0 → -0.76.

        During power emergencies (UPS on battery), efficiency is suppressed
        to zero — the agent should not be penalized for load shedding that
        increases PUE but correctly preserves battery life.
        """
        # Suppress efficiency signal during power emergencies
        if power_sim is not None:
            for ups in power_sim.state.ups_units:
                if ups.mode in (UPSMode.ON_BATTERY, UPSMode.FAULT):
                    return 0.0

        pue = thermal_sim.state.pue
        return -math.tanh((pue - 1.0) / _PUE_NORM)

    def _scenario_progress(self, scenario_result: Optional[ScenarioResult]) -> float:
        """Delta-based progress toward scenario resolution.

        Returns value in [-1, 1].
        Rewards the CHANGE in progress — gives credit to the action that
        actually caused forward progress.
        """
        if scenario_result is None:
            return 0.0

        current = scenario_result.progress
        delta = current - self._prev_progress
        self._prev_progress = current

        return max(-1.0, min(1.0, delta))

    @staticmethod
    def _procedure(scenario_result: Optional[ScenarioResult]) -> float:
        """Procedural correctness from scenario rules.

        Returns value in [-1, 1].
        """
        if scenario_result is None:
            return 0.0
        return max(-1.0, min(1.0, scenario_result.procedure_reward))

    @staticmethod
    def _action_quality(
        cmd_result: CommandResult,
        action_command: str,
        action_history: list[str],
        thermal_sim: ThermalSimulation,
        power_sim: Optional[PowerSimulation],
    ) -> float:
        """Action quality assessment.

        Returns value in [-1, 1].
        Considers: validity, repetition, action type, urgency context.
        """
        if not cmd_result.success:
            return -0.5

        cmd_lower = action_command.strip().lower()
        name = cmd_result.command_name

        # Check for exact repeated command — but whitelist commands that
        # are legitimately repeatable (wait, check_status).
        if name not in _REPEAT_WHITELIST:
            prior = (
                [h.strip().lower() for h in action_history[:-1]]
                if len(action_history) > 1
                else []
            )
            if cmd_lower in prior:
                return -0.2

        # "wait" quality depends on whether there's an active concern
        if name == "wait":
            if _has_active_concern(thermal_sim, power_sim):
                # Waiting during a power event where we're waiting for
                # generator startup is acceptable — check if generator
                # is in startup sequence.
                if power_sim is not None and _generator_starting(power_sim):
                    return 0.1  # Waiting for gen to warm up is reasonable
                return -0.2  # Waiting during a thermal problem
            return 0.0  # Nothing wrong, waiting is fine

        # Information-gathering actions are valuable
        if name in ("diagnose", "check_status"):
            return 0.3

        # Active interventions
        if name in (
            "adjust_setpoint", "set_fan_speed", "set_rack_load",
            "migrate_workload", "start_generator", "stop_generator",
            "set_ups_mode", "start_crac", "stop_crac", "refuel_generator",
        ):
            return 0.2

        # Administrative
        if name == "acknowledge_alarm":
            return 0.1

        # Escalation — handled solely by scenario procedure rules now,
        # no extra penalty here. The environment no longer double-penalizes.
        if name == "escalate":
            return -0.1

        return 0.1  # Other valid commands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_active_concern(
    thermal_sim: ThermalSimulation,
    power_sim: Optional[PowerSimulation],
) -> bool:
    """Check if there is an active thermal or power concern."""
    for zone in thermal_sim.state.zones:
        ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
        if ashrae and zone.max_inlet_temp_c > ashrae.recommended_max_c:
            return True

    if power_sim:
        for ups in power_sim.state.ups_units:
            if ups.mode == UPSMode.ON_BATTERY:
                return True

    return False


def _generator_starting(power_sim: PowerSimulation) -> bool:
    """Check if the generator is in a startup sequence (agent should wait)."""
    from ..simulation.types import GeneratorState
    return power_sim.state.generator.state in (
        GeneratorState.START_DELAY,
        GeneratorState.CRANKING,
        GeneratorState.WARMING,
    )
