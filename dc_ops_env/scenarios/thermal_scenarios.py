# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Thermal operation scenarios (Category A).

A1: Cooling Setpoint Optimization (Easy)
    - PUE is high because setpoints are too low
    - Agent must raise setpoints to improve efficiency without violating ASHRAE
A2: Thermal Event Response (Medium)
    - Single CRAC failure causes temperature rise
    - Agent must diagnose, compensate, and stabilize
A4: CRAC Failure Cascade (Hard)
    - Two CRACs fail in quick succession
    - Agent must triage, redistribute cooling, migrate workload
"""

from __future__ import annotations

from ..config import ASHRAE_CLASSES, DatacenterConfig
from ..simulation.thermal import ThermalSimulation
from ..simulation.power import PowerSimulation
from ..simulation.types import CRACFaultType
from .base import ProcedureRule, Scenario, ScenarioResult
from .registry import register_scenario


# ===========================================================================
# A1: Cooling Setpoint Optimization (Easy)
# ===========================================================================
@register_scenario
class CoolingSetpointOptimization(Scenario):
    """Agent must optimize CRAC setpoints to reduce PUE.

    Initial condition: All CRACs at 15°C setpoint (overly aggressive cooling).
    This wastes energy — PUE is unnecessarily high.

    Goal: Raise setpoints closer to ASHRAE recommended range (18-27°C for A2)
    while keeping all inlet temps within recommended limits.

    Resolution: PUE drops below target AND all temps within recommended range.
    """

    _PUE_TARGET = 1.6  # Achievable PUE with proper setpoints

    def configure(self, base_config: DatacenterConfig) -> DatacenterConfig:
        # Set all CRACs to 15°C (too cold, wasting energy)
        for zone_cfg in base_config.zones:
            for crac_cfg in zone_cfg.crac_units:
                crac_cfg.initial_setpoint_c = 15.0
        return base_config

    def inject_fault(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
    ) -> None:
        # No fault — this is an optimization scenario
        # The "problem" is already baked into the config (low setpoints)
        pass

    def evaluate_step(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
        action_command: str,
        action_history: list[str],
        step: int,
    ) -> ScenarioResult:
        dc = thermal_sim.state
        pue = dc.pue

        # Check all zones within recommended
        all_within_recommended = True
        for zone in dc.zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if ashrae and zone.max_inlet_temp_c > ashrae.recommended_max_c:
                all_within_recommended = False
                break

        # Reward: improvement toward target PUE
        # Baseline PUE at 15°C setpoints is ~2.0+, target is ~1.6
        pue_reward = max(0, 2.0 - pue)  # Higher is better as PUE drops

        resolved = pue < self._PUE_TARGET and all_within_recommended
        procedure_reward = self.check_procedure(action_command, action_history)

        # Progress: PUE improvement toward target + temperature compliance
        pue_progress = max(0.0, min(1.0, (2.0 - pue) / (2.0 - self._PUE_TARGET)))
        temp_factor = 1.0 if all_within_recommended else 0.0
        progress = 0.7 * pue_progress + 0.3 * temp_factor

        return ScenarioResult(
            resolved=resolved,
            resolution_message="PUE optimized within target range." if resolved else "",
            scenario_reward=pue_reward * 0.5,
            procedure_reward=procedure_reward,
            progress=progress,
            info={"pue": pue, "target_pue": self._PUE_TARGET},
        )

    @property
    def scenario_id(self) -> str:
        return "A1"

    @property
    def name(self) -> str:
        return "Cooling Setpoint Optimization"

    @property
    def scenario_type(self) -> str:
        return "thermal"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def step_budget(self) -> int:
        return 10

    @property
    def alert_message(self) -> str:
        return (
            "NOTICE: PUE exceeds 1.8 — cooling setpoints may be suboptimal. "
            "Review CRAC setpoints and adjust for energy efficiency."
        )

    @property
    def procedure_rules(self) -> list[ProcedureRule]:
        return [
            ProcedureRule(
                required_before=["check_status"],
                trigger_command="adjust_setpoint",
                bonus=0.2,
                penalty=-0.1,
                description="Check status before adjusting setpoints",
            ),
        ]


# ===========================================================================
# A2: Thermal Event Response (Medium)
# ===========================================================================
@register_scenario
class ThermalEventResponse(Scenario):
    """Agent must respond to a single CRAC compressor failure.

    A CRAC unit suffers a compressor failure, reducing cooling capacity.
    With N+1 provisioning the remaining CRACs can handle the load,
    but the agent should:
      1. Diagnose the failed unit
      2. Increase fan speeds or adjust setpoints on remaining CRACs
      3. Optionally reduce load on hottest racks

    Resolution: All inlet temps within recommended range for 2+ consecutive steps.
    """

    _FAILED_UNIT = "CRAC-3"
    _CONSECUTIVE_STABLE_STEPS = 2
    _MIN_STEPS_BEFORE_RESOLUTION = 8  # Agent must take at least 8 actions

    def __init__(self) -> None:
        super().__init__()
        self._stable_count = 0
        self._diagnosed_fault = False  # Must diagnose the faulty unit

    def reset_state(self) -> None:
        self._stable_count = 0
        self._diagnosed_fault = False

    def configure(self, base_config: DatacenterConfig) -> DatacenterConfig:
        return base_config

    def inject_fault(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
    ) -> None:
        thermal_sim.inject_crac_fault(self._FAILED_UNIT, CRACFaultType.COMPRESSOR)

    def evaluate_step(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
        action_command: str,
        action_history: list[str],
        step: int,
    ) -> ScenarioResult:
        dc = thermal_sim.state

        # Track if agent diagnosed the faulty unit
        cmd_parts = action_command.strip().split()
        if (len(cmd_parts) >= 2 and cmd_parts[0].lower() == "diagnose"
                and cmd_parts[1].upper() == self._FAILED_UNIT):
            self._diagnosed_fault = True

        # Check if all zones within recommended
        all_within_recommended = True
        max_over = 0.0
        for zone in dc.zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if not ashrae:
                continue
            if zone.max_inlet_temp_c > ashrae.recommended_max_c:
                all_within_recommended = False
                max_over = max(max_over, zone.max_inlet_temp_c - ashrae.recommended_max_c)

        if all_within_recommended:
            self._stable_count += 1
        else:
            self._stable_count = 0

        # Resolution requires:
        #   1. Agent diagnosed the faulty unit (proper procedure)
        #   2. Temps stable for N consecutive steps
        #   3. At least _MIN_STEPS_BEFORE_RESOLUTION steps taken
        resolved = (
            self._diagnosed_fault
            and self._stable_count >= self._CONSECUTIVE_STABLE_STEPS
            and step >= self._MIN_STEPS_BEFORE_RESOLUTION
        )

        # Scenario reward: penalty proportional to temperature overshoot
        scenario_reward = -max_over * 0.5 if max_over > 0 else 0.1

        procedure_reward = self.check_procedure(action_command, action_history)

        # Progress: partial credit for being close, full credit for stability
        if all_within_recommended:
            progress = 0.5 + 0.5 * min(1.0, self._stable_count / self._CONSECUTIVE_STABLE_STEPS)
        else:
            progress = max(0.0, 0.4 / (1.0 + max_over))

        return ScenarioResult(
            resolved=resolved,
            resolution_message="Thermal event stabilized. All zones within recommended range." if resolved else "",
            scenario_reward=scenario_reward,
            procedure_reward=procedure_reward,
            progress=progress,
            info={"max_overshoot_c": max_over, "stable_count": self._stable_count},
        )

    @property
    def scenario_id(self) -> str:
        return "A2"

    @property
    def name(self) -> str:
        return "Thermal Event Response"

    @property
    def scenario_type(self) -> str:
        return "thermal"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def step_budget(self) -> int:
        return 15

    @property
    def alert_message(self) -> str:
        return (
            f"CRITICAL: {self._FAILED_UNIT} compressor failure detected. "
            "Zone B temperatures rising. Investigate and stabilize."
        )

    @property
    def procedure_rules(self) -> list[ProcedureRule]:
        return [
            ProcedureRule(
                required_before=["diagnose"],
                trigger_command="adjust_setpoint",
                bonus=0.3,
                penalty=-0.2,
                description="Diagnose the fault before adjusting setpoints",
            ),
            ProcedureRule(
                required_before=["diagnose"],
                trigger_command="set_fan_speed",
                bonus=0.3,
                penalty=-0.2,
                description="Diagnose the fault before adjusting fan speed",
            ),
            ProcedureRule(
                required_before=[],
                trigger_command="escalate",
                bonus=0.0,
                penalty=-0.3,
                description="Escalated without attempting diagnosis or fix",
            ),
        ]


# ===========================================================================
# A4: CRAC Failure Cascade (Hard)
# ===========================================================================
@register_scenario
class CRACFailureCascade(Scenario):
    """Two CRACs fail, overwhelming remaining cooling capacity.

    CRAC-1 has a compressor failure and CRAC-3 has a fan failure.
    With only 2 of 4 CRACs operational, cooling is severely degraded.
    The agent must:
      1. Diagnose both failures
      2. Aggressively compensate (max fan speeds, lower setpoints on survivors)
      3. Reduce IT load on hottest racks (workload migration)
      4. Monitor and stabilize before thermal runaway

    Resolution: All inlet temps below allowable max for 2+ steps.
    """

    _FAILED_UNITS = [
        ("CRAC-1", CRACFaultType.COMPRESSOR),
        ("CRAC-3", CRACFaultType.FAN),
    ]
    _CONSECUTIVE_STABLE_STEPS = 2
    _MIN_STEPS_BEFORE_RESOLUTION = 8  # Hard scenario needs investigation time

    def __init__(self) -> None:
        super().__init__()
        self._stable_count = 0

    def reset_state(self) -> None:
        self._stable_count = 0

    def configure(self, base_config: DatacenterConfig) -> DatacenterConfig:
        return base_config

    def inject_fault(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
    ) -> None:
        for unit_id, fault_type in self._FAILED_UNITS:
            thermal_sim.inject_crac_fault(unit_id, fault_type)

    def evaluate_step(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
        action_command: str,
        action_history: list[str],
        step: int,
    ) -> ScenarioResult:
        dc = thermal_sim.state

        all_within_allowable = True
        max_over = 0.0
        for zone in dc.zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if not ashrae:
                continue
            if zone.max_inlet_temp_c > ashrae.allowable_max_c:
                all_within_allowable = False
                max_over = max(max_over, zone.max_inlet_temp_c - ashrae.allowable_max_c)

        if all_within_allowable:
            self._stable_count += 1
        else:
            self._stable_count = 0

        # Bonus for diagnosing both units
        diagnosed_units = set()
        for h in action_history:
            parts = h.strip().split()
            if len(parts) >= 2 and parts[0].lower() == "diagnose":
                diagnosed_units.add(parts[1].upper())

        resolved = (
            self._stable_count >= self._CONSECUTIVE_STABLE_STEPS
            and "CRAC-1" in diagnosed_units
            and "CRAC-3" in diagnosed_units
            and step >= self._MIN_STEPS_BEFORE_RESOLUTION
        )

        # Heavy penalty for being over allowable
        scenario_reward = -max_over * 2.0 if max_over > 0 else 0.2

        procedure_reward = self.check_procedure(action_command, action_history)

        if "CRAC-1" in diagnosed_units and "CRAC-3" in diagnosed_units:
            procedure_reward += 0.2  # Bonus for thorough diagnosis

        # Progress: partial credit for being close, full credit for stability
        if all_within_allowable:
            progress = 0.5 + 0.5 * min(1.0, self._stable_count / self._CONSECUTIVE_STABLE_STEPS)
        else:
            progress = max(0.0, 0.4 / (1.0 + max_over))

        return ScenarioResult(
            resolved=resolved,
            resolution_message="CRAC cascade stabilized. Temps within allowable range." if resolved else "",
            scenario_reward=scenario_reward,
            procedure_reward=procedure_reward,
            progress=progress,
            info={
                "max_overshoot_c": max_over,
                "stable_count": self._stable_count,
                "diagnosed_units": list(diagnosed_units),
            },
        )

    @property
    def scenario_id(self) -> str:
        return "A4"

    @property
    def name(self) -> str:
        return "CRAC Failure Cascade"

    @property
    def scenario_type(self) -> str:
        return "thermal"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def step_budget(self) -> int:
        return 20

    @property
    def alert_message(self) -> str:
        return (
            "CRITICAL: Multiple CRAC failures detected. "
            "CRAC-1 compressor fault, CRAC-3 fan fault. "
            "Temperatures rising rapidly. Immediate action required."
        )

    @property
    def procedure_rules(self) -> list[ProcedureRule]:
        return [
            ProcedureRule(
                required_before=["diagnose"],
                trigger_command="adjust_setpoint",
                bonus=0.2,
                penalty=-0.3,
                description="Diagnose before adjusting setpoints during cascade",
            ),
            ProcedureRule(
                required_before=["diagnose"],
                trigger_command="set_fan_speed",
                bonus=0.2,
                penalty=-0.3,
                description="Diagnose before adjusting fan speed during cascade",
            ),
            ProcedureRule(
                required_before=["diagnose"],
                trigger_command="set_rack_load",
                bonus=0.3,
                penalty=-0.1,
                description="Diagnose before migrating workloads",
            ),
        ]

    @property
    def game_time_per_step_s(self) -> float:
        # Faster time progression — cascade is urgent
        return 30.0
