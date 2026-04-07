# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Power operation scenarios (Category B).

B1: UPS Alarm Response (Medium)
    - UPS switches to battery after utility micro-outage
    - Agent must verify UPS status, check battery, ensure generator readiness
B3: Generator Test Protocol (Easy)
    - Monthly generator test — agent must follow proper procedure
    - Start generator, verify output, run loaded test, cooldown, shutdown
B4: Power Failure Cascade (Hard)
    - Full utility loss + generator fails to start
    - Agent must manage UPS battery time, shed load, troubleshoot generator
"""

from __future__ import annotations

from ..config import ASHRAE_CLASSES, DatacenterConfig
from ..simulation.thermal import ThermalSimulation
from ..simulation.power import PowerSimulation
from ..simulation.types import GeneratorState, UPSMode
from .base import ProcedureRule, Scenario, ScenarioResult
from .registry import register_scenario


# ===========================================================================
# B1: UPS Alarm Response (Medium)
# ===========================================================================
@register_scenario
class UPSAlarmResponse(Scenario):
    """Agent responds to UPS switching to battery.

    Scenario: A brief utility dip caused UPS to transfer to battery.
    Utility has been restored, but the agent should:
      1. Check UPS status (diagnose UPS-1)
      2. Verify battery SOC
      3. Verify generator is in standby and ready
      4. Verify ATS is back on utility
      5. Acknowledge the alarm

    Resolution: Agent diagnoses UPS AND acknowledges alarm.
    The system will self-recover, but proper procedure matters.
    """

    _BATTERY_DRAIN_SECONDS = 30  # Brief outage duration

    def __init__(self) -> None:
        super().__init__()
        self._diagnosed_ups = False
        self._acknowledged = False

    def reset_state(self) -> None:
        self._diagnosed_ups = False
        self._acknowledged = False

    def configure(self, base_config: DatacenterConfig) -> DatacenterConfig:
        return base_config

    def inject_fault(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
    ) -> None:
        if power_sim is None:
            return
        # Simulate a brief utility outage that has already ended
        # Drain some battery to show it was on battery
        for ups in power_sim.state.ups_units:
            ups.battery_soc = 0.85  # ~15% used during brief outage
            ups.mode = UPSMode.DOUBLE_CONVERSION  # Already back on utility

    def evaluate_step(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
        action_command: str,
        action_history: list[str],
        step: int,
    ) -> ScenarioResult:
        cmd = action_command.strip().lower()

        # Track diagnosis
        if cmd.startswith("diagnose") and "ups" in cmd:
            self._diagnosed_ups = True
        if cmd.startswith("acknowledge"):
            self._acknowledged = True

        resolved = self._diagnosed_ups and self._acknowledged

        # Reward for proper investigation
        scenario_reward = 0.0
        if self._diagnosed_ups:
            scenario_reward += 0.3
        if self._acknowledged:
            scenario_reward += 0.2

        procedure_reward = self.check_procedure(action_command, action_history)

        # Progress: 50% for diagnose, 50% for acknowledge
        progress = 0.0
        if self._diagnosed_ups:
            progress += 0.5
        if self._acknowledged:
            progress += 0.5

        return ScenarioResult(
            resolved=resolved,
            resolution_message="UPS alarm properly investigated and acknowledged." if resolved else "",
            scenario_reward=scenario_reward,
            procedure_reward=procedure_reward,
            progress=progress,
            info={
                "diagnosed_ups": self._diagnosed_ups,
                "acknowledged": self._acknowledged,
            },
        )

    @property
    def scenario_id(self) -> str:
        return "B1"

    @property
    def name(self) -> str:
        return "UPS Alarm Response"

    @property
    def scenario_type(self) -> str:
        return "power"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def step_budget(self) -> int:
        return 10

    @property
    def alert_message(self) -> str:
        return (
            "WARNING: UPS-1 transferred to battery at 14:23:05. "
            "Utility restored at 14:23:35. Battery SOC: 85%. "
            "Verify system status and acknowledge."
        )

    @property
    def procedure_rules(self) -> list[ProcedureRule]:
        return [
            ProcedureRule(
                required_before=["diagnose"],
                trigger_command="acknowledge_alarm",
                bonus=0.3,
                penalty=-0.2,
                description="Diagnose UPS before acknowledging alarm",
            ),
        ]


# ===========================================================================
# B3: Generator Test Protocol (Easy)
# ===========================================================================
@register_scenario
class GeneratorTestProtocol(Scenario):
    """Agent must follow proper monthly generator test procedure.

    Correct sequence:
      1. check_status — Review current system state
      2. start_generator — Initiate startup
      3. diagnose GEN-1 — Verify engine started and output is stable
      4. stop_generator — Initiate cooldown
      5. acknowledge_alarm — Log test completion

    Resolution: Generator successfully started, verified, and shut down.
    """

    def __init__(self) -> None:
        super().__init__()
        self._started = False
        self._verified = False
        self._stopped = False
        self._completed = False

    def reset_state(self) -> None:
        self._started = False
        self._verified = False
        self._stopped = False
        self._completed = False

    def configure(self, base_config: DatacenterConfig) -> DatacenterConfig:
        return base_config

    def inject_fault(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
    ) -> None:
        # No fault — this is a routine test procedure
        pass

    def evaluate_step(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
        action_command: str,
        action_history: list[str],
        step: int,
    ) -> ScenarioResult:
        cmd = action_command.strip().lower()

        if cmd.startswith("start_generator"):
            self._started = True
        if self._started and cmd.startswith("diagnose") and "gen" in cmd:
            # Only count as verified if generator is actually running
            if power_sim and power_sim.state.generator.state in (
                GeneratorState.READY, GeneratorState.LOADED,
                GeneratorState.WARMING, GeneratorState.CRANKING,
            ):
                self._verified = True
        if cmd.startswith("stop_generator"):
            if self._started and self._verified:
                self._stopped = True
        if cmd.startswith("acknowledge") and self._stopped:
            self._completed = True

        # Check generator state
        gen_running = False
        if power_sim:
            gen_running = power_sim.state.generator.state in (
                GeneratorState.READY, GeneratorState.LOADED,
                GeneratorState.WARMING, GeneratorState.CRANKING,
                GeneratorState.START_DELAY,
            )

        resolved = self._completed

        scenario_reward = 0.0
        if self._started:
            scenario_reward += 0.1
        if self._verified:
            scenario_reward += 0.2
        if self._stopped:
            scenario_reward += 0.2
        if self._completed:
            scenario_reward += 0.3

        procedure_reward = self.check_procedure(action_command, action_history)

        # Progress: 25% per protocol step
        progress = 0.0
        if self._started:
            progress += 0.25
        if self._verified:
            progress += 0.25
        if self._stopped:
            progress += 0.25
        if self._completed:
            progress += 0.25

        return ScenarioResult(
            resolved=resolved,
            resolution_message="Generator test protocol completed successfully." if resolved else "",
            scenario_reward=scenario_reward,
            procedure_reward=procedure_reward,
            progress=progress,
            info={
                "started": self._started,
                "verified": self._verified,
                "stopped": self._stopped,
                "completed": self._completed,
                "gen_running": gen_running,
            },
        )

    @property
    def scenario_id(self) -> str:
        return "B3"

    @property
    def name(self) -> str:
        return "Generator Test Protocol"

    @property
    def scenario_type(self) -> str:
        return "power"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def step_budget(self) -> int:
        return 15

    @property
    def alert_message(self) -> str:
        return (
            "SCHEDULED: Monthly generator test due. "
            "Follow standard test protocol: start, verify, loaded test, shutdown."
        )

    @property
    def procedure_rules(self) -> list[ProcedureRule]:
        return [
            ProcedureRule(
                required_before=["check_status"],
                trigger_command="start_generator",
                bonus=0.2,
                penalty=-0.1,
                description="Check system status before starting generator",
            ),
            ProcedureRule(
                required_before=["start_generator"],
                trigger_command="stop_generator",
                bonus=0.2,
                penalty=-0.3,
                description="Must start generator before stopping it",
            ),
        ]

    @property
    def game_time_per_step_s(self) -> float:
        # Generator startup is ~17s. At 10s/step the agent can observe
        # intermediate states (CRANKING, WARMING) across 2 steps.
        return 10.0


# ===========================================================================
# B4: Power Failure Cascade (Hard)
# ===========================================================================
@register_scenario
class PowerFailureCascade(Scenario):
    """Full utility loss with degraded generator response.

    Scenario: Utility power fails. Generator starts but takes longer than
    usual (warm-up extended). UPS batteries are bridging the gap.
    Meanwhile, battery SOC is dropping.

    The agent must:
      1. Diagnose UPS status and battery levels
      2. Verify generator startup sequence
      3. Shed non-critical IT load to extend battery life
      4. Monitor temperatures (no cooling compressors during transfer)
      5. Stabilize once generator is online

    Resolution: Generator loaded AND all temps within allowable limits
    AND UPS battery SOC stabilized (charging or >20%).
    """

    _CONSECUTIVE_STABLE_STEPS = 2

    def __init__(self) -> None:
        super().__init__()
        self._stable_count = 0

    def reset_state(self) -> None:
        self._stable_count = 0

    def configure(self, base_config: DatacenterConfig) -> DatacenterConfig:
        # Extend generator warmup to make it more challenging
        base_config.power.generator.warmup_time_s = 15.0  # Longer than default 8s
        return base_config

    def inject_fault(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
    ) -> None:
        if power_sim is None:
            return
        power_sim.set_utility_available(False)

    def evaluate_step(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
        action_command: str,
        action_history: list[str],
        step: int,
    ) -> ScenarioResult:
        dc = thermal_sim.state

        # Check temperatures
        all_within_allowable = True
        max_over = 0.0
        for zone in dc.zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if not ashrae:
                continue
            if zone.max_inlet_temp_c > ashrae.allowable_max_c:
                all_within_allowable = False
                max_over = max(max_over, zone.max_inlet_temp_c - ashrae.allowable_max_c)

        # Check power recovery
        gen_loaded = False
        battery_ok = True
        if power_sim:
            gen_loaded = power_sim.state.generator.state == GeneratorState.LOADED
            for ups in power_sim.state.ups_units:
                if ups.battery_soc < 0.10:
                    battery_ok = False

        stable = all_within_allowable and gen_loaded and battery_ok
        if stable:
            self._stable_count += 1
        else:
            self._stable_count = 0

        resolved = self._stable_count >= self._CONSECUTIVE_STABLE_STEPS

        # Reward shaping
        scenario_reward = 0.0
        # Penalty for temperature overshoot
        if max_over > 0:
            scenario_reward -= max_over * 1.5
        # Reward for generator online
        if gen_loaded:
            scenario_reward += 0.3
        # Penalty for low battery
        if power_sim:
            min_soc = min(u.battery_soc for u in power_sim.state.ups_units) if power_sim.state.ups_units else 1.0
            if min_soc < 0.20:
                scenario_reward -= (0.20 - min_soc) * 5.0

        procedure_reward = self.check_procedure(action_command, action_history)

        # Progress: partial credit per condition, full credit for stability
        conditions_met = sum([gen_loaded, all_within_allowable, battery_ok])
        if conditions_met == 3:
            progress = 0.5 + 0.5 * min(1.0, self._stable_count / self._CONSECUTIVE_STABLE_STEPS)
        else:
            progress = (conditions_met / 3.0) * 0.5

        return ScenarioResult(
            resolved=resolved,
            resolution_message="Power failure resolved. Generator online, temps stable." if resolved else "",
            scenario_reward=scenario_reward,
            procedure_reward=procedure_reward,
            progress=progress,
            info={
                "max_overshoot_c": max_over,
                "gen_loaded": gen_loaded,
                "battery_ok": battery_ok,
                "stable_count": self._stable_count,
            },
        )

    @property
    def scenario_id(self) -> str:
        return "B4"

    @property
    def name(self) -> str:
        return "Power Failure Cascade"

    @property
    def scenario_type(self) -> str:
        return "power"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def step_budget(self) -> int:
        return 20

    @property
    def alert_message(self) -> str:
        return (
            "CRITICAL: Utility power lost. UPS on battery. "
            "Generator startup in progress. "
            "Battery SOC declining. Immediate action required."
        )

    @property
    def procedure_rules(self) -> list[ProcedureRule]:
        return [
            ProcedureRule(
                required_before=["diagnose"],
                trigger_command="set_rack_load",
                bonus=0.3,
                penalty=-0.1,
                description="Diagnose before shedding load",
            ),
            ProcedureRule(
                required_before=[],
                trigger_command="escalate",
                bonus=0.0,
                penalty=-0.5,
                description="Escalation during power cascade is heavily penalized",
            ),
        ]

    @property
    def game_time_per_step_s(self) -> float:
        # Fast progression — every second counts with battery draining
        return 15.0
