# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DC-Ops Environment Implementation.

Wires the thermal and power simulations into OpenEnv's Environment interface.
Each step:
  1. Parse the agent's command
  2. Apply mutations to simulation state
  3. Advance simulation by game-time dt (default 60s)
  4. Render dashboard observation
  5. Compute reward (via multi-objective RewardFunction)
  6. Check termination conditions
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..config import (
        ASHRAE_CLASSES,
        DatacenterConfig,
        PowerConfig,
        make_default_datacenter_config,
        load_datacenter_config,
    )
    from ..models import DcOpsAction, DcOpsObservation
    from ..actions.parser import AVAILABLE_ACTIONS, CommandResult, parse_command
    from ..rendering.dashboard import render_dashboard
    from ..simulation.thermal import ThermalAlarm, ThermalSimulation
    from ..simulation.power import PowerAlarm, PowerSimulation
    from ..scenarios.base import Scenario, ScenarioResult
    from ..scenarios.registry import get_scenario, random_scenario
    from ..rewards.reward_function import RewardFunction
except ImportError:
    from config import (
        ASHRAE_CLASSES,
        DatacenterConfig,
        PowerConfig,
        make_default_datacenter_config,
        load_datacenter_config,
    )
    from models import DcOpsAction, DcOpsObservation
    from actions.parser import AVAILABLE_ACTIONS, CommandResult, parse_command
    from rendering.dashboard import render_dashboard
    from simulation.thermal import ThermalAlarm, ThermalSimulation
    from simulation.power import PowerAlarm, PowerSimulation
    from scenarios.base import Scenario, ScenarioResult
    from scenarios.registry import get_scenario, random_scenario
    from rewards.reward_function import RewardFunction


# Default episode configuration
DEFAULT_STEP_BUDGET = 15
DEFAULT_GAME_TIME_PER_STEP_S = 60.0  # 1 minute of sim time per agent step
DEFAULT_SIM_DT_S = 1.0               # Physics integration timestep


class DcOpsEnvironment(Environment):
    """Datacenter operations environment for LLM-based RL agents.

    The agent observes a text-based monitoring dashboard and issues
    natural-language operator commands. The environment simulates
    physics-based thermal and power dynamics.

    Episode flow:
      1. reset() initializes the datacenter and optionally injects a fault
      2. step() parses the command, advances simulation, returns dashboard
      3. Episode ends on: budget exhaustion, critical failure, escalation, or resolution
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._thermal_sim: ThermalSimulation | None = None
        self._power_sim: PowerSimulation | None = None
        self._config: DatacenterConfig | None = None
        self._scenario: Scenario | None = None
        self._reward_fn: RewardFunction | None = None
        self._step_budget: int = DEFAULT_STEP_BUDGET
        self._game_time_per_step_s: float = DEFAULT_GAME_TIME_PER_STEP_S
        self._sim_dt_s: float = DEFAULT_SIM_DT_S
        self._alert: str = ""
        self._scenario_type: str = ""
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._action_history: list[str] = []
        self._escalated: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DcOpsObservation:
        """Reset the environment and return initial observation.

        Kwargs:
            scenario (str | Scenario): Scenario ID (e.g. 'A1') or Scenario instance.
                If provided, overrides config/alert/step_budget/scenario_type.
                If not provided, uses raw kwargs (backward compatible).
            config (DatacenterConfig): Custom datacenter configuration.
            config_name (str): Built-in config name ("default", "small", "large").
                Used when config is not provided (e.g. from WebSocket/HTTP JSON).
            step_budget (int): Max steps for the episode.
            game_time_per_step_s (float): Simulation time per step.
            scenario_type (str): Scenario category label.
            alert (str): Initial alert message.
            fault_injection (dict): Fault to inject, e.g.
                {"type": "crac_fault", "unit_id": "CRAC-3", "fault": "compressor"}
        """
        # Episode state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._done = False
        self._cumulative_reward = 0.0
        self._action_history = []
        self._escalated = False

        # Resolve scenario
        scenario_arg = kwargs.get("scenario")
        if isinstance(scenario_arg, str):
            self._scenario = get_scenario(scenario_arg)
        elif isinstance(scenario_arg, Scenario):
            self._scenario = scenario_arg
        elif scenario_arg is None and kwargs.get("random_scenario"):
            self._scenario = random_scenario(
                scenario_type=kwargs.get("scenario_type"),
                difficulty=kwargs.get("difficulty"),
                seed=seed,
            )
        else:
            self._scenario = None

        # Reset scenario mutable state (counters, flags) for episode reuse
        if self._scenario:
            self._scenario.reset_state()

        # Configuration — scenario can modify the base config
        # Support config_name (string) from JSON APIs, or config (DatacenterConfig) from Python
        config_arg = kwargs.get("config")
        config_name = kwargs.get("config_name")
        if isinstance(config_arg, DatacenterConfig):
            self._config = config_arg
        elif config_name and isinstance(config_name, str) and config_name != "default":
            self._config = load_datacenter_config(config_name)
        else:
            self._config = make_default_datacenter_config()
        if self._scenario:
            self._config = self._scenario.configure(self._config)

        # Episode parameters — scenario provides defaults, kwargs can override
        if self._scenario:
            self._step_budget = kwargs.get("step_budget", self._scenario.step_budget)
            self._game_time_per_step_s = kwargs.get(
                "game_time_per_step_s", self._scenario.game_time_per_step_s
            )
            self._scenario_type = kwargs.get("scenario_type", self._scenario.scenario_type)
            self._alert = kwargs.get("alert", self._scenario.alert_message)
        else:
            self._step_budget = kwargs.get("step_budget", DEFAULT_STEP_BUDGET)
            self._game_time_per_step_s = kwargs.get("game_time_per_step_s", DEFAULT_GAME_TIME_PER_STEP_S)
            self._scenario_type = kwargs.get("scenario_type", "")
            self._alert = kwargs.get("alert", "")

        self._sim_dt_s = self._config.simulation_dt_s

        # Initialize reward function with scenario-type-aware weights
        self._reward_fn = RewardFunction(scenario_type=self._scenario_type)

        # Initialize simulations
        self._thermal_sim = ThermalSimulation(self._config)

        # Initialize power sim if config has power infrastructure
        if self._config.power and self._config.power.ups_units:
            it_load = self._thermal_sim.state.total_it_load_kw
            self._power_sim = PowerSimulation(self._config.power, it_load_kw=it_load)
            # Wire power state into datacenter state
            self._thermal_sim.state.power = self._power_sim.state
        else:
            self._power_sim = None

        # Apply fault injection — scenario or raw kwargs
        if self._scenario:
            # Warmup FIRST, then inject fault (so DC is at steady-state)
            self._warmup_simulation()
            self._scenario.inject_fault(self._thermal_sim, self._power_sim)
        else:
            fault = kwargs.get("fault_injection")
            if fault:
                self._apply_fault_injection(fault)
            self._warmup_simulation()

        # Render initial observation
        return self._make_observation(action_result="Environment initialized. Awaiting your command.")

    def step(
        self,
        action: DcOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DcOpsObservation:
        """Execute one agent step.

        1. Parse and execute the command
        2. Advance simulation by game_time_per_step_s
        3. Check for alarms and termination
        4. Compute reward via RewardFunction
        5. Return observation
        """
        if self._done:
            return self._make_observation(
                action_result="Episode already ended. Call reset().",
                reward=0.0,
            )

        self._state.step_count += 1
        self._action_history.append(action.command)

        # 1. Parse and execute command
        cmd_result = parse_command(
            action.command,
            self._thermal_sim,
            self._power_sim,
        )

        # Handle escalation
        if cmd_result.command_name == "escalate":
            self._escalated = True
            self._done = True
            # Evaluate scenario for procedure penalties
            scenario_result: ScenarioResult | None = None
            if self._scenario:
                scenario_result = self._scenario.evaluate_step(
                    self._thermal_sim, self._power_sim,
                    action.command, self._action_history,
                    self._state.step_count,
                )
            # Compute base reward components — escalation penalty is handled
            # by scenario procedure rules + action_quality, not doubled here
            components = self._reward_fn.compute(
                self._thermal_sim, self._power_sim, cmd_result,
                action.command, self._action_history, scenario_result,
            )
            reward = components.total
            self._cumulative_reward += reward
            return self._make_observation(
                action_result=cmd_result.message,
                reward=reward,
            )

        # 2. Handle acknowledge_alarm — clear alert before new alarms overwrite
        if cmd_result.command_name == "acknowledge_alarm" and cmd_result.success:
            self._alert = ""

        # 3. Advance simulation
        thermal_alarms, power_alarms = self._advance_simulation()

        # 4. Build alert from alarms (only new critical/warning alarms override)
        self._update_alert(thermal_alarms, power_alarms)

        # 5. Evaluate scenario (before reward, so progress is available)
        scenario_result = None
        if self._scenario:
            scenario_result = self._scenario.evaluate_step(
                self._thermal_sim, self._power_sim,
                action.command, self._action_history,
                self._state.step_count,
            )

        # 6. Compute reward via RewardFunction
        components = self._reward_fn.compute(
            self._thermal_sim, self._power_sim, cmd_result,
            action.command, self._action_history, scenario_result,
        )
        reward = components.total

        self._cumulative_reward += reward

        # 7. Check termination
        self._check_termination(thermal_alarms, power_alarms)

        # 7b. Scenario resolution
        if scenario_result and scenario_result.resolved and not self._done:
            self._done = True
            # Speed bonus: fraction of budget remaining
            speed_bonus = (self._step_budget - self._state.step_count) / self._step_budget
            reward += speed_bonus
            self._cumulative_reward += speed_bonus
            if scenario_result.resolution_message:
                self._alert = scenario_result.resolution_message

        return self._make_observation(
            action_result=cmd_result.message,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state

    # -------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------
    def _warmup_simulation(self, warmup_steps: int = 120) -> None:
        """Run simulation for a brief warmup to reach quasi-steady-state."""
        for _ in range(warmup_steps):
            self._thermal_sim.step(self._sim_dt_s)
            if self._power_sim:
                it_load = self._thermal_sim.state.total_it_load_kw
                self._power_sim.step(self._sim_dt_s, it_load)

    def _advance_simulation(self) -> tuple[list[ThermalAlarm], list[PowerAlarm]]:
        """Advance simulation by game_time_per_step_s seconds."""
        n_substeps = int(self._game_time_per_step_s / self._sim_dt_s)
        all_thermal_alarms: list[ThermalAlarm] = []
        all_power_alarms: list[PowerAlarm] = []

        for _ in range(n_substeps):
            # Thermal step
            thermal_result = self._thermal_sim.step(self._sim_dt_s)
            all_thermal_alarms.extend(thermal_result.alarms)

            # Power step
            if self._power_sim:
                it_load = self._thermal_sim.state.total_it_load_kw
                power_result = self._power_sim.step(self._sim_dt_s, it_load)
                all_power_alarms.extend(power_result.alarms)

        # Deduplicate alarms by type (keep most recent)
        thermal_alarms = _dedupe_alarms_by_type(all_thermal_alarms)
        power_alarms = _dedupe_alarms_by_type(all_power_alarms)

        return thermal_alarms, power_alarms

    def _update_alert(
        self,
        thermal_alarms: list[ThermalAlarm],
        power_alarms: list[PowerAlarm],
    ) -> None:
        """Update the active alert string from current alarms."""
        critical_messages: list[str] = []

        for alarm in thermal_alarms:
            if alarm.severity == "critical":
                critical_messages.append(alarm.message)

        for alarm in power_alarms:
            if alarm.severity == "critical":
                critical_messages.append(alarm.message)

        if critical_messages:
            self._alert = " | ".join(critical_messages[:3])  # Limit to 3 alerts
        else:
            # Check for warnings
            warnings = []
            for alarm in thermal_alarms:
                if alarm.severity == "warning":
                    warnings.append(alarm.message)
            for alarm in power_alarms:
                if alarm.severity == "warning":
                    warnings.append(alarm.message)
            if warnings:
                self._alert = warnings[0]
            else:
                self._alert = ""

    def _check_termination(
        self,
        thermal_alarms: list[ThermalAlarm],
        power_alarms: list[PowerAlarm],
    ) -> None:
        """Check if episode should end."""
        # Step budget exhausted
        if self._state.step_count >= self._step_budget:
            self._done = True
            return

        # Critical thermal failure: any rack above allowable max
        for zone in self._thermal_sim.state.zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if not ashrae:
                continue
            if zone.max_inlet_temp_c > ashrae.allowable_max_c + 5.0:
                self._done = True
                self._alert = (
                    f"CRITICAL: Zone {zone.zone_id} inlet temp "
                    f"{zone.max_inlet_temp_c:.1f}°C exceeds allowable max "
                    f"{ashrae.allowable_max_c:.1f}°C by >5°C. Emergency shutdown."
                )
                return

        # UPS battery exhausted
        if self._power_sim:
            for ups in self._power_sim.state.ups_units:
                if ups.mode.value == "fault" and ups.battery_soc <= 0:
                    self._done = True
                    self._alert = f"CRITICAL: {ups.unit_id} battery exhausted. Unprotected load."
                    return

    def _apply_fault_injection(self, fault: dict) -> None:
        """Apply a fault injection to the simulation.

        Supported fault types:
          - crac_fault: {"type": "crac_fault", "unit_id": "CRAC-3", "fault": "compressor"}
          - utility_loss: {"type": "utility_loss"}
          - ups_fault: {"type": "ups_fault", "unit_id": "UPS-1"}
          - rack_load_change: {"type": "rack_load_change", "rack_id": "A-01", "load_kw": 15.0}
          - outside_temp: {"type": "outside_temp", "temp_c": 42.0}
        """
        fault_type = fault.get("type", "")

        if fault_type == "crac_fault":
            from ..simulation.types import CRACFaultType
            unit_id = fault.get("unit_id", "")
            fault_name = fault.get("fault", "compressor")
            try:
                ft = CRACFaultType(fault_name)
            except ValueError:
                ft = CRACFaultType.COMPRESSOR
            self._thermal_sim.inject_crac_fault(unit_id, ft)

        elif fault_type == "utility_loss":
            if self._power_sim:
                self._power_sim.set_utility_available(False)

        elif fault_type == "ups_fault":
            if self._power_sim:
                unit_id = fault.get("unit_id", "")
                self._power_sim.inject_ups_fault(unit_id)

        elif fault_type == "rack_load_change":
            rack_id = fault.get("rack_id", "")
            load_kw = fault.get("load_kw", 8.0)
            self._thermal_sim.set_rack_load(rack_id, load_kw)

        elif fault_type == "outside_temp":
            temp_c = fault.get("temp_c", 35.0)
            self._thermal_sim.set_outside_temp(temp_c)

    def _make_observation(
        self,
        action_result: str = "",
        reward: float = 0.0,
    ) -> DcOpsObservation:
        """Build the observation to return to the agent."""
        dashboard = render_dashboard(
            self._thermal_sim.state,
            alert=self._alert,
            step=self._state.step_count,
            max_steps=self._step_budget,
            scenario_type=self._scenario_type,
        )

        steps_remaining = max(0, self._step_budget - self._state.step_count)

        # Build metadata with structured data
        dc_state = self._thermal_sim.state
        metadata = {
            "sim_time_s": dc_state.sim_time_s,
            "total_it_load_kw": dc_state.total_it_load_kw,
            "total_cooling_power_kw": dc_state.total_cooling_power_kw,
            "pue": dc_state.pue,
            "outside_temp_c": dc_state.outside_temp_c,
            "cumulative_reward": self._cumulative_reward,
            "zones": {},
        }

        for zone in dc_state.zones:
            metadata["zones"][zone.zone_id] = {
                "cold_aisle_temp_c": zone.cold_aisle_temp_c,
                "hot_aisle_temp_c": zone.hot_aisle_temp_c,
                "max_inlet_temp_c": zone.max_inlet_temp_c,
                "total_it_load_kw": zone.total_it_load_kw,
            }

        if self._power_sim:
            power = self._power_sim.state
            metadata["power"] = {
                "utility_available": power.utility_available,
                "on_generator": power.on_generator,
                "total_ups_loss_kw": power.total_ups_loss_kw,
                "total_pdu_loss_kw": power.total_pdu_loss_kw,
            }
            for ups in power.ups_units:
                metadata["power"][ups.unit_id] = {
                    "mode": ups.mode.value,
                    "battery_soc": ups.battery_soc,
                    "load_fraction": ups.load_fraction,
                    "efficiency": ups.efficiency,
                }

        if self._scenario:
            metadata["scenario"] = {
                "id": self._scenario.scenario_id,
                "name": self._scenario.name,
                "difficulty": self._scenario.difficulty,
            }

        # Use scenario-specific actions if defined, otherwise all actions
        actions = AVAILABLE_ACTIONS
        if self._scenario and self._scenario.available_actions is not None:
            actions = self._scenario.available_actions

        return DcOpsObservation(
            dashboard=dashboard,
            available_actions=actions,
            alert=self._alert,
            scenario_type=self._scenario_type,
            steps_remaining=steps_remaining,
            action_result=action_result,
            done=self._done,
            reward=reward,
            metadata=metadata,
        )


def _dedupe_alarms_by_type(alarms: list) -> list:
    """Keep only the last alarm of each (component, alarm_type) pair."""
    seen: dict[tuple[str, str], Any] = {}
    for alarm in alarms:
        key = (getattr(alarm, "component", ""), getattr(alarm, "alarm_type", ""))
        seen[key] = alarm
    return list(seen.values())
