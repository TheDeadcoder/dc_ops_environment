# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract base class for datacenter operation scenarios.

A Scenario defines:
  - Initial datacenter configuration overrides
  - Fault injection (what goes wrong)
  - Available actions for the agent
  - Resolution criteria (how to "win")
  - Scenario-specific reward shaping
  - Procedural correctness rules (diagnose before repair, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import DatacenterConfig
from ..simulation.thermal import ThermalSimulation
from ..simulation.power import PowerSimulation


@dataclass
class ProcedureRule:
    """A procedural correctness rule for reward shaping.

    Attributes:
        required_before: Commands that must appear before `trigger_command`.
        trigger_command: The command this rule applies to.
        bonus: Reward bonus if required_before was satisfied.
        penalty: Reward penalty if trigger_command issued without required_before.
        description: Human-readable explanation.
    """
    required_before: list[str]
    trigger_command: str
    bonus: float = 0.3
    penalty: float = -0.2
    description: str = ""


@dataclass
class ScenarioResult:
    """Outcome of checking scenario state after a step.

    Attributes:
        resolved: True if the incident is successfully resolved.
        resolution_message: Human-readable message on resolution.
        scenario_reward: Legacy scenario-specific reward (kept for compat).
        procedure_reward: Procedural correctness reward from check_procedure().
        progress: Normalized [0, 1] progress toward resolution.
            Used by the delta-based reward function for credit assignment.
        info: Additional scenario-specific data for logging.
    """
    resolved: bool = False
    resolution_message: str = ""
    scenario_reward: float = 0.0
    procedure_reward: float = 0.0
    progress: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class Scenario(ABC):
    """Abstract base class for datacenter operation scenarios.

    Lifecycle:
      1. Environment calls `configure(config)` to get modified DatacenterConfig
      2. Environment calls `inject_fault(thermal_sim, power_sim)` after warmup
      3. Each step, environment calls `evaluate_step(...)` for reward + resolution
      4. Environment uses `alert_message`, `step_budget`, etc. for episode control

    Subclasses must implement all abstract methods/properties.
    """

    @abstractmethod
    def configure(self, base_config: DatacenterConfig) -> DatacenterConfig:
        """Optionally modify the datacenter configuration for this scenario.

        Override to change rack loads, outside temperature, number of CRACs, etc.
        Return the base_config unchanged if no modifications needed.
        """

    def reset_state(self) -> None:
        """Reset mutable episode state between episodes.

        Called by the environment at the start of each episode, before
        configure() / inject_fault(). Subclasses with mutable state
        (counters, flags) MUST override this and reset them.
        """

    @abstractmethod
    def inject_fault(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
    ) -> None:
        """Inject the fault or initial condition into the running simulation.

        Called after warmup, so the datacenter is at quasi-steady-state.
        """

    @abstractmethod
    def evaluate_step(
        self,
        thermal_sim: ThermalSimulation,
        power_sim: PowerSimulation | None,
        action_command: str,
        action_history: list[str],
        step: int,
    ) -> ScenarioResult:
        """Evaluate the current state after a step.

        Returns ScenarioResult with:
          - resolved: True if the incident is successfully resolved
          - scenario_reward: Scenario-specific reward component
          - procedure_reward: Procedural correctness reward
        """

    @property
    @abstractmethod
    def scenario_id(self) -> str:
        """Unique identifier, e.g. 'A1', 'B4'."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scenario name."""

    @property
    @abstractmethod
    def scenario_type(self) -> str:
        """Category: 'thermal', 'power', 'network', 'incident'."""

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """'easy', 'medium', 'hard'."""

    @property
    @abstractmethod
    def step_budget(self) -> int:
        """Maximum steps allowed for this scenario."""

    @property
    @abstractmethod
    def alert_message(self) -> str:
        """Initial alert shown to the agent."""

    @property
    def game_time_per_step_s(self) -> float:
        """Simulation time per agent step. Override for faster/slower scenarios."""
        return 60.0

    @property
    def procedure_rules(self) -> list[ProcedureRule]:
        """Procedural correctness rules. Override to define scenario-specific rules."""
        return []

    @property
    def available_actions(self) -> list[str] | None:
        """Override to restrict available actions. None = all actions available."""
        return None

    def check_procedure(self, action_command: str, action_history: list[str]) -> float:
        """Check procedural correctness of the current action against history.

        Returns reward bonus/penalty based on whether required prerequisites
        were satisfied before the current action.
        """
        if not self.procedure_rules:
            return 0.0

        # Extract just the command name (first word)
        cmd_name = action_command.strip().split()[0].lower() if action_command.strip() else ""
        history_cmds = [h.strip().split()[0].lower() for h in action_history[:-1] if h.strip()]

        reward = 0.0
        for rule in self.procedure_rules:
            if cmd_name == rule.trigger_command:
                # Check if all required_before commands appeared in history
                all_satisfied = all(
                    any(req == h for h in history_cmds)
                    for req in rule.required_before
                )
                if all_satisfied:
                    reward += rule.bonus
                else:
                    reward += rule.penalty
        return reward
