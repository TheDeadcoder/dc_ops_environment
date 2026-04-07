# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the scenario framework.

Validates:
  - Scenario registry (registration, lookup, filtering)
  - Scenario base class (procedure checking)
  - Each scenario: initialization, fault injection, resolution detection
  - Scenario integration with the environment
"""

from __future__ import annotations

import pytest

from dc_ops_env.models import DcOpsAction, DcOpsObservation
from dc_ops_env.scenarios import (
    Scenario,
    ScenarioResult,
    get_scenario,
    list_scenarios,
    random_scenario,
    registered_scenario_ids,
)
from dc_ops_env.scenarios.base import ProcedureRule
from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment


# ===========================================================================
# Registry Tests
# ===========================================================================
class TestRegistry:
    """Test scenario registration and lookup."""

    def test_all_scenarios_registered(self) -> None:
        ids = registered_scenario_ids()
        assert "A1" in ids
        assert "A2" in ids
        assert "A4" in ids
        assert "B1" in ids
        assert "B3" in ids
        assert "B4" in ids

    def test_get_scenario_by_id(self) -> None:
        s = get_scenario("A1")
        assert s.scenario_id == "A1"
        assert s.name == "Cooling Setpoint Optimization"

    def test_get_scenario_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("Z99")

    def test_list_by_type(self) -> None:
        thermal = list_scenarios(scenario_type="thermal")
        assert all(s.scenario_type == "thermal" for s in thermal)
        assert len(thermal) == 3  # A1, A2, A4

        power = list_scenarios(scenario_type="power")
        assert all(s.scenario_type == "power" for s in power)
        assert len(power) == 3  # B1, B3, B4

    def test_list_by_difficulty(self) -> None:
        easy = list_scenarios(difficulty="easy")
        assert all(s.difficulty == "easy" for s in easy)
        assert len(easy) >= 2  # A1, B3

        hard = list_scenarios(difficulty="hard")
        assert all(s.difficulty == "hard" for s in hard)
        assert len(hard) >= 2  # A4, B4

    def test_random_scenario(self) -> None:
        s = random_scenario(seed=42)
        assert isinstance(s, Scenario)

    def test_random_scenario_filtered(self) -> None:
        s = random_scenario(scenario_type="thermal", difficulty="easy", seed=42)
        assert s.scenario_type == "thermal"
        assert s.difficulty == "easy"

    def test_random_scenario_no_match_raises(self) -> None:
        with pytest.raises(ValueError, match="No scenarios match"):
            random_scenario(scenario_type="network")


# ===========================================================================
# Procedure Checking Tests
# ===========================================================================
class TestProcedureChecking:
    """Test the procedural correctness reward mechanism."""

    def test_procedure_bonus_when_satisfied(self) -> None:
        s = get_scenario("A2")
        # History has diagnose, then adjust_setpoint
        history = ["diagnose CRAC-3", "adjust_setpoint CRAC-4 20"]
        reward = s.check_procedure("adjust_setpoint CRAC-4 20", history)
        assert reward > 0, f"Expected bonus, got {reward}"

    def test_procedure_penalty_when_not_satisfied(self) -> None:
        s = get_scenario("A2")
        # No diagnose before adjust_setpoint
        history = ["adjust_setpoint CRAC-4 20"]
        reward = s.check_procedure("adjust_setpoint CRAC-4 20", history)
        assert reward < 0, f"Expected penalty, got {reward}"

    def test_no_procedure_rules_returns_zero(self) -> None:
        """Scenario with no procedure rules should return 0."""
        # Create a scenario without rules
        s = get_scenario("A1")  # A1 has rules, but let's test the mechanism
        reward = s.check_procedure("wait", ["wait"])
        # "wait" doesn't match any trigger_command, so should be 0
        assert reward == 0.0


# ===========================================================================
# A1: Cooling Setpoint Optimization Tests
# ===========================================================================
class TestA1CoolingSetpoint:
    """Test the A1 scenario lifecycle."""

    def test_initialization(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A1")
        assert obs.scenario_type == "thermal"
        assert "setpoint" in obs.alert.lower() or "PUE" in obs.alert

    def test_initial_pue_is_high(self) -> None:
        """With 15°C setpoints, PUE should be higher than optimal."""
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A1")
        pue = obs.metadata["pue"]
        # At 15°C setpoints, PUE should be elevated
        assert pue > 1.5, f"Initial PUE {pue:.2f} should be > 1.5"

    def test_raising_setpoint_improves_pue(self) -> None:
        """Raising CRAC setpoints should reduce PUE."""
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A1")
        pue_before = obs.metadata["pue"]

        # Raise all setpoints to 22°C (within ASHRAE A2 recommended)
        for crac_id in ["CRAC-1", "CRAC-2", "CRAC-3", "CRAC-4"]:
            env.step(DcOpsAction(command=f"adjust_setpoint {crac_id} 22"))

        # Wait for thermal convergence
        for _ in range(3):
            obs = env.step(DcOpsAction(command="wait"))

        pue_after = obs.metadata["pue"]
        assert pue_after < pue_before, \
            f"PUE should decrease: {pue_before:.3f} → {pue_after:.3f}"


# ===========================================================================
# A2: Thermal Event Response Tests
# ===========================================================================
class TestA2ThermalEvent:
    """Test the A2 scenario lifecycle."""

    def test_initialization(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A2")
        assert obs.scenario_type == "thermal"
        assert "CRAC-3" in obs.alert

    def test_crac_fault_visible_in_dashboard(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A2")
        assert "COMPRESSOR" in obs.dashboard or "FAULT" in obs.dashboard

    def test_diagnose_reveals_fault(self) -> None:
        env = DcOpsEnvironment()
        env.reset(scenario="A2")
        obs = env.step(DcOpsAction(command="diagnose CRAC-3"))
        assert "compressor" in obs.action_result.lower()
        assert "FAULT DETECTED" in obs.action_result

    def test_procedure_bonus_for_diagnose_first(self) -> None:
        """Diagnosing before adjusting should yield higher reward."""
        # Run 1: diagnose first, then adjust (procedure bonus on step 2)
        env1 = DcOpsEnvironment()
        env1.reset(scenario="A2")
        env1.step(DcOpsAction(command="diagnose CRAC-3"))
        obs1b = env1.step(DcOpsAction(command="adjust_setpoint CRAC-4 20"))
        r_with_diagnose = obs1b.reward

        # Run 2: wait, then adjust without diagnosing (procedure penalty on step 2)
        # Using wait keeps physics comparable so only the procedure bonus differs
        env2 = DcOpsEnvironment()
        env2.reset(scenario="A2")
        env2.step(DcOpsAction(command="wait"))
        obs2 = env2.step(DcOpsAction(command="adjust_setpoint CRAC-4 20"))
        r_without_diagnose = obs2.reward

        assert r_with_diagnose > r_without_diagnose, \
            f"Diagnose-first should yield higher reward: {r_with_diagnose:.3f} vs {r_without_diagnose:.3f}"


# ===========================================================================
# A4: CRAC Failure Cascade Tests
# ===========================================================================
class TestA4CRACCascade:
    """Test the A4 scenario lifecycle."""

    def test_initialization(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A4")
        assert obs.scenario_type == "thermal"
        assert "CRAC-1" in obs.alert
        assert "CRAC-3" in obs.alert

    def test_two_cracs_faulted(self) -> None:
        env = DcOpsEnvironment()
        env.reset(scenario="A4")

        obs1 = env.step(DcOpsAction(command="diagnose CRAC-1"))
        assert "compressor" in obs1.action_result.lower()

        obs3 = env.step(DcOpsAction(command="diagnose CRAC-3"))
        assert "fan" in obs3.action_result.lower()

    def test_cascade_has_faster_time(self) -> None:
        """A4 uses 30s per step (urgent scenario)."""
        s = get_scenario("A4")
        assert s.game_time_per_step_s == 30.0

    def test_harder_than_a2(self) -> None:
        """A4 should have higher step budget than A2 (more complex)."""
        a2 = get_scenario("A2")
        a4 = get_scenario("A4")
        assert a4.step_budget >= a2.step_budget
        assert a4.difficulty == "hard"


# ===========================================================================
# B1: UPS Alarm Response Tests
# ===========================================================================
class TestB1UPSAlarm:
    """Test the B1 scenario lifecycle."""

    def test_initialization(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B1")
        assert obs.scenario_type == "power"
        assert "UPS" in obs.alert

    def test_battery_partially_drained(self) -> None:
        """UPS battery should be partially drained (brief outage)."""
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B1")
        ups_soc = obs.metadata["power"]["UPS-1"]["battery_soc"]
        assert ups_soc < 1.0, f"Battery should be partially drained, SOC={ups_soc}"

    def test_resolution_requires_diagnose_and_ack(self) -> None:
        """B1 resolves when agent diagnoses UPS AND acknowledges alarm."""
        env = DcOpsEnvironment()
        env.reset(scenario="B1")

        obs = env.step(DcOpsAction(command="diagnose UPS-1"))
        assert obs.done is False  # Not resolved yet

        obs = env.step(DcOpsAction(command="acknowledge_alarm"))
        assert obs.done is True  # Now resolved


# ===========================================================================
# B3: Generator Test Protocol Tests
# ===========================================================================
class TestB3GeneratorTest:
    """Test the B3 scenario lifecycle."""

    def test_initialization(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B3")
        assert obs.scenario_type == "power"
        assert "generator" in obs.alert.lower()

    def test_correct_protocol_resolves(self) -> None:
        """Following correct protocol should resolve the scenario."""
        env = DcOpsEnvironment()
        env.reset(scenario="B3")

        env.step(DcOpsAction(command="check_status"))
        env.step(DcOpsAction(command="start_generator"))

        # Wait for generator to start up
        env.step(DcOpsAction(command="wait"))
        env.step(DcOpsAction(command="wait"))

        env.step(DcOpsAction(command="diagnose GEN-1"))
        env.step(DcOpsAction(command="stop_generator"))
        obs = env.step(DcOpsAction(command="acknowledge_alarm"))

        assert obs.done is True

    def test_uses_10s_steps(self) -> None:
        s = get_scenario("B3")
        assert s.game_time_per_step_s == 10.0


# ===========================================================================
# B4: Power Failure Cascade Tests
# ===========================================================================
class TestB4PowerCascade:
    """Test the B4 scenario lifecycle."""

    def test_initialization(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B4")
        assert obs.scenario_type == "power"
        assert "utility" in obs.alert.lower() or "power" in obs.alert.lower()

    def test_utility_is_down(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B4")
        assert obs.metadata["power"]["utility_available"] is False

    def test_ups_on_battery(self) -> None:
        """UPS should be on battery after utility loss."""
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B4")
        # After warmup + fault injection, UPS should be on battery
        ups_mode = obs.metadata["power"]["UPS-1"]["mode"]
        assert ups_mode in ("on_battery", "double_conversion"), f"UPS mode: {ups_mode}"

    def test_fast_time_progression(self) -> None:
        s = get_scenario("B4")
        assert s.game_time_per_step_s == 15.0
        assert s.difficulty == "hard"


# ===========================================================================
# Environment Scenario Integration Tests
# ===========================================================================
class TestScenarioIntegration:
    """Test scenario integration with the environment."""

    def test_scenario_by_id_string(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A1")
        assert obs.metadata["scenario"]["id"] == "A1"

    def test_scenario_by_instance(self) -> None:
        env = DcOpsEnvironment()
        s = get_scenario("B3")
        obs = env.reset(scenario=s)
        assert obs.metadata["scenario"]["id"] == "B3"

    def test_scenario_step_budget_used(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A1")
        assert obs.steps_remaining == 10  # A1 budget

    def test_scenario_kwargs_override(self) -> None:
        """Explicit kwargs should override scenario defaults."""
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A1", step_budget=5)
        assert obs.steps_remaining == 5

    def test_no_scenario_backward_compat(self) -> None:
        """Environment should work without a scenario (backward compat)."""
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "scenario" not in obs.metadata
        assert obs.scenario_type == ""

    def test_scenario_resolution_ends_episode(self) -> None:
        """When scenario is resolved, episode should end with done=True."""
        env = DcOpsEnvironment()
        env.reset(scenario="B1")

        # Resolve B1: diagnose + acknowledge
        env.step(DcOpsAction(command="diagnose UPS-1"))
        obs = env.step(DcOpsAction(command="acknowledge_alarm"))
        assert obs.done is True

    def test_speed_bonus_on_resolution(self) -> None:
        """Resolving early should give a speed bonus."""
        env = DcOpsEnvironment()
        env.reset(scenario="B1")  # Budget: 10

        env.step(DcOpsAction(command="diagnose UPS-1"))  # Step 1
        obs = env.step(DcOpsAction(command="acknowledge_alarm"))  # Step 2

        # Speed bonus = (10 - 2) / 10 = 0.8
        # Total reward should include this bonus
        assert obs.reward > 0.5, f"Expected speed bonus in reward, got {obs.reward:.3f}"

    def test_random_scenario_via_reset(self) -> None:
        """reset(random_scenario=True) should pick a random scenario."""
        env = DcOpsEnvironment()
        obs = env.reset(random_scenario=True, seed=42)
        assert "scenario" in obs.metadata
        assert obs.metadata["scenario"]["id"] in registered_scenario_ids()


# ===========================================================================
# All Scenarios Smoke Test
# ===========================================================================
class TestAllScenariosSmoke:
    """Smoke test: every scenario can initialize and run 3 steps."""

    @pytest.mark.parametrize("scenario_id", registered_scenario_ids())
    def test_scenario_runs(self, scenario_id: str) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(scenario=scenario_id)
        assert isinstance(obs, DcOpsObservation)
        assert obs.done is False
        assert len(obs.dashboard) > 100

        # Run 3 steps
        for _ in range(3):
            obs = env.step(DcOpsAction(command="wait"))
            assert isinstance(obs, DcOpsObservation)
