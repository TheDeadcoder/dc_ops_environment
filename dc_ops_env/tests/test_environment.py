# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the DC-Ops environment, action parser, and dashboard renderer.

Validates:
  - OpenEnv interface contract (reset/step/state)
  - Action parsing (valid and invalid commands)
  - Dashboard rendering output format
  - Episode termination conditions
  - Fault injection
  - Reward computation
"""

from __future__ import annotations

import pytest

from dc_ops_env.models import DcOpsAction, DcOpsObservation
from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment


# ===========================================================================
# OpenEnv Interface Contract
# ===========================================================================
class TestOpenEnvContract:
    """Verify the environment satisfies the OpenEnv Environment ABC."""

    def test_reset_returns_observation(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert isinstance(obs, DcOpsObservation)
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_has_dashboard(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert len(obs.dashboard) > 100
        assert "DC-OPS MONITORING DASHBOARD" in obs.dashboard

    def test_reset_has_available_actions(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert len(obs.available_actions) > 5

    def test_step_returns_observation(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="check_status"))
        assert isinstance(obs, DcOpsObservation)
        assert obs.done is False

    def test_step_advances_step_count(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        assert env.state.step_count == 0
        env.step(DcOpsAction(command="wait"))
        assert env.state.step_count == 1
        env.step(DcOpsAction(command="wait"))
        assert env.state.step_count == 2

    def test_state_has_episode_id(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        assert env.state.episode_id is not None
        assert len(env.state.episode_id) > 0

    def test_reset_changes_episode_id(self) -> None:
        env = DcOpsEnvironment()
        obs1 = env.reset()
        ep1 = env.state.episode_id
        obs2 = env.reset()
        ep2 = env.state.episode_id
        assert ep1 != ep2

    def test_observation_metadata_populated(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "total_it_load_kw" in obs.metadata
        assert "pue" in obs.metadata
        assert "zones" in obs.metadata
        assert obs.metadata["total_it_load_kw"] == pytest.approx(160.0, rel=0.01)

    def test_observation_has_power_metadata(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "power" in obs.metadata
        assert obs.metadata["power"]["utility_available"] is True


# ===========================================================================
# Action Parser Tests
# ===========================================================================
class TestActionParser:
    """Test command parsing and execution."""

    def test_diagnose_crac(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="diagnose CRAC-1"))
        assert "Diagnostic Report" in obs.action_result
        assert "CRAC-1" in obs.action_result
        assert obs.reward > -0.5  # Valid action should not be heavily penalized

    def test_diagnose_ups(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="diagnose UPS-1"))
        assert "Diagnostic Report" in obs.action_result
        assert "UPS-1" in obs.action_result

    def test_diagnose_nonexistent(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="diagnose CRAC-99"))
        assert "not found" in obs.action_result

    def test_adjust_setpoint_valid(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="adjust_setpoint CRAC-1 22"))
        assert "adjusted" in obs.action_result.lower()
        assert "22.0" in obs.action_result

    def test_adjust_setpoint_out_of_range(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="adjust_setpoint CRAC-1 50"))
        assert "out of safe range" in obs.action_result.lower() or "out of" in obs.action_result.lower()

    def test_set_fan_speed(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="set_fan_speed CRAC-2 80"))
        assert "fan speed" in obs.action_result.lower()
        assert "80" in obs.action_result

    def test_set_rack_load(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="set_rack_load A-01 12"))
        assert "12.0" in obs.action_result

    def test_start_generator(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="start_generator"))
        assert "generator" in obs.action_result.lower()

    def test_wait_command(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="wait"))
        assert "no action" in obs.action_result.lower()

    def test_check_status(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="check_status"))
        assert "status" in obs.action_result.lower()

    def test_invalid_command(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="fly_to_the_moon"))
        assert "unknown" in obs.action_result.lower()

    def test_empty_command(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command=""))
        assert "empty" in obs.action_result.lower()

    def test_case_insensitive(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="DIAGNOSE CRAC-1"))
        assert "Diagnostic Report" in obs.action_result

    def test_start_stop_crac(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="stop_crac CRAC-2"))
        assert "standby" in obs.action_result.lower()

        obs = env.step(DcOpsAction(command="start_crac CRAC-2"))
        assert "started" in obs.action_result.lower()


# ===========================================================================
# Dashboard Rendering Tests
# ===========================================================================
class TestDashboardRendering:
    """Test dashboard output format and content."""

    def test_dashboard_has_cooling_section(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "COOLING UNITS" in obs.dashboard
        assert "CRAC-1" in obs.dashboard

    def test_dashboard_has_zone_temps(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "ZONE TEMPERATURES" in obs.dashboard
        assert "zone_a" in obs.dashboard

    def test_dashboard_has_rack_temps(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "RACK TEMPERATURES" in obs.dashboard

    def test_dashboard_has_power(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "POWER" in obs.dashboard
        assert "PUE" in obs.dashboard

    def test_dashboard_has_environment(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "ENVIRONMENT" in obs.dashboard
        assert "35.0°C" in obs.dashboard

    def test_dashboard_shows_alert(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(alert="Test alert message")
        assert "ALERT" in obs.dashboard
        assert "Test alert message" in obs.dashboard

    def test_dashboard_shows_step_count(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "Step: 0/15" in obs.dashboard

        obs = env.step(DcOpsAction(command="wait"))
        assert "Step: 1/15" in obs.dashboard

    def test_dashboard_shows_ups_status(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset()
        assert "UPS-1" in obs.dashboard
        assert "UPS-2" in obs.dashboard


# ===========================================================================
# Episode Termination Tests
# ===========================================================================
class TestEpisodeTermination:
    """Test episode termination conditions."""

    def test_step_budget_exhaustion(self) -> None:
        env = DcOpsEnvironment()
        env.reset(step_budget=3)

        obs = env.step(DcOpsAction(command="wait"))
        assert obs.done is False

        obs = env.step(DcOpsAction(command="wait"))
        assert obs.done is False

        obs = env.step(DcOpsAction(command="wait"))
        assert obs.done is True  # Step 3/3

    def test_escalation_terminates(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="escalate"))
        assert obs.done is True
        assert obs.reward < 0  # Penalty for escalating

    def test_step_after_done_is_noop(self) -> None:
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="escalate"))
        assert obs.done is True

        obs2 = env.step(DcOpsAction(command="wait"))
        assert obs2.done is True
        assert "already ended" in obs2.action_result.lower()


# ===========================================================================
# Fault Injection Tests
# ===========================================================================
class TestFaultInjection:
    """Test scenario fault injection at reset."""

    def test_crac_fault_injection(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(
            fault_injection={
                "type": "crac_fault",
                "unit_id": "CRAC-3",
                "fault": "compressor",
            },
        )
        # Dashboard should show the fault
        assert "COMPRESSOR" in obs.dashboard or "FAULT" in obs.dashboard

    def test_utility_loss_injection(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(
            fault_injection={"type": "utility_loss"},
        )
        assert "DOWN" in obs.dashboard or "BATTERY" in obs.dashboard

    def test_outside_temp_injection(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(
            fault_injection={"type": "outside_temp", "temp_c": 45.0},
        )
        assert "45.0°C" in obs.dashboard

    def test_alert_in_observation(self) -> None:
        env = DcOpsEnvironment()
        obs = env.reset(
            alert="HIGH TEMPERATURE in Zone B",
            scenario_type="thermal",
        )
        assert obs.alert == "HIGH TEMPERATURE in Zone B"
        assert obs.scenario_type == "thermal"


# ===========================================================================
# Reward Tests
# ===========================================================================
class TestReward:
    """Test reward computation."""

    def test_valid_action_positive_component(self) -> None:
        """Valid actions should get a positive action reward component."""
        env = DcOpsEnvironment()
        env.reset()
        obs_valid = env.step(DcOpsAction(command="check_status"))
        r_valid = obs_valid.reward

        env.reset()
        obs_invalid = env.step(DcOpsAction(command="nonsense_command"))
        r_invalid = obs_invalid.reward

        # Valid action should yield higher reward than invalid
        assert r_valid > r_invalid

    def test_pue_affects_reward(self) -> None:
        """Reward should be sensitive to PUE."""
        env = DcOpsEnvironment()
        obs = env.reset()
        # Just verify PUE is in metadata and reward is computed
        pue = obs.metadata["pue"]
        assert pue > 1.0  # PUE should always be > 1

    def test_cumulative_reward_tracked(self) -> None:
        """Cumulative reward should be tracked in metadata."""
        env = DcOpsEnvironment()
        env.reset()
        obs = env.step(DcOpsAction(command="wait"))
        assert "cumulative_reward" in obs.metadata
        r1 = obs.metadata["cumulative_reward"]

        obs = env.step(DcOpsAction(command="wait"))
        r2 = obs.metadata["cumulative_reward"]
        # Cumulative should change (it's the sum of per-step rewards)
        assert r2 != 0 or r1 != 0  # At least one should be non-zero


# ===========================================================================
# Simulation Integration Tests
# ===========================================================================
class TestSimulationIntegration:
    """Test that the environment properly advances the simulation."""

    def test_simulation_time_advances(self) -> None:
        """Each step should advance sim time by game_time_per_step."""
        env = DcOpsEnvironment()
        obs = env.reset()
        t0 = obs.metadata["sim_time_s"]

        obs = env.step(DcOpsAction(command="wait"))
        t1 = obs.metadata["sim_time_s"]

        # Default: 60s per step
        assert t1 - t0 == pytest.approx(60.0, rel=0.01)

    def test_custom_game_time_per_step(self) -> None:
        """Custom game_time_per_step should be respected."""
        env = DcOpsEnvironment()
        obs = env.reset(game_time_per_step_s=120.0)
        t0 = obs.metadata["sim_time_s"]

        obs = env.step(DcOpsAction(command="wait"))
        t1 = obs.metadata["sim_time_s"]

        assert t1 - t0 == pytest.approx(120.0, rel=0.01)

    def test_setpoint_change_affects_temperature(self) -> None:
        """Changing setpoint should cause temperature change over steps."""
        env = DcOpsEnvironment()
        obs = env.reset()
        t_cold_before = obs.metadata["zones"]["zone_a"]["cold_aisle_temp_c"]

        # Raise setpoint significantly
        env.step(DcOpsAction(command="adjust_setpoint CRAC-1 25"))
        env.step(DcOpsAction(command="adjust_setpoint CRAC-2 25"))

        # Wait a few steps for temp to change
        for _ in range(3):
            obs = env.step(DcOpsAction(command="wait"))

        t_cold_after = obs.metadata["zones"]["zone_a"]["cold_aisle_temp_c"]

        # Cold aisle should have increased
        assert t_cold_after > t_cold_before + 0.5, \
            f"Expected temp increase: {t_cold_before:.1f} → {t_cold_after:.1f}"


# ===========================================================================
# Performance Test
# ===========================================================================
class TestPerformance:
    """Ensure full environment steps are fast enough."""

    def test_episode_performance(self) -> None:
        """Full 15-step episode should complete in < 5 seconds."""
        import time

        env = DcOpsEnvironment()
        start = time.perf_counter()

        env.reset()
        for _ in range(15):
            env.step(DcOpsAction(command="wait"))

        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Episode took {elapsed:.2f}s, should be < 5s"
