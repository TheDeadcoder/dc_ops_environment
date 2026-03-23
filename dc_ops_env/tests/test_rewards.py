# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the multi-objective reward function.

Validates:
  - softplus numerical stability
  - Individual reward component behavior and bounds
  - Weight profiles sum to 1.0
  - Delta-based progress tracking
  - Action quality heuristics
  - End-to-end reward computation
  - Integration with the full environment
"""

from __future__ import annotations

import math

import pytest

from dc_ops_env.rewards.reward_function import (
    RewardComponents,
    RewardFunction,
    RewardWeights,
    WEIGHT_PROFILES,
    softplus,
)
from dc_ops_env.config import (
    ASHRAE_CLASSES,
    make_default_datacenter_config,
)
from dc_ops_env.simulation.thermal import ThermalSimulation
from dc_ops_env.simulation.power import PowerSimulation
from dc_ops_env.simulation.types import UPSMode
from dc_ops_env.actions.parser import CommandResult
from dc_ops_env.scenarios.base import ScenarioResult


# ===========================================================================
# Helpers
# ===========================================================================
def _make_thermal_sim(setpoint_c: float = 20.0) -> ThermalSimulation:
    """Create a warmed-up thermal simulation with a given CRAC setpoint."""
    config = make_default_datacenter_config()
    for zone_cfg in config.zones:
        for crac_cfg in zone_cfg.crac_units:
            crac_cfg.initial_setpoint_c = setpoint_c
    sim = ThermalSimulation(config)
    # Warmup to reach steady state
    for _ in range(120):
        sim.step(1.0)
    return sim


def _make_power_sim(
    utility_available: bool = True,
) -> PowerSimulation:
    """Create a power simulation with default config."""
    config = make_default_datacenter_config()
    it_load = 160.0  # Default total IT load
    power_sim = PowerSimulation(config.power, it_load_kw=it_load)
    if not utility_available:
        power_sim.set_utility_available(False)
        # Step a bit so UPS transitions to battery
        for _ in range(5):
            power_sim.step(1.0, it_load)
    return power_sim


def _ok_cmd(name: str = "check_status") -> CommandResult:
    return CommandResult(success=True, message="OK", command_name=name)


def _fail_cmd() -> CommandResult:
    return CommandResult(success=False, message="Unknown command", command_name="")


# ===========================================================================
# softplus Unit Tests
# ===========================================================================
class TestSoftplus:
    """Validate the numerically stable softplus implementation."""

    def test_softplus_positive(self) -> None:
        assert softplus(1.0) == pytest.approx(math.log1p(math.exp(1.0)), abs=1e-10)

    def test_softplus_zero(self) -> None:
        assert softplus(0.0) == pytest.approx(math.log(2.0), abs=1e-10)

    def test_softplus_negative(self) -> None:
        assert softplus(-5.0) == pytest.approx(math.log1p(math.exp(-5.0)), abs=1e-10)

    def test_softplus_large_positive_clamp(self) -> None:
        """x > 20 should return x directly (avoid exp overflow)."""
        assert softplus(25.0) == 25.0
        assert softplus(100.0) == 100.0

    def test_softplus_large_negative_clamp(self) -> None:
        """x < -20 should return 0.0 (avoid underflow noise)."""
        assert softplus(-25.0) == 0.0
        assert softplus(-100.0) == 0.0

    def test_softplus_monotonic(self) -> None:
        """softplus should be monotonically increasing."""
        values = [-10, -5, -1, 0, 1, 5, 10, 15]
        results = [softplus(x) for x in values]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_softplus_always_nonnegative(self) -> None:
        for x in [-20, -10, -1, 0, 1, 10]:
            assert softplus(x) >= 0.0


# ===========================================================================
# Weight Profile Tests
# ===========================================================================
class TestWeightProfiles:
    """Validate weight profiles sum to 1.0 and are well-formed."""

    @pytest.mark.parametrize("profile_name", ["thermal", "power", "default"])
    def test_weights_sum_to_one(self, profile_name: str) -> None:
        w = WEIGHT_PROFILES[profile_name]
        total = (
            w.thermal_safety + w.power_safety + w.efficiency
            + w.scenario_progress + w.procedure + w.action_quality
        )
        assert total == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("profile_name", ["thermal", "power", "default"])
    def test_weights_nonnegative(self, profile_name: str) -> None:
        w = WEIGHT_PROFILES[profile_name]
        assert w.thermal_safety >= 0
        assert w.power_safety >= 0
        assert w.efficiency >= 0
        assert w.scenario_progress >= 0
        assert w.procedure >= 0
        assert w.action_quality >= 0

    def test_thermal_profile_emphasizes_thermal(self) -> None:
        w = WEIGHT_PROFILES["thermal"]
        assert w.thermal_safety >= w.power_safety
        assert w.thermal_safety >= w.efficiency

    def test_power_profile_emphasizes_power(self) -> None:
        w = WEIGHT_PROFILES["power"]
        assert w.power_safety >= w.thermal_safety
        assert w.power_safety >= w.efficiency

    def test_unknown_profile_falls_back_to_default(self) -> None:
        rf = RewardFunction(scenario_type="unknown_type")
        # Should use default weights without error
        thermal_sim = _make_thermal_sim()
        components = rf.compute(
            thermal_sim, None, _ok_cmd(), "check_status", ["check_status"], None,
        )
        assert isinstance(components, RewardComponents)


# ===========================================================================
# Thermal Safety Component Tests
# ===========================================================================
class TestThermalSafety:
    """Validate the dual-softplus thermal safety barrier."""

    def test_safe_temps_near_zero(self) -> None:
        """With comfortable temps (20°C setpoint), penalty should be near 0."""
        thermal_sim = _make_thermal_sim(setpoint_c=20.0)
        r = RewardFunction._thermal_safety(thermal_sim)
        # Should be in [-1, 0] and close to 0 for safe temps
        assert -0.3 <= r <= 0.0

    def test_returns_negative_or_zero(self) -> None:
        """Thermal safety should never return positive values."""
        for sp in [15.0, 20.0, 24.0]:
            thermal_sim = _make_thermal_sim(setpoint_c=sp)
            r = RewardFunction._thermal_safety(thermal_sim)
            assert r <= 0.0

    def test_higher_setpoint_more_penalty(self) -> None:
        """Higher setpoints → hotter temps → more penalty."""
        r_low = RewardFunction._thermal_safety(_make_thermal_sim(15.0))
        r_high = RewardFunction._thermal_safety(_make_thermal_sim(24.0))
        # Higher setpoint should yield equal or more negative reward
        assert r_high <= r_low

    def test_bounded_to_neg_one(self) -> None:
        """Even extreme temps should be bounded to [-1, 0] via tanh."""
        thermal_sim = _make_thermal_sim(setpoint_c=15.0)
        # Force extreme rack inlet temps
        for zone in thermal_sim.state.zones:
            for rack in zone.racks:
                rack.inlet_temp_c = 50.0
        r = RewardFunction._thermal_safety(thermal_sim)
        assert r >= -1.0
        assert r <= 0.0


# ===========================================================================
# Power Safety Component Tests
# ===========================================================================
class TestPowerSafety:
    """Validate UPS battery and fault penalty."""

    def test_no_power_sim_returns_zero(self) -> None:
        assert RewardFunction._power_safety(None) == 0.0

    def test_utility_available_near_zero(self) -> None:
        """Normal operation (utility available) should have near-zero penalty."""
        power_sim = _make_power_sim(utility_available=True)
        r = RewardFunction._power_safety(power_sim)
        # On utility with full battery → no penalty
        assert -0.15 <= r <= 0.0

    def test_on_battery_gives_penalty(self) -> None:
        """UPS on battery should yield a meaningful penalty."""
        power_sim = _make_power_sim(utility_available=False)
        r = RewardFunction._power_safety(power_sim)
        assert r < 0.0  # Should be negative when on battery

    def test_low_soc_increases_penalty(self) -> None:
        """Lower SOC while on battery should increase penalty."""
        power_sim = _make_power_sim(utility_available=False)
        # Force low SOC
        for ups in power_sim.state.ups_units:
            ups.battery_soc = 0.3
        r_low = RewardFunction._power_safety(power_sim)

        power_sim2 = _make_power_sim(utility_available=False)
        for ups in power_sim2.state.ups_units:
            ups.battery_soc = 0.8
        r_high = RewardFunction._power_safety(power_sim2)

        assert r_low < r_high  # Low SOC → more negative

    def test_fault_mode_heavy_penalty(self) -> None:
        """UPS in FAULT mode should yield heavy penalty."""
        power_sim = _make_power_sim(utility_available=True)
        for ups in power_sim.state.ups_units:
            ups.mode = UPSMode.FAULT
        r = RewardFunction._power_safety(power_sim)
        assert r < -0.7  # Should be very negative

    def test_bounded(self) -> None:
        """Power safety should be in [-1, 0]."""
        power_sim = _make_power_sim(utility_available=False)
        for ups in power_sim.state.ups_units:
            ups.mode = UPSMode.FAULT
            ups.battery_soc = 0.0
        r = RewardFunction._power_safety(power_sim)
        assert -1.0 <= r <= 0.0


# ===========================================================================
# Efficiency Component Tests
# ===========================================================================
class TestEfficiency:
    """Validate PUE-based efficiency penalty."""

    def test_low_pue_near_zero_penalty(self) -> None:
        """PUE close to 1.0 should yield near-zero penalty."""
        thermal_sim = _make_thermal_sim(20.0)
        r = RewardFunction._efficiency(thermal_sim)
        pue = thermal_sim.state.pue
        # PUE is typically 1.4-1.8 in our sim, so some penalty is expected
        assert -0.5 <= r <= 0.0

    def test_returns_negative_or_zero(self) -> None:
        thermal_sim = _make_thermal_sim(20.0)
        r = RewardFunction._efficiency(thermal_sim)
        assert r <= 0.0

    def test_bounded(self) -> None:
        """Even extreme PUE should be bounded."""
        thermal_sim = _make_thermal_sim(15.0)
        # Force extreme PUE by manipulating state
        thermal_sim.state._pue = 5.0
        r = RewardFunction._efficiency(thermal_sim)
        assert -1.0 <= r <= 0.0


# ===========================================================================
# Scenario Progress Component Tests
# ===========================================================================
class TestScenarioProgress:
    """Validate delta-based progress reward."""

    def test_no_scenario_returns_zero(self) -> None:
        rf = RewardFunction()
        assert rf._scenario_progress(None) == 0.0

    def test_first_step_progress(self) -> None:
        """First step with progress > 0 should yield positive delta."""
        rf = RewardFunction()
        result = ScenarioResult(progress=0.5)
        r = rf._scenario_progress(result)
        assert r == pytest.approx(0.5)

    def test_delta_tracking(self) -> None:
        """Only the delta should be rewarded, not cumulative progress."""
        rf = RewardFunction()

        r1 = rf._scenario_progress(ScenarioResult(progress=0.3))
        assert r1 == pytest.approx(0.3)

        r2 = rf._scenario_progress(ScenarioResult(progress=0.3))
        assert r2 == pytest.approx(0.0)  # No change → no reward

        r3 = rf._scenario_progress(ScenarioResult(progress=0.7))
        assert r3 == pytest.approx(0.4)  # 0.7 - 0.3

    def test_negative_delta_penalized(self) -> None:
        """Progress regression should yield negative reward."""
        rf = RewardFunction()
        rf._scenario_progress(ScenarioResult(progress=0.8))
        r = rf._scenario_progress(ScenarioResult(progress=0.5))
        assert r == pytest.approx(-0.3)

    def test_bounded(self) -> None:
        """Progress delta should be clamped to [-1, 1]."""
        rf = RewardFunction()
        r = rf._scenario_progress(ScenarioResult(progress=1.0))
        assert -1.0 <= r <= 1.0

    def test_reset_clears_state(self) -> None:
        """reset() should clear the previous progress."""
        rf = RewardFunction()
        rf._scenario_progress(ScenarioResult(progress=0.5))

        rf.reset()
        r = rf._scenario_progress(ScenarioResult(progress=0.3))
        assert r == pytest.approx(0.3)  # From 0, not from 0.5


# ===========================================================================
# Procedure Component Tests
# ===========================================================================
class TestProcedure:
    """Validate procedural correctness pass-through."""

    def test_no_scenario_returns_zero(self) -> None:
        assert RewardFunction._procedure(None) == 0.0

    def test_positive_procedure_reward(self) -> None:
        r = RewardFunction._procedure(ScenarioResult(procedure_reward=0.3))
        assert r == pytest.approx(0.3)

    def test_negative_procedure_reward(self) -> None:
        r = RewardFunction._procedure(ScenarioResult(procedure_reward=-0.2))
        assert r == pytest.approx(-0.2)

    def test_clamped_to_bounds(self) -> None:
        r = RewardFunction._procedure(ScenarioResult(procedure_reward=5.0))
        assert r == 1.0
        r = RewardFunction._procedure(ScenarioResult(procedure_reward=-5.0))
        assert r == -1.0


# ===========================================================================
# Action Quality Component Tests
# ===========================================================================
class TestActionQuality:
    """Validate contextual action quality assessment."""

    def test_invalid_command_penalty(self) -> None:
        thermal_sim = _make_thermal_sim()
        r = RewardFunction._action_quality(
            _fail_cmd(), "nonsense", ["nonsense"], thermal_sim, None,
        )
        assert r == pytest.approx(-0.5)

    def test_diagnose_rewarded(self) -> None:
        thermal_sim = _make_thermal_sim()
        r = RewardFunction._action_quality(
            _ok_cmd("diagnose"), "diagnose CRAC-1", ["diagnose CRAC-1"],
            thermal_sim, None,
        )
        assert r == pytest.approx(0.3)

    def test_check_status_rewarded(self) -> None:
        thermal_sim = _make_thermal_sim()
        r = RewardFunction._action_quality(
            _ok_cmd("check_status"), "check_status", ["check_status"],
            thermal_sim, None,
        )
        assert r == pytest.approx(0.3)

    def test_intervention_rewarded(self) -> None:
        thermal_sim = _make_thermal_sim()
        r = RewardFunction._action_quality(
            _ok_cmd("adjust_setpoint"), "adjust_setpoint CRAC-1 22",
            ["adjust_setpoint CRAC-1 22"], thermal_sim, None,
        )
        assert r == pytest.approx(0.2)

    def test_acknowledge_rewarded(self) -> None:
        thermal_sim = _make_thermal_sim()
        r = RewardFunction._action_quality(
            _ok_cmd("acknowledge_alarm"), "acknowledge_alarm",
            ["acknowledge_alarm"], thermal_sim, None,
        )
        assert r == pytest.approx(0.1)

    def test_repeated_command_penalized(self) -> None:
        thermal_sim = _make_thermal_sim()
        history = ["check_status", "check_status"]
        r = RewardFunction._action_quality(
            _ok_cmd("check_status"), "check_status", history,
            thermal_sim, None,
        )
        assert r == pytest.approx(-0.2)

    def test_wait_no_concern_neutral(self) -> None:
        """Waiting when nothing is wrong should be neutral (0.0)."""
        thermal_sim = _make_thermal_sim(20.0)  # Safe temps
        r = RewardFunction._action_quality(
            _ok_cmd("wait"), "wait", ["wait"], thermal_sim, None,
        )
        assert r == pytest.approx(0.0)

    def test_wait_during_concern_penalized(self) -> None:
        """Waiting during a thermal concern should be penalized."""
        thermal_sim = _make_thermal_sim(20.0)
        # Force rack inlet temps above recommended max to create concern
        for zone in thermal_sim.state.zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if ashrae:
                for rack in zone.racks:
                    rack.inlet_temp_c = ashrae.recommended_max_c + 2.0
        r = RewardFunction._action_quality(
            _ok_cmd("wait"), "wait", ["wait"], thermal_sim, None,
        )
        assert r == pytest.approx(-0.2)

    def test_wait_during_battery_concern(self) -> None:
        """Waiting while UPS on battery should be penalized."""
        thermal_sim = _make_thermal_sim(20.0)
        power_sim = _make_power_sim(utility_available=False)
        r = RewardFunction._action_quality(
            _ok_cmd("wait"), "wait", ["wait"], thermal_sim, power_sim,
        )
        assert r == pytest.approx(-0.2)


# ===========================================================================
# Full Compute Tests
# ===========================================================================
class TestRewardCompute:
    """Validate full reward computation."""

    def test_compute_returns_components(self) -> None:
        rf = RewardFunction(scenario_type="thermal")
        thermal_sim = _make_thermal_sim()
        components = rf.compute(
            thermal_sim, None, _ok_cmd(), "check_status",
            ["check_status"], None,
        )
        assert isinstance(components, RewardComponents)
        assert hasattr(components, "total")
        assert hasattr(components, "thermal_safety")

    def test_total_bounded(self) -> None:
        """Total reward should be in [-1, 1]."""
        rf = RewardFunction(scenario_type="thermal")
        thermal_sim = _make_thermal_sim()
        components = rf.compute(
            thermal_sim, None, _ok_cmd(), "check_status",
            ["check_status"], None,
        )
        assert -1.0 <= components.total <= 1.0

    def test_total_bounded_worst_case(self) -> None:
        """Even with all-negative components, total should be >= -1."""
        rf = RewardFunction(scenario_type="thermal")
        thermal_sim = _make_thermal_sim()
        # Force extreme conditions
        for zone in thermal_sim.state.zones:
            for rack in zone.racks:
                rack.inlet_temp_c = 50.0
        components = rf.compute(
            thermal_sim, None, _fail_cmd(), "nonsense",
            ["nonsense"],
            ScenarioResult(procedure_reward=-1.0, progress=0.0),
        )
        assert components.total >= -1.0

    def test_valid_action_better_than_invalid(self) -> None:
        """Same conditions, valid action should score higher than invalid."""
        rf = RewardFunction(scenario_type="default")
        thermal_sim = _make_thermal_sim()

        c_valid = rf.compute(
            thermal_sim, None, _ok_cmd(), "check_status",
            ["check_status"], None,
        )
        rf.reset()
        c_invalid = rf.compute(
            thermal_sim, None, _fail_cmd(), "nonsense",
            ["nonsense"], None,
        )
        assert c_valid.total > c_invalid.total

    def test_progress_delta_affects_total(self) -> None:
        """Making progress should increase total reward."""
        rf = RewardFunction(scenario_type="thermal")
        thermal_sim = _make_thermal_sim()

        c1 = rf.compute(
            thermal_sim, None, _ok_cmd("diagnose"), "diagnose CRAC-1",
            ["diagnose CRAC-1"],
            ScenarioResult(progress=0.5),
        )

        c2 = rf.compute(
            thermal_sim, None, _ok_cmd("diagnose"), "diagnose CRAC-2",
            ["diagnose CRAC-1", "diagnose CRAC-2"],
            ScenarioResult(progress=0.5),  # No change
        )

        # Step with progress delta should score higher (all else similar)
        assert c1.scenario_progress > c2.scenario_progress

    def test_with_power_sim(self) -> None:
        """Compute should work with both thermal and power sims."""
        rf = RewardFunction(scenario_type="power")
        thermal_sim = _make_thermal_sim()
        power_sim = _make_power_sim(utility_available=True)

        components = rf.compute(
            thermal_sim, power_sim, _ok_cmd(), "check_status",
            ["check_status"], None,
        )
        assert -1.0 <= components.total <= 1.0

    def test_custom_weights(self) -> None:
        """Custom weights should override profile."""
        custom = RewardWeights(
            thermal_safety=0.0, power_safety=0.0, efficiency=0.0,
            scenario_progress=0.0, procedure=0.0, action_quality=1.0,
        )
        rf = RewardFunction(weights=custom)
        thermal_sim = _make_thermal_sim()

        c = rf.compute(
            thermal_sim, None, _ok_cmd("diagnose"), "diagnose CRAC-1",
            ["diagnose CRAC-1"], None,
        )
        # With only action_quality weighted, total should equal action_quality
        assert c.total == pytest.approx(c.action_quality, abs=0.01)


# ===========================================================================
# Integration with Full Environment
# ===========================================================================
class TestRewardIntegration:
    """Validate reward function works correctly inside the environment."""

    def test_scenario_reward_uses_reward_function(self) -> None:
        """Environment should use RewardFunction, not old _compute_reward."""
        from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
        from dc_ops_env.models import DcOpsAction

        env = DcOpsEnvironment()
        env.reset(scenario="A1")  # Cooling setpoint optimization
        obs = env.step(DcOpsAction(command="check_status"))
        # Reward should be a float from the new system
        assert isinstance(obs.reward, float)
        assert obs.reward != 0.0  # Should have some signal

    def test_escalation_has_penalty(self) -> None:
        """Escalation should add a -0.3 penalty on top of base reward."""
        from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
        from dc_ops_env.models import DcOpsAction

        env = DcOpsEnvironment()
        env.reset(scenario="A2")
        obs = env.step(DcOpsAction(command="escalate"))
        assert obs.done is True
        assert obs.reward < 0  # Should be negative due to penalty

    def test_scenario_resolution_has_speed_bonus(self) -> None:
        """Resolving a scenario early should yield a speed bonus."""
        from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
        from dc_ops_env.models import DcOpsAction

        env = DcOpsEnvironment()
        env.reset(scenario="B1")  # UPS Alarm Response

        # Solve B1: diagnose UPS then acknowledge
        env.step(DcOpsAction(command="diagnose UPS-1"))
        obs = env.step(DcOpsAction(command="acknowledge_alarm"))

        # Should be resolved with speed bonus
        assert obs.done is True
        # Speed bonus = (budget - steps) / budget = (10 - 2) / 10 = 0.8
        # Total reward includes base + speed bonus, should be positive
        assert obs.reward > 0.5

    def test_reward_function_reset_on_env_reset(self) -> None:
        """RewardFunction state should reset between episodes."""
        from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
        from dc_ops_env.models import DcOpsAction

        env = DcOpsEnvironment()

        # Episode 1
        env.reset(scenario="A1")
        env.step(DcOpsAction(command="check_status"))

        # Episode 2 — progress delta should start fresh
        env.reset(scenario="A1")
        obs = env.step(DcOpsAction(command="check_status"))
        assert isinstance(obs.reward, float)
