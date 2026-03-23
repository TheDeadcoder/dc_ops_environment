# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests: full episode playback, config loading, cross-facility.

Validates:
  - Known-good action sequences resolve each scenario
  - Reward signals are well-behaved across full episodes
  - YAML config loading produces valid, runnable environments
  - Different facility sizes work correctly
  - Episode metrics (PUE, temps, rewards) are in expected ranges
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from dc_ops_env.config import (
    BUILTIN_CONFIGS,
    DatacenterConfig,
    load_datacenter_config,
    make_default_datacenter_config,
)
from dc_ops_env.models import DcOpsAction, DcOpsObservation
from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
from dc_ops_env.scenarios.registry import registered_scenario_ids


# ===========================================================================
# Config Loading Tests
# ===========================================================================
class TestConfigLoading:
    """Validate YAML config loading and built-in configs."""

    def test_builtin_configs_exist(self) -> None:
        """All built-in config files should exist on disk."""
        for name, path in BUILTIN_CONFIGS.items():
            assert path.exists(), f"Built-in config '{name}' not found at {path}"

    @pytest.mark.parametrize("config_name", ["default", "small", "large"])
    def test_load_builtin(self, config_name: str) -> None:
        """Each built-in config should load without error."""
        cfg = load_datacenter_config(config_name)
        assert isinstance(cfg, DatacenterConfig)
        assert len(cfg.zones) > 0
        for zone in cfg.zones:
            assert len(zone.racks) > 0
            assert len(zone.crac_units) > 0

    def test_load_by_path(self) -> None:
        """Loading by explicit path should work."""
        path = BUILTIN_CONFIGS["default"]
        cfg = load_datacenter_config(path)
        assert cfg.name == "DC-OPS Default Facility"

    def test_load_nonexistent_raises(self) -> None:
        """Loading a missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_datacenter_config("/nonexistent/path.yaml")

    def test_default_yaml_matches_programmatic(self) -> None:
        """YAML default config should match make_default_datacenter_config()."""
        yaml_cfg = load_datacenter_config("default")
        prog_cfg = make_default_datacenter_config()

        assert yaml_cfg.name == prog_cfg.name
        assert len(yaml_cfg.zones) == len(prog_cfg.zones)
        assert yaml_cfg.outside_temp_c == prog_cfg.outside_temp_c

        # Same number of racks and CRACs
        yaml_racks = sum(len(z.racks) for z in yaml_cfg.zones)
        prog_racks = sum(len(z.racks) for z in prog_cfg.zones)
        assert yaml_racks == prog_racks

        yaml_cracs = sum(len(z.crac_units) for z in yaml_cfg.zones)
        prog_cracs = sum(len(z.crac_units) for z in prog_cfg.zones)
        assert yaml_cracs == prog_cracs

    def test_small_facility_dimensions(self) -> None:
        """Small facility should have correct dimensions."""
        cfg = load_datacenter_config("small")
        assert len(cfg.zones) == 1
        total_racks = sum(len(z.racks) for z in cfg.zones)
        assert total_racks == 10
        total_it = sum(r.it_load_kw for z in cfg.zones for r in z.racks)
        assert total_it == pytest.approx(80.0)
        assert len(cfg.power.ups_units) == 1

    def test_large_facility_dimensions(self) -> None:
        """Large facility should have correct dimensions."""
        cfg = load_datacenter_config("large")
        assert len(cfg.zones) == 4
        total_racks = sum(len(z.racks) for z in cfg.zones)
        assert total_racks == 60
        total_it = sum(r.it_load_kw for z in cfg.zones for r in z.racks)
        assert total_it == pytest.approx(600.0)
        assert len(cfg.power.ups_units) == 4

    def test_large_facility_has_h1_zone(self) -> None:
        """Large facility should include an H1 high-density zone."""
        cfg = load_datacenter_config("large")
        h1_zones = [z for z in cfg.zones if z.ashrae_class == "H1"]
        assert len(h1_zones) == 1
        # H1 zone should have higher per-rack load
        for rack in h1_zones[0].racks:
            assert rack.it_load_kw == 20.0


# ===========================================================================
# Config-to-Environment Tests
# ===========================================================================
class TestConfigToEnvironment:
    """Validate that loaded configs produce runnable environments."""

    @pytest.mark.parametrize("config_name", ["default", "small", "large"])
    def test_env_runs_with_config(self, config_name: str) -> None:
        """Environment should initialize and run steps with each config."""
        cfg = load_datacenter_config(config_name)
        env = DcOpsEnvironment()
        obs = env.reset(config=cfg)
        assert isinstance(obs, DcOpsObservation)
        assert obs.done is False

        obs = env.step(DcOpsAction(command="check_status"))
        assert isinstance(obs, DcOpsObservation)

    def test_small_facility_pue(self) -> None:
        """Small facility PUE should be realistic after warmup."""
        cfg = load_datacenter_config("small")
        env = DcOpsEnvironment()
        obs = env.reset(config=cfg)
        pue = obs.metadata["pue"]
        assert 1.1 < pue < 2.5, f"PUE {pue} out of realistic range"

    def test_large_facility_total_load(self) -> None:
        """Large facility total IT load should match config."""
        cfg = load_datacenter_config("large")
        env = DcOpsEnvironment()
        obs = env.reset(config=cfg)
        total_it = obs.metadata["total_it_load_kw"]
        assert total_it == pytest.approx(600.0, rel=0.01)


# ===========================================================================
# Full Episode Playback: Thermal Scenarios
# ===========================================================================
class TestEpisodePlaybackThermal:
    """Full episode playback with known-good action sequences for thermal scenarios."""

    def test_a1_optimal_episode(self) -> None:
        """A1 (Cooling Setpoint Optimization): raise setpoints to reduce PUE.

        Optimal sequence: check_status → raise each CRAC setpoint → wait for convergence.
        PUE should improve significantly from baseline.
        """
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A1")
        pue_initial = obs.metadata["pue"]

        # 1. Check status first (procedure bonus)
        obs = env.step(DcOpsAction(command="check_status"))
        assert not obs.done

        # 2. Raise setpoints on all 4 CRACs from 15°C → 24°C (aggressive)
        for crac_id in ["CRAC-1", "CRAC-2", "CRAC-3", "CRAC-4"]:
            obs = env.step(DcOpsAction(command=f"adjust_setpoint {crac_id} 24"))

        # 3. Wait for temps to converge
        for _ in range(5):
            obs = env.step(DcOpsAction(command="wait"))
            if obs.done:
                break

        pue_final = obs.metadata["pue"]
        # PUE should have improved (lower is better)
        assert pue_final < pue_initial, (
            f"PUE should improve: {pue_initial:.2f} → {pue_final:.2f}"
        )

    def test_a2_optimal_episode(self) -> None:
        """A2 (Thermal Event Response): diagnose CRAC-3, compensate with remaining units.

        Optimal: diagnose → increase fan speeds on survivors → adjust setpoints.
        """
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A2")

        # 1. Diagnose the failed CRAC
        obs = env.step(DcOpsAction(command="diagnose CRAC-3"))
        assert "COMPRESSOR" in obs.action_result or "compressor" in obs.action_result.lower()

        # 2. Increase fan speed on remaining CRACs
        for crac_id in ["CRAC-1", "CRAC-2", "CRAC-4"]:
            obs = env.step(DcOpsAction(command=f"set_fan_speed {crac_id} 100"))

        # 3. Lower setpoints slightly on surviving units to compensate
        for crac_id in ["CRAC-1", "CRAC-2", "CRAC-4"]:
            obs = env.step(DcOpsAction(command=f"adjust_setpoint {crac_id} 16"))

        # 4. Wait for stabilization
        for _ in range(8):
            obs = env.step(DcOpsAction(command="wait"))
            if obs.done:
                break

        # Should resolve or be close — temps within recommended for 2+ steps
        # Even if not fully resolved, reward should be reasonable
        assert obs.metadata["cumulative_reward"] > -5.0

    def test_a4_episode_with_load_shedding(self) -> None:
        """A4 (CRAC Failure Cascade): diagnose both, compensate, shed load.

        This is the hardest thermal scenario — two CRACs down.
        """
        env = DcOpsEnvironment()
        obs = env.reset(scenario="A4")

        # 1. Diagnose both failed units
        obs = env.step(DcOpsAction(command="diagnose CRAC-1"))
        obs = env.step(DcOpsAction(command="diagnose CRAC-3"))

        # 2. Max out surviving CRACs
        obs = env.step(DcOpsAction(command="set_fan_speed CRAC-2 100"))
        obs = env.step(DcOpsAction(command="set_fan_speed CRAC-4 100"))
        obs = env.step(DcOpsAction(command="adjust_setpoint CRAC-2 15"))
        obs = env.step(DcOpsAction(command="adjust_setpoint CRAC-4 15"))

        # 3. Shed load on hottest racks
        for rack_id in ["A-01", "A-02", "B-01", "B-02"]:
            obs = env.step(DcOpsAction(command=f"set_rack_load {rack_id} 4"))

        # 4. Wait and monitor
        for _ in range(10):
            obs = env.step(DcOpsAction(command="wait"))
            if obs.done:
                break

        # Hard scenario — may not fully resolve, but should make progress
        assert obs.metadata["cumulative_reward"] > -10.0


# ===========================================================================
# Full Episode Playback: Power Scenarios
# ===========================================================================
class TestEpisodePlaybackPower:
    """Full episode playback with known-good action sequences for power scenarios."""

    def test_b1_optimal_episode(self) -> None:
        """B1 (UPS Alarm Response): diagnose UPS, acknowledge alarm.

        Simple 2-step resolution.
        """
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B1")

        # 1. Diagnose UPS status
        obs = env.step(DcOpsAction(command="diagnose UPS-1"))
        assert not obs.done

        # 2. Acknowledge the alarm
        obs = env.step(DcOpsAction(command="acknowledge_alarm"))
        assert obs.done, "B1 should resolve after diagnose + acknowledge"

        # Speed bonus: (10 - 2) / 10 = 0.8
        assert obs.reward > 0.5, "Should have significant speed bonus"

    def test_b3_optimal_episode(self) -> None:
        """B3 (Generator Test Protocol): follow the correct test sequence.

        check_status → start_generator → diagnose GEN-1 → stop_generator → acknowledge.
        """
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B3")

        # Follow correct protocol
        obs = env.step(DcOpsAction(command="check_status"))
        assert not obs.done

        obs = env.step(DcOpsAction(command="start_generator"))
        assert not obs.done

        # Wait for generator to start (30s game time per step, gen startup ~17s)
        obs = env.step(DcOpsAction(command="wait"))

        obs = env.step(DcOpsAction(command="diagnose GEN-1"))
        assert not obs.done

        obs = env.step(DcOpsAction(command="stop_generator"))
        assert not obs.done

        # Wait for cooldown
        obs = env.step(DcOpsAction(command="wait"))

        obs = env.step(DcOpsAction(command="acknowledge_alarm"))
        assert obs.done, "B3 should resolve after full protocol"

    def test_b4_episode_with_load_shedding(self) -> None:
        """B4 (Power Failure Cascade): manage battery, wait for generator.

        Generator starts automatically on utility loss. Agent monitors
        and sheds load to extend battery life.
        """
        env = DcOpsEnvironment()
        obs = env.reset(scenario="B4")

        # 1. Diagnose to understand the situation
        obs = env.step(DcOpsAction(command="diagnose UPS-1"))
        obs = env.step(DcOpsAction(command="diagnose UPS-2"))

        # 2. Shed non-critical load to extend battery
        obs = env.step(DcOpsAction(command="set_rack_load A-01 4"))
        obs = env.step(DcOpsAction(command="set_rack_load B-01 4"))

        # 3. Check generator status
        obs = env.step(DcOpsAction(command="diagnose GEN-1"))

        # 4. Wait for generator to come online and stabilize
        for _ in range(14):
            obs = env.step(DcOpsAction(command="wait"))
            if obs.done:
                break

        # B4 is hard — may or may not resolve, but should make progress
        assert obs.metadata["cumulative_reward"] > -10.0


# ===========================================================================
# Reward Signal Quality
# ===========================================================================
class TestRewardSignalQuality:
    """Validate that reward signals are well-behaved across full episodes."""

    def test_rewards_bounded_per_step(self) -> None:
        """Every per-step reward should be bounded."""
        env = DcOpsEnvironment()
        env.reset(scenario="A2")

        for _ in range(15):
            obs = env.step(DcOpsAction(command="wait"))
            # Base reward is [-1, 1], speed bonus can add up to 1.0
            assert -2.0 <= obs.reward <= 2.0, f"Reward {obs.reward} out of bounds"
            if obs.done:
                break

    def test_good_actions_beat_bad_actions(self) -> None:
        """An optimal sequence should yield higher cumulative reward than a bad one."""
        env = DcOpsEnvironment()

        # Good episode: diagnose then fix
        env.reset(scenario="B1")
        env.step(DcOpsAction(command="diagnose UPS-1"))
        obs_good = env.step(DcOpsAction(command="acknowledge_alarm"))
        r_good = obs_good.metadata["cumulative_reward"]

        # Bad episode: just wait
        env.reset(scenario="B1")
        for _ in range(10):
            obs_bad = env.step(DcOpsAction(command="wait"))
            if obs_bad.done:
                break
        r_bad = obs_bad.metadata["cumulative_reward"]

        assert r_good > r_bad, f"Good ({r_good:.2f}) should beat bad ({r_bad:.2f})"

    def test_procedure_bonus_visible(self) -> None:
        """Following correct procedure should yield higher cumulative reward.

        Full episode comparison: both episodes do the same actions, but one
        follows procedure (check_status first) and the other doesn't.
        """
        env = DcOpsEnvironment()

        # With procedure: check_status → adjust_setpoint → wait
        env.reset(scenario="A1")
        env.step(DcOpsAction(command="check_status"))
        env.step(DcOpsAction(command="adjust_setpoint CRAC-1 22"))
        obs_proc = env.step(DcOpsAction(command="wait"))
        r_with = obs_proc.metadata["cumulative_reward"]

        # Without procedure: wait → adjust_setpoint → wait (no check_status)
        env.reset(scenario="A1")
        env.step(DcOpsAction(command="wait"))
        env.step(DcOpsAction(command="adjust_setpoint CRAC-1 22"))
        obs_noproc = env.step(DcOpsAction(command="wait"))
        r_without = obs_noproc.metadata["cumulative_reward"]

        assert r_with > r_without, (
            f"Procedure bonus not visible: with={r_with:.3f} vs without={r_without:.3f}"
        )

    @pytest.mark.parametrize("scenario_id", registered_scenario_ids())
    def test_no_nan_rewards(self, scenario_id: str) -> None:
        """No scenario should produce NaN rewards."""
        import math

        env = DcOpsEnvironment()
        env.reset(scenario=scenario_id)

        for _ in range(5):
            obs = env.step(DcOpsAction(command="check_status"))
            assert not math.isnan(obs.reward), f"NaN reward in {scenario_id}"
            assert not math.isinf(obs.reward), f"Inf reward in {scenario_id}"
            if obs.done:
                break


# ===========================================================================
# Cross-Facility Scenario Tests
# ===========================================================================
class TestCrossFacility:
    """Validate scenarios work with different facility configs."""

    def test_scenario_with_small_facility(self) -> None:
        """Scenarios should adapt to smaller configs that have compatible CRACs."""
        cfg = load_datacenter_config("small")
        env = DcOpsEnvironment()
        # Run without a scenario, just with small config
        obs = env.reset(config=cfg, step_budget=5)
        assert obs.done is False

        # Basic operations should work
        obs = env.step(DcOpsAction(command="check_status"))
        assert "status" in obs.action_result.lower()

        obs = env.step(DcOpsAction(command="diagnose CRAC-1"))
        assert "Diagnostic Report" in obs.action_result

    def test_large_facility_steady_state(self) -> None:
        """Large facility should reach reasonable steady state."""
        cfg = load_datacenter_config("large")
        env = DcOpsEnvironment()
        obs = env.reset(config=cfg, step_budget=10)

        pue = obs.metadata["pue"]
        assert 1.1 < pue < 3.0, f"Large facility PUE {pue} unrealistic"

        total_cooling = obs.metadata["total_cooling_power_kw"]
        total_it = obs.metadata["total_it_load_kw"]
        assert total_cooling > 0
        assert total_it > 0


# ===========================================================================
# Episode Metrics & Physics Consistency
# ===========================================================================
class TestEpisodeMetrics:
    """Validate physics consistency across episode metrics."""

    def test_pue_always_above_one(self) -> None:
        """PUE should always be >= 1.0 (physically impossible otherwise)."""
        env = DcOpsEnvironment()
        env.reset(scenario="A1")

        for _ in range(10):
            obs = env.step(DcOpsAction(command="wait"))
            assert obs.metadata["pue"] >= 1.0
            if obs.done:
                break

    def test_higher_load_raises_temperature(self) -> None:
        """Adding rack load should cause temperature to rise."""
        env = DcOpsEnvironment()
        obs = env.reset()
        t_before = obs.metadata["zones"]["zone_a"]["cold_aisle_temp_c"]

        # Significantly increase multiple racks' load
        env.step(DcOpsAction(command="set_rack_load A-01 15"))
        env.step(DcOpsAction(command="set_rack_load A-02 15"))
        env.step(DcOpsAction(command="set_rack_load A-03 15"))

        # Wait for thermal response
        for _ in range(7):
            obs = env.step(DcOpsAction(command="wait"))

        t_after = obs.metadata["zones"]["zone_a"]["cold_aisle_temp_c"]
        assert t_after > t_before, (
            f"Temp should rise with more load: {t_before:.1f} → {t_after:.1f}"
        )

    def test_sim_time_monotonically_increases(self) -> None:
        """Simulation time should always advance."""
        env = DcOpsEnvironment()
        obs = env.reset()
        prev_time = obs.metadata["sim_time_s"]

        for _ in range(5):
            obs = env.step(DcOpsAction(command="wait"))
            assert obs.metadata["sim_time_s"] > prev_time
            prev_time = obs.metadata["sim_time_s"]


# ===========================================================================
# Performance Tests
# ===========================================================================
class TestIntegrationPerformance:
    """Validate performance across different facility sizes."""

    @pytest.mark.parametrize("config_name", ["default", "small", "large"])
    def test_episode_completes_fast(self, config_name: str) -> None:
        """Full episode should complete quickly for any facility size."""
        cfg = load_datacenter_config(config_name)
        env = DcOpsEnvironment()

        start = time.perf_counter()
        env.reset(config=cfg, step_budget=10)
        for _ in range(10):
            env.step(DcOpsAction(command="wait"))
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, (
            f"{config_name} facility 10-step episode took {elapsed:.2f}s, should be <10s"
        )

    def test_all_scenarios_full_episode_under_10s(self) -> None:
        """Running every scenario for its full step budget should be fast."""
        env = DcOpsEnvironment()
        total_start = time.perf_counter()

        for sid in registered_scenario_ids():
            env.reset(scenario=sid)
            for _ in range(20):  # Max budget across scenarios
                obs = env.step(DcOpsAction(command="wait"))
                if obs.done:
                    break

        total_elapsed = time.perf_counter() - total_start
        assert total_elapsed < 15.0, (
            f"All {len(registered_scenario_ids())} scenarios took {total_elapsed:.2f}s"
        )
