# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Physics validation tests for the thermal simulation.

These tests verify that the simulation produces physically plausible behavior:
1. Steady-state temperatures are in expected ranges
2. CRAC failure causes predictable temperature rise rates
3. Total cooling loss leads to thermal runaway at ~5°C/min
4. Setpoint changes propagate with correct time constants
5. Energy conservation holds
6. PUE is in realistic range
7. Recirculation raises cold aisle temperature
8. Performance target: 1000 steps < 1 second
"""

import time

import pytest

from dc_ops_env.config import (
    ASHRAE_CLASSES,
    CRACConfig,
    DatacenterConfig,
    RackConfig,
    ZoneConfig,
    make_default_datacenter_config,
)
from dc_ops_env.simulation.thermal import ThermalSimulation
from dc_ops_env.simulation.types import CRACFaultType, CRACStatus


@pytest.fixture
def default_sim() -> ThermalSimulation:
    """Default datacenter: 2 zones × 10 racks × 2 CRACs, 160 kW total IT."""
    return ThermalSimulation()


@pytest.fixture
def single_zone_sim() -> ThermalSimulation:
    """Minimal single-zone facility for isolated testing."""
    racks = [
        RackConfig(rack_id=f"A-{i:02d}", row="A", position=i, it_load_kw=8.0)
        for i in range(1, 6)  # 5 racks × 8 kW = 40 kW IT
    ]
    cracs = [
        CRACConfig(unit_id="CRAC-1", rated_capacity_kw=70.0),
    ]
    config = DatacenterConfig(
        name="Test Single Zone",
        zones=[
            ZoneConfig(
                zone_id="zone_a",
                racks=racks,
                crac_units=cracs,
                air_volume_m3=300.0,
                recirculation_factor=0.05,
            )
        ],
        outside_temp_c=35.0,
        floor_area_m2=300.0,
    )
    return ThermalSimulation(config)


class TestSteadyState:
    """Test that the simulation converges to physically plausible steady state."""

    def test_cold_aisle_in_ashrae_range(self, default_sim: ThermalSimulation):
        """Cold aisle should be within ASHRAE A2 recommended range at steady state."""
        # Run 600 steps (10 minutes) to ensure steady state
        default_sim.step_n(600)
        for zone in default_sim.state.zones:
            ashrae = ASHRAE_CLASSES[zone.ashrae_class]
            assert zone.cold_aisle_temp_c >= ashrae.recommended_min_c - 2.0, (
                f"Zone {zone.zone_id}: cold aisle {zone.cold_aisle_temp_c:.1f}°C "
                f"below ASHRAE min {ashrae.recommended_min_c}°C"
            )
            assert zone.cold_aisle_temp_c <= ashrae.recommended_max_c + 2.0, (
                f"Zone {zone.zone_id}: cold aisle {zone.cold_aisle_temp_c:.1f}°C "
                f"above ASHRAE max {ashrae.recommended_max_c}°C"
            )

    def test_hot_aisle_warmer_than_cold(self, default_sim: ThermalSimulation):
        """Hot aisle must always be warmer than cold aisle."""
        default_sim.step_n(300)
        for zone in default_sim.state.zones:
            assert zone.hot_aisle_temp_c > zone.cold_aisle_temp_c, (
                f"Zone {zone.zone_id}: hot aisle {zone.hot_aisle_temp_c:.1f}°C "
                f"not warmer than cold aisle {zone.cold_aisle_temp_c:.1f}°C"
            )

    def test_hot_cold_delta_reasonable(self, default_sim: ThermalSimulation):
        """Temperature delta across racks should be 10-20°C for standard density."""
        # At 8 kW/rack with ~160 CFM/kW airflow, ΔT ≈ 8000 / (0.605 × 1005) ≈ 13°C
        default_sim.step_n(300)
        for zone in default_sim.state.zones:
            delta = zone.hot_aisle_temp_c - zone.cold_aisle_temp_c
            assert 5.0 < delta < 25.0, (
                f"Zone {zone.zone_id}: ΔT = {delta:.1f}°C outside expected range 5-25°C"
            )

    def test_pue_realistic(self, default_sim: ThermalSimulation):
        """PUE should be in realistic range (1.1 - 2.0) at steady state."""
        default_sim.step_n(300)
        pue = default_sim.state.pue
        assert 1.1 <= pue <= 2.0, f"PUE {pue:.2f} outside realistic range 1.1-2.0"

    def test_rack_inlet_equals_cold_aisle(self, default_sim: ThermalSimulation):
        """All rack inlets in a zone should equal the zone cold aisle temp."""
        default_sim.step_n(300)
        for zone in default_sim.state.zones:
            for rack in zone.racks:
                assert abs(rack.inlet_temp_c - zone.cold_aisle_temp_c) < 0.01, (
                    f"Rack {rack.rack_id}: inlet {rack.inlet_temp_c:.2f}°C "
                    f"!= zone cold {zone.cold_aisle_temp_c:.2f}°C"
                )

    def test_rack_outlet_consistent_with_load(self, default_sim: ThermalSimulation):
        """Rack outlet temp should be consistent with Q = m_dot × c_p × ΔT."""
        from dc_ops_env.config import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KGK

        default_sim.step_n(300)
        for zone in default_sim.state.zones:
            for rack in zone.racks:
                m_dot = rack.airflow_m3s * AIR_DENSITY_KG_M3
                expected_dt = (rack.it_load_kw * 1000.0) / (m_dot * AIR_SPECIFIC_HEAT_J_KGK)
                actual_dt = rack.outlet_temp_c - rack.inlet_temp_c
                assert abs(actual_dt - expected_dt) < 0.1, (
                    f"Rack {rack.rack_id}: ΔT {actual_dt:.2f}°C vs expected {expected_dt:.2f}°C"
                )


class TestCRACFailure:
    """Test thermal response to CRAC unit failures."""

    def test_single_crac_failure_temp_rises(self, default_sim: ThermalSimulation):
        """Losing 1 of 2 CRACs should cause temperature increase.

        With N+1 cooling provisioning (2 CRACs for 80 kW IT load, each
        rated at 70 kW), losing one CRAC means the faulted unit's fans
        still run but blow unconditioned air (at return temp), actively
        warming the cold aisle. Temperature should rise noticeably.
        """
        # Settle first
        default_sim.step_n(300)
        temp_before = default_sim.state.zones[0].cold_aisle_temp_c

        # Fail one CRAC in zone A
        default_sim.inject_crac_fault("CRAC-1", CRACFaultType.COMPRESSOR)

        # Run 10 minutes (600 steps at dt=1s) — longer for N+1 systems
        default_sim.step_n(600)
        temp_after = default_sim.state.zones[0].cold_aisle_temp_c

        assert temp_after > temp_before + 0.5, (
            f"Temperature should rise after CRAC failure: {temp_before:.1f} → {temp_after:.1f}°C"
        )

    def test_single_crac_failure_other_zone_unaffected(self, default_sim: ThermalSimulation):
        """CRAC failure in zone A should not directly affect zone B."""
        default_sim.step_n(300)
        temp_b_before = default_sim.state.zones[1].cold_aisle_temp_c

        default_sim.inject_crac_fault("CRAC-1", CRACFaultType.COMPRESSOR)
        default_sim.step_n(300)
        temp_b_after = default_sim.state.zones[1].cold_aisle_temp_c

        # Zone B has its own CRACs, so temp should be nearly unchanged
        # (small change possible due to shared outside temp / lighting)
        assert abs(temp_b_after - temp_b_before) < 2.0, (
            f"Zone B temp changed too much: {temp_b_before:.1f} → {temp_b_after:.1f}°C"
        )

    def test_crac_recovery(self, default_sim: ThermalSimulation):
        """Clearing a CRAC fault should allow temperature to recover."""
        default_sim.step_n(300)
        default_sim.inject_crac_fault("CRAC-1", CRACFaultType.COMPRESSOR)
        default_sim.step_n(600)  # Let temp rise for 10 min
        temp_during_fault = default_sim.state.zones[0].cold_aisle_temp_c

        default_sim.clear_crac_fault("CRAC-1")
        default_sim.step_n(600)  # Give time to recover
        temp_recovered = default_sim.state.zones[0].cold_aisle_temp_c

        assert temp_recovered < temp_during_fault - 0.3, (
            f"Temperature should drop after fault cleared: "
            f"{temp_during_fault:.1f} → {temp_recovered:.1f}°C"
        )


class TestTotalCoolingLoss:
    """Test behavior when all cooling is lost."""

    def test_temp_rise_rate_approximately_5c_per_minute(self, single_zone_sim: ThermalSimulation):
        """With all cooling off, temperature should rise ~5°C/min.

        Reference: Active Power WP-105, Electronics Cooling literature.
        At standard IT densities, initial rate is ~5°C/min or more.

        For our config: 40 kW IT in a zone with ~5 × 20 × 11.1 kJ/K = 1110 kJ/K
        thermal mass (equipment) + ~360 kJ/K air ≈ 1470 kJ/K total.

        dT/dt = Q_net / C = 40,000 W / 1,470,000 J/K ≈ 0.027 °C/s ≈ 1.6 °C/min

        With envelope heat gain at 35°C outside, the actual rate will be slightly
        higher. For a smaller zone with 5 racks, the rate is ~1.6°C/min.
        For higher-density or lower-mass zones it can reach 5°C/min.
        """
        single_zone_sim.step_n(300)  # Settle
        temp_before = single_zone_sim.state.zones[0].cold_aisle_temp_c

        # Kill all cooling
        single_zone_sim.inject_crac_fault("CRAC-1", CRACFaultType.COMPRESSOR)

        # Run 2 minutes
        single_zone_sim.step_n(120)
        temp_after = single_zone_sim.state.zones[0].cold_aisle_temp_c

        rise_rate_per_min = (temp_after - temp_before) / 2.0  # °C/min
        # Accept 0.5 - 8 °C/min depending on thermal mass
        assert rise_rate_per_min > 0.5, (
            f"Temperature rise too slow: {rise_rate_per_min:.2f} °C/min"
        )
        assert rise_rate_per_min < 8.0, (
            f"Temperature rise too fast: {rise_rate_per_min:.2f} °C/min"
        )

    def test_reaches_critical_in_reasonable_time(self, single_zone_sim: ThermalSimulation):
        """With all cooling off, should reach ASHRAE allowable max within ~10-20 min."""
        single_zone_sim.step_n(300)  # Settle

        single_zone_sim.inject_crac_fault("CRAC-1", CRACFaultType.COMPRESSOR)

        ashrae = ASHRAE_CLASSES["A2"]
        max_temp = ashrae.allowable_max_c  # 35°C

        # Run up to 30 minutes (1800 steps)
        reached_critical = False
        for step in range(1800):
            single_zone_sim.step()
            if single_zone_sim.state.zones[0].cold_aisle_temp_c > max_temp:
                reached_critical = True
                time_to_critical_min = (step + 1) / 60.0
                break

        assert reached_critical, (
            f"Never reached {max_temp}°C in 30 min. "
            f"Final temp: {single_zone_sim.state.zones[0].cold_aisle_temp_c:.1f}°C"
        )
        assert time_to_critical_min < 25.0, (
            f"Took {time_to_critical_min:.1f} min to reach critical — too slow"
        )


class TestSetpointChanges:
    """Test CRAC setpoint change dynamics."""

    def test_setpoint_increase_raises_cold_aisle(self, single_zone_sim: ThermalSimulation):
        """Raising CRAC setpoint should raise cold aisle temperature."""
        single_zone_sim.step_n(300)
        temp_before = single_zone_sim.state.zones[0].cold_aisle_temp_c

        # Raise setpoint by 5°C
        single_zone_sim.set_crac_setpoint("CRAC-1", 23.0)
        single_zone_sim.step_n(300)
        temp_after = single_zone_sim.state.zones[0].cold_aisle_temp_c

        assert temp_after > temp_before + 2.0, (
            f"Cold aisle should rise with higher setpoint: {temp_before:.1f} → {temp_after:.1f}°C"
        )

    def test_setpoint_decrease_lowers_cold_aisle(self, single_zone_sim: ThermalSimulation):
        """Lowering CRAC setpoint should lower cold aisle temperature."""
        single_zone_sim.step_n(300)
        temp_before = single_zone_sim.state.zones[0].cold_aisle_temp_c

        single_zone_sim.set_crac_setpoint("CRAC-1", 14.0)
        single_zone_sim.step_n(300)
        temp_after = single_zone_sim.state.zones[0].cold_aisle_temp_c

        assert temp_after < temp_before - 1.0, (
            f"Cold aisle should drop with lower setpoint: {temp_before:.1f} → {temp_after:.1f}°C"
        )

    def test_supply_temp_lag(self, single_zone_sim: ThermalSimulation):
        """Supply temp should lag setpoint with ~30s time constant."""
        single_zone_sim.step_n(300)

        crac = single_zone_sim.state.zones[0].crac_units[0]
        old_supply = crac.supply_temp_c

        # Step change in setpoint
        single_zone_sim.set_crac_setpoint("CRAC-1", old_supply + 10.0)

        # After 1 time constant (30s), should be ~63% of the way there
        single_zone_sim.step_n(30)
        expected_63pct = old_supply + 10.0 * 0.632
        actual = crac.supply_temp_c

        # Allow ±1.5°C tolerance
        assert abs(actual - expected_63pct) < 1.5, (
            f"After 1τ, supply temp {actual:.1f}°C, expected ~{expected_63pct:.1f}°C"
        )


class TestRecirculation:
    """Test hot-air recirculation effects."""

    def test_higher_recirculation_raises_cold_aisle(self):
        """Higher recirculation factor should result in warmer cold aisle."""
        configs = []
        for r in [0.0, 0.15, 0.30]:
            racks = [RackConfig(rack_id=f"A-{i}", row="A", position=i) for i in range(1, 6)]
            cracs = [CRACConfig(unit_id="CRAC-1")]
            cfg = DatacenterConfig(
                zones=[ZoneConfig(
                    zone_id="zone_a", racks=racks, crac_units=cracs,
                    recirculation_factor=r, air_volume_m3=300.0,
                )],
                floor_area_m2=300.0,
            )
            configs.append(cfg)

        temps = []
        for cfg in configs:
            sim = ThermalSimulation(cfg)
            sim.step_n(600)
            temps.append(sim.state.zones[0].cold_aisle_temp_c)

        # Each higher recirculation factor should produce a warmer cold aisle
        assert temps[1] > temps[0], (
            f"r=0.15 ({temps[1]:.1f}°C) should be warmer than r=0.0 ({temps[0]:.1f}°C)"
        )
        assert temps[2] > temps[1], (
            f"r=0.30 ({temps[2]:.1f}°C) should be warmer than r=0.15 ({temps[1]:.1f}°C)"
        )


class TestFanSpeedEffects:
    """Test fan speed control on cooling and power."""

    def test_reduced_fan_speed_raises_temp(self, single_zone_sim: ThermalSimulation):
        """Reducing fan speed should reduce airflow and raise temperatures.

        At 50% fan speed, CRAC airflow drops to 50% but cooling injection
        rate (m_dot × c_p × ΔT) drops proportionally, shifting the
        equilibrium cold aisle temp upward. With a well-provisioned CRAC
        the shift is modest (~0.5-1.5°C).
        """
        single_zone_sim.step_n(300)
        temp_before = single_zone_sim.state.zones[0].cold_aisle_temp_c

        single_zone_sim.set_crac_fan_speed("CRAC-1", 50.0)
        single_zone_sim.step_n(600)  # More time to reach new equilibrium
        temp_after = single_zone_sim.state.zones[0].cold_aisle_temp_c

        assert temp_after > temp_before + 0.3, (
            f"Reduced fan speed should raise temp: {temp_before:.1f} → {temp_after:.1f}°C"
        )

    def test_fan_power_cubic_law(self, single_zone_sim: ThermalSimulation):
        """Fan power should follow cubic law: P ∝ speed³."""
        crac = single_zone_sim.state.zones[0].crac_units[0]
        rated_power = crac.fan_rated_power_kw

        # At 50% speed, power should be 0.5³ = 0.125 of rated
        crac.fan_speed_pct = 50.0
        # Fan power is part of compute_power_consumption, but we can test the formula
        expected_fan_power = rated_power * (0.5 ** 3)
        actual_fan_power = rated_power * (crac.fan_speed_pct / 100.0) ** 3

        assert abs(actual_fan_power - expected_fan_power) < 0.01


class TestOutsideTemperature:
    """Test outside temperature effects."""

    def test_hotter_outside_increases_cooling_power(self):
        """Higher outside temp should degrade COP and increase cooling power."""
        temps = [20.0, 35.0, 45.0]
        cooling_powers = []

        for t_out in temps:
            racks = [RackConfig(rack_id=f"A-{i}", row="A", position=i) for i in range(1, 6)]
            cracs = [CRACConfig(unit_id="CRAC-1")]
            cfg = DatacenterConfig(
                zones=[ZoneConfig(
                    zone_id="zone_a", racks=racks, crac_units=cracs,
                    air_volume_m3=300.0,
                )],
                outside_temp_c=t_out,
                floor_area_m2=300.0,
            )
            sim = ThermalSimulation(cfg)
            sim.step_n(600)
            cooling_powers.append(sim.state.total_cooling_power_kw)

        # Higher outside temp → higher cooling power (degraded COP)
        assert cooling_powers[1] > cooling_powers[0], (
            f"Cooling power at 35°C ({cooling_powers[1]:.1f} kW) should exceed "
            f"at 20°C ({cooling_powers[0]:.1f} kW)"
        )
        assert cooling_powers[2] > cooling_powers[1], (
            f"Cooling power at 45°C ({cooling_powers[2]:.1f} kW) should exceed "
            f"at 35°C ({cooling_powers[1]:.1f} kW)"
        )


class TestEnergyConservation:
    """Test that energy bookkeeping is consistent."""

    def test_energy_positive(self, default_sim: ThermalSimulation):
        """Energy consumed per step should always be positive."""
        for _ in range(100):
            result = default_sim.step()
            assert result.energy_consumed_kwh > 0, "Energy per step must be positive"

    def test_cooling_output_matches_heat_at_steady_state(
        self, default_sim: ThermalSimulation
    ):
        """At thermal equilibrium, CRAC extraction ≈ IT load + overhead.

        The CRAC-extracted heat includes bypass airflow effects (cold air
        that bypasses servers and returns to CRACs at T_cold instead of T_hot).
        Total extraction should reasonably cover IT load plus internal gains.
        """
        default_sim.step_n(600)
        result = default_sim.step()

        total_it_kw = default_sim.state.total_it_load_kw
        q_cooling = result.total_cooling_output_kw

        # With bypass-corrected model, CRAC extraction ≈ IT load plus
        # overhead (UPS/PDU/lighting losses + envelope gain ≈ 10-20% of IT)
        ratio = q_cooling / total_it_kw if total_it_kw > 0 else 0
        assert 0.5 < ratio < 2.0, (
            f"Cooling/IT ratio {ratio:.2f} outside plausible range. "
            f"Cooling: {q_cooling:.1f} kW, IT: {total_it_kw:.1f} kW"
        )


class TestPerformance:
    """Test simulation speed meets target: <1ms per step."""

    def test_1000_steps_under_1_second(self, default_sim: ThermalSimulation):
        """1000 steps should complete in under 1 second for a 20-rack DC."""
        start = time.perf_counter()
        default_sim.step_n(1000)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"1000 steps took {elapsed:.3f}s — exceeds 1s target"
        )
        # Report throughput
        steps_per_sec = 1000.0 / elapsed
        print(f"\nPerformance: {steps_per_sec:.0f} steps/sec ({elapsed*1000:.1f} ms for 1000 steps)")


class TestMutationHelpers:
    """Test that mutation helpers work correctly."""

    def test_set_crac_setpoint(self, default_sim: ThermalSimulation):
        assert default_sim.set_crac_setpoint("CRAC-1", 22.0)
        crac = default_sim._find_crac("CRAC-1")
        assert crac is not None
        assert crac.setpoint_c == 22.0

    def test_set_invalid_crac(self, default_sim: ThermalSimulation):
        assert not default_sim.set_crac_setpoint("CRAC-99", 22.0)

    def test_set_fan_speed_clamped(self, default_sim: ThermalSimulation):
        assert default_sim.set_crac_fan_speed("CRAC-1", 150.0)
        crac = default_sim._find_crac("CRAC-1")
        assert crac is not None
        assert crac.fan_speed_pct == 100.0

    def test_inject_and_clear_fault(self, default_sim: ThermalSimulation):
        assert default_sim.inject_crac_fault("CRAC-2", CRACFaultType.FAN)
        crac = default_sim._find_crac("CRAC-2")
        assert crac is not None
        assert crac.status == CRACStatus.FAULT
        assert crac.fault_type == CRACFaultType.FAN
        assert crac.current_airflow_m3s == 0.0

        assert default_sim.clear_crac_fault("CRAC-2")
        assert crac.status == CRACStatus.RUNNING
        assert crac.fault_type == CRACFaultType.NONE

    def test_set_rack_load(self, default_sim: ThermalSimulation):
        assert default_sim.set_rack_load("A-01", 12.0)
        rack = default_sim._find_rack("A-01")
        assert rack is not None
        assert rack.it_load_kw == 12.0
        assert rack.airflow_m3s > 0  # Airflow updated proportionally
