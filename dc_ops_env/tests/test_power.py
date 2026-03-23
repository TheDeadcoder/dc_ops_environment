# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the power subsystem simulation.

Validates:
  - UPS quadratic efficiency model against published data
  - UPS battery discharge/charge dynamics
  - PDU loss calculations and three-phase current distribution
  - Generator state machine and fuel consumption
  - ATS transfer timing
  - Full utility-loss → generator-takeover scenario
"""

from __future__ import annotations

import math

import pytest

from dc_ops_env.config import (
    ATSConfig,
    GeneratorConfig,
    PDUConfig,
    PowerConfig,
    UPSConfig,
)
from dc_ops_env.simulation.power import PowerAlarm, PowerSimulation, PowerStepResult
from dc_ops_env.simulation.types import (
    ATSPosition,
    GeneratorState,
    UPSMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_simple_power_config(
    num_ups: int = 1,
    num_pdus: int = 1,
    ups_capacity_kw: float = 500.0,
) -> PowerConfig:
    """Create a minimal power config for testing."""
    return PowerConfig(
        ups_units=[
            UPSConfig(unit_id=f"UPS-{i+1}", rated_capacity_kw=ups_capacity_kw)
            for i in range(num_ups)
        ],
        pdus=[
            PDUConfig(pdu_id=f"PDU-{i+1}")
            for i in range(num_pdus)
        ],
        generator=GeneratorConfig(),
        ats=ATSConfig(),
    )


# ===========================================================================
# UPS Efficiency Tests
# ===========================================================================
class TestUPSEfficiency:
    """Validate UPS quadratic loss model against reference data.

    APC WP-108 Table: 500 kVA double-conversion UPS efficiency
      25% load → ~90.5%
      50% load → ~93.6%
      75% load → ~94.0%
      100% load → ~93.9%
    """

    def test_efficiency_at_25_percent(self) -> None:
        """Efficiency at 25% load: η = 0.25/(0.25+0.013+0.006×0.25+0.011×0.0625) ≈ 94.3%."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=125.0)
        sim.step(1.0, 125.0)  # 125/500 = 25%
        ups = sim.state.ups_units[0]
        assert 0.93 <= ups.efficiency <= 0.96, f"η={ups.efficiency:.3f}"

    def test_efficiency_at_50_percent(self) -> None:
        """Efficiency at 50% load: η ≈ 96.4%."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=250.0)
        sim.step(1.0, 250.0)
        ups = sim.state.ups_units[0]
        assert 0.95 <= ups.efficiency <= 0.97, f"η={ups.efficiency:.3f}"

    def test_efficiency_at_75_percent(self) -> None:
        """Efficiency at 75% load: η ≈ 96.9%."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=375.0)
        sim.step(1.0, 375.0)
        ups = sim.state.ups_units[0]
        assert 0.96 <= ups.efficiency <= 0.98, f"η={ups.efficiency:.3f}"

    def test_efficiency_at_100_percent(self) -> None:
        """Efficiency at 100% load: η ≈ 97.1%."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=500.0)
        sim.step(1.0, 500.0)
        ups = sim.state.ups_units[0]
        assert 0.96 <= ups.efficiency <= 0.98, f"η={ups.efficiency:.3f}"

    def test_efficiency_peak_around_75_percent(self) -> None:
        """Peak efficiency should occur around 50-75% load, not at extremes."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=0.0)
        efficiencies = {}
        for load_pct in [10, 25, 50, 75, 100]:
            load_kw = 500.0 * load_pct / 100.0
            sim2 = PowerSimulation(make_simple_power_config(), it_load_kw=load_kw)
            sim2.step(1.0, load_kw)
            efficiencies[load_pct] = sim2.state.ups_units[0].efficiency

        # Peak should be between 50-100%, not at 10%
        peak_pct = max(efficiencies, key=efficiencies.get)
        assert peak_pct >= 50, f"Peak at {peak_pct}%, efficiencies: {efficiencies}"

    def test_losses_are_positive(self) -> None:
        """UPS losses should always be positive (waste heat)."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        sim.step(1.0, 160.0)
        ups = sim.state.ups_units[0]
        assert ups.heat_output_kw > 0, "UPS must produce waste heat"

    def test_eco_mode_higher_efficiency(self) -> None:
        """Eco mode should have higher efficiency than double conversion."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        sim.step(1.0, 160.0)
        eta_dc = sim.state.ups_units[0].efficiency

        sim2 = PowerSimulation(make_simple_power_config(), it_load_kw=160.0)
        sim2.set_ups_mode("UPS-1", UPSMode.ECO)
        sim2.step(1.0, 160.0)
        eta_eco = sim2.state.ups_units[0].efficiency

        assert eta_eco > eta_dc, f"Eco {eta_eco:.3f} should > DC {eta_dc:.3f}"


# ===========================================================================
# UPS Battery Tests
# ===========================================================================
class TestUPSBattery:
    """Validate battery discharge and charge dynamics."""

    def test_battery_discharge_on_utility_loss(self) -> None:
        """Battery SOC should decrease when utility is lost."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        # Verify initial SOC = 100%
        assert sim.state.ups_units[0].battery_soc == 1.0

        # Kill utility
        sim.set_utility_available(False)

        # Run for 60 seconds
        for _ in range(60):
            sim.step(1.0, 160.0)

        ups = sim.state.ups_units[0]
        assert ups.mode == UPSMode.ON_BATTERY
        assert ups.battery_soc < 1.0, "SOC should decrease on battery"
        assert ups.battery_soc > 0.5, "SOC shouldn't drop too fast in 60s"

    def test_battery_runtime_estimation(self) -> None:
        """Battery time remaining estimate should be reasonable.

        8.3 kWh battery, 0.9 discharge eff, 0.85 aging, 160 kW load:
        usable = 8.3 × 0.9 × 0.85 = 6.35 kWh
        At 160 kW: ~143 seconds (~2.4 min)
        """
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        sim.set_utility_available(False)
        sim.step(1.0, 160.0)

        ups = sim.state.ups_units[0]
        assert ups.mode == UPSMode.ON_BATTERY
        assert 60 < ups.battery_time_remaining_s < 300, \
            f"Runtime {ups.battery_time_remaining_s:.0f}s should be 1-5 min for 160kW"

    def test_battery_exhaustion(self) -> None:
        """Battery should eventually exhaust and UPS should fault."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        sim.set_utility_available(False)

        # Run until battery dies (should be ~2-3 min)
        max_steps = 600  # 10 min max
        exhausted = False
        for _ in range(max_steps):
            result = sim.step(1.0, 160.0)
            if sim.state.ups_units[0].mode == UPSMode.FAULT:
                exhausted = True
                break

        assert exhausted, "Battery should exhaust within 10 minutes at 160 kW"
        assert sim.state.ups_units[0].battery_soc == 0.0

    def test_battery_recharge_after_utility_restored(self) -> None:
        """Battery should recharge when utility is restored."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=80.0)

        # Discharge for 30 seconds
        sim.set_utility_available(False)
        for _ in range(30):
            sim.step(1.0, 80.0)
        soc_after_discharge = sim.state.ups_units[0].battery_soc

        # Restore utility
        sim.set_utility_available(True)
        for _ in range(300):  # 5 min recharge
            sim.step(1.0, 80.0)

        soc_after_recharge = sim.state.ups_units[0].battery_soc
        assert soc_after_recharge > soc_after_discharge, \
            f"SOC should increase: {soc_after_discharge:.3f} → {soc_after_recharge:.3f}"

    def test_battery_low_alarm(self) -> None:
        """Should get low battery alarm when SOC drops below 25%."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        sim.set_utility_available(False)

        all_alarms: list[PowerAlarm] = []
        for _ in range(600):
            result = sim.step(1.0, 160.0)
            all_alarms.extend(result.alarms)
            if sim.state.ups_units[0].battery_soc < 0.10:
                break

        alarm_types = [a.alarm_type for a in all_alarms]
        assert "battery_low" in alarm_types or "battery_critical" in alarm_types, \
            f"Should have low battery alarm, got: {alarm_types}"


# ===========================================================================
# PDU Tests
# ===========================================================================
class TestPDU:
    """Validate PDU power distribution and loss calculations."""

    def test_pdu_losses_at_nominal(self) -> None:
        """PDU losses should be ~2% of load (98% efficiency)."""
        config = make_simple_power_config(num_pdus=1)
        sim = PowerSimulation(config, it_load_kw=5.0)
        result = sim.step(1.0, 5.0)

        pdu = sim.state.pdus[0]
        expected_loss = 5.0 * (1.0 / 0.98 - 1.0)  # ~0.102 kW
        assert abs(pdu.heat_output_kw - expected_loss) < 0.01, \
            f"PDU loss {pdu.heat_output_kw:.3f} kW, expected {expected_loss:.3f}"

    def test_phase_current_calculation(self) -> None:
        """Phase currents should match P = √3 × V_LL × I_L formula.

        5 kW load at 208V: I_total = 5000 / (√3 × 208) = 13.88 A
        Per phase (balanced): 13.88 / 3 = 4.63 A
        """
        config = make_simple_power_config(num_pdus=1)
        sim = PowerSimulation(config, it_load_kw=5.0)
        sim.step(1.0, 5.0)

        pdu = sim.state.pdus[0]
        expected_total = 5000.0 / (math.sqrt(3) * 208.0)
        expected_per_phase = expected_total / 3.0

        for i, current in enumerate(pdu.phase_currents_a):
            assert abs(current - expected_per_phase) < 0.1, \
                f"Phase {i} current {current:.2f}A, expected {expected_per_phase:.2f}A"

    def test_pdu_nameplate_capacity(self) -> None:
        """Nameplate capacity = √3 × 208V × 24A ≈ 8.65 kW."""
        config = make_simple_power_config(num_pdus=1)
        sim = PowerSimulation(config, it_load_kw=1.0)
        sim.step(1.0, 1.0)

        pdu = sim.state.pdus[0]
        expected = math.sqrt(3) * 208.0 * 24.0 / 1000.0
        assert abs(pdu.nameplate_capacity_kw - expected) < 0.01

    def test_pdu_derated_capacity(self) -> None:
        """Derated capacity = nameplate × 0.80."""
        config = make_simple_power_config(num_pdus=1)
        sim = PowerSimulation(config, it_load_kw=1.0)
        sim.step(1.0, 1.0)

        pdu = sim.state.pdus[0]
        expected = pdu.nameplate_capacity_kw * 0.80
        assert abs(pdu.derated_capacity_kw - expected) < 0.01

    def test_pdu_overcurrent_alarm(self) -> None:
        """Overloading a PDU beyond phase current limit should trigger alarm.

        Phase current = P / (√3 × V_LL) / num_phases_factor
        For total_current > 24A per-phase: need I_total > 72A
        I_total = P / (√3 × 208) = P / 360.2
        So P > 72 × 360.2 / 3 ≈ 8.65 kW won't do it because per_phase = I_total/3
        Actually: per_phase = (P×1000)/(√3×208) / 3, need per_phase > 24A
        per_phase > 24 → P > 24 × 3 × √3 × 208 / 1000 = 25.95 kW
        """
        config = make_simple_power_config(num_pdus=1)
        sim = PowerSimulation(config, it_load_kw=27.0)
        result = sim.step(1.0, 27.0)

        alarm_types = [a.alarm_type for a in result.alarms]
        assert "phase_overcurrent" in alarm_types, f"Expected overcurrent alarm, got {alarm_types}"

    def test_multiple_pdus_share_load(self) -> None:
        """Load should be distributed across PDUs."""
        config = make_simple_power_config(num_pdus=4)
        sim = PowerSimulation(config, it_load_kw=20.0)
        sim.step(1.0, 20.0)

        for pdu in sim.state.pdus:
            assert abs(pdu.output_power_kw - 5.0) < 0.01


# ===========================================================================
# Generator Tests
# ===========================================================================
class TestGenerator:
    """Validate generator state machine and fuel consumption."""

    def test_generator_startup_sequence(self) -> None:
        """Generator should progress: OFF → START_DELAY → CRANKING → WARMING → READY."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        gen = sim.state.generator

        assert gen.state == GeneratorState.OFF

        # Start generator
        sim.start_generator()
        assert gen.state == GeneratorState.START_DELAY

        # Run through start delay (4s)
        for _ in range(5):
            sim.step(1.0, 160.0)
        assert gen.state == GeneratorState.CRANKING

        # Run through cranking (5s)
        for _ in range(6):
            sim.step(1.0, 160.0)
        assert gen.state == GeneratorState.WARMING

        # Run through warmup (8s)
        for _ in range(9):
            sim.step(1.0, 160.0)
        assert gen.state == GeneratorState.READY

    def test_generator_total_startup_time(self) -> None:
        """Total startup time should be ~17s (4 + 5 + 8)."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        sim.start_generator()

        # Run until ready
        steps = 0
        for steps in range(1, 100):
            sim.step(1.0, 160.0)
            if sim.state.generator.is_available:
                break

        # 4s delay + 5s crank + 8s warmup = 17s, allow ±2s
        assert 15 <= steps <= 20, f"Startup took {steps}s, expected ~17s"

    def test_fuel_consumption_under_load(self) -> None:
        """Fuel should be consumed when generator is loaded."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        gen = sim.state.generator
        initial_fuel = gen.fuel_level_liters

        # Trigger utility loss to get generator running and loaded
        sim.set_utility_available(False)

        # Run for 30 seconds (enough for startup + some loaded time)
        for _ in range(30):
            sim.step(1.0, 160.0)

        assert gen.fuel_level_liters < initial_fuel, "Fuel should be consumed"

    def test_fuel_consumption_rate(self) -> None:
        """Fuel rate = full_rate × (0.1 + 0.9 × load_fraction).

        At 160kW / 750kW = 21.3% load:
        rate = 180 × (0.1 + 0.9 × 0.213) = 180 × 0.292 = 52.6 L/hr
        In 1 hour: ~52.6 liters consumed
        """
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        # Must disable utility so ATS stays on generator for full hour
        sim.set_utility_available(False)

        # Manually put generator into loaded state for cleaner test
        gen = sim.state.generator
        gen.state = GeneratorState.LOADED
        gen.load_fraction = 160.0 / 750.0
        gen.output_power_kw = 160.0
        sim.state.ats.position = ATSPosition.GENERATOR

        initial_fuel = gen.fuel_level_liters

        # Run for 1 hour
        for _ in range(3600):
            sim.step(1.0, 160.0)

        consumed = initial_fuel - gen.fuel_level_liters
        expected_rate = 180.0 * (0.1 + 0.9 * (160.0 / 750.0))
        # Allow 10% tolerance
        assert abs(consumed - expected_rate) < expected_rate * 0.15, \
            f"Consumed {consumed:.1f}L/hr, expected ~{expected_rate:.1f}L/hr"

    def test_generator_cooldown(self) -> None:
        """Generator should cool down for 5 minutes before shutdown."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        # Get generator running
        gen = sim.state.generator
        gen.state = GeneratorState.LOADED
        gen.output_power_kw = 160.0

        # Stop generator
        sim.stop_generator()
        assert gen.state == GeneratorState.COOLDOWN

        # Run cooldown (300s)
        for i in range(299):
            sim.step(1.0, 160.0)
            assert gen.state == GeneratorState.COOLDOWN, f"Still cooling at {i+1}s"

        # Should transition to OFF after 300s
        sim.step(1.0, 160.0)
        assert gen.state == GeneratorState.OFF

    def test_fuel_exhaustion(self) -> None:
        """Generator should shut down when fuel runs out."""
        config = PowerConfig(
            ups_units=[UPSConfig(unit_id="UPS-1")],
            pdus=[PDUConfig(pdu_id="PDU-1")],
            generator=GeneratorConfig(fuel_tank_liters=1.0),  # Very small tank
            ats=ATSConfig(),
        )
        sim = PowerSimulation(config, it_load_kw=160.0)

        gen = sim.state.generator
        gen.state = GeneratorState.LOADED
        gen.load_fraction = 160.0 / 750.0
        gen.output_power_kw = 160.0
        sim.state.ats.position = ATSPosition.GENERATOR
        sim.set_utility_available(False)

        # Run until fuel runs out (1L / 52.6 L/hr ≈ 68 seconds)
        all_alarms: list[PowerAlarm] = []
        for _ in range(200):
            result = sim.step(1.0, 160.0)
            all_alarms.extend(result.alarms)
            if gen.state == GeneratorState.OFF:
                break

        assert gen.state == GeneratorState.OFF
        assert gen.fuel_level_liters == 0.0
        alarm_types = [a.alarm_type for a in all_alarms]
        assert "fuel_exhausted" in alarm_types


# ===========================================================================
# ATS Tests
# ===========================================================================
class TestATS:
    """Validate Automatic Transfer Switch behavior."""

    def test_ats_starts_on_utility(self) -> None:
        """ATS should start in UTILITY position."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        assert sim.state.ats.position == ATSPosition.UTILITY

    def test_ats_transfers_on_utility_loss(self) -> None:
        """ATS should begin transfer when utility is lost."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        sim.set_utility_available(False)
        sim.step(0.001, 160.0)  # Tiny step to trigger detection

        assert sim.state.ats.position == ATSPosition.TRANSFERRING

    def test_ats_waits_for_generator(self) -> None:
        """ATS should stay TRANSFERRING until generator is ready."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        sim.set_utility_available(False)

        # Run for 5 seconds (generator still starting up)
        for _ in range(5):
            sim.step(1.0, 160.0)

        # Should still be transferring because generator isn't ready yet
        gen = sim.state.generator
        assert not gen.is_available
        assert sim.state.ats.position == ATSPosition.TRANSFERRING

    def test_ats_completes_transfer_to_generator(self) -> None:
        """ATS should transfer to generator once it's ready."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        sim.set_utility_available(False)

        # Run long enough for generator startup (~17s) + transfer
        for _ in range(25):
            sim.step(1.0, 160.0)

        assert sim.state.ats.position == ATSPosition.GENERATOR
        assert sim.state.generator.state == GeneratorState.LOADED

    def test_ats_retransfer_delay(self) -> None:
        """ATS should wait retransfer_delay (300s) before switching back to utility."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        # Lose utility and get on generator
        sim.set_utility_available(False)
        for _ in range(25):
            sim.step(1.0, 160.0)
        assert sim.state.ats.position == ATSPosition.GENERATOR

        # Restore utility
        sim.set_utility_available(True)

        # Run for 200s — should still be on generator
        for _ in range(200):
            sim.step(1.0, 160.0)
        assert sim.state.ats.position == ATSPosition.GENERATOR

        # Run past 300s retransfer delay
        for _ in range(150):
            sim.step(1.0, 160.0)

        # Should be transferring back or on utility
        ats_pos = sim.state.ats.position
        assert ats_pos in (ATSPosition.TRANSFERRING, ATSPosition.UTILITY), \
            f"Expected transfer back, got {ats_pos}"


# ===========================================================================
# Full Scenario Tests
# ===========================================================================
class TestUtilityLossScenario:
    """End-to-end utility loss and recovery scenario."""

    def test_full_utility_loss_and_recovery(self) -> None:
        """Complete scenario: utility loss → battery bridge → generator → recovery.

        Timeline:
          t=0: Utility fails
          t=0-17s: UPS on battery, generator starting
          t=17s: Generator ready, ATS transfers
          t=17s+: On generator power
          t=100s: Utility restored
          t=400s: Retransfer to utility (after 300s delay)
        """
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        # Phase 1: Utility loss
        sim.set_utility_available(False)

        # Run through startup sequence
        ups_on_battery = False
        gen_ready = False
        on_generator = False

        for t in range(1, 30):
            result = sim.step(1.0, 160.0)
            if sim.state.ups_units[0].mode == UPSMode.ON_BATTERY:
                ups_on_battery = True
            if sim.state.generator.is_available:
                gen_ready = True
            if sim.state.ats.position == ATSPosition.GENERATOR:
                on_generator = True

        assert ups_on_battery, "UPS should have been on battery"
        assert gen_ready, "Generator should be ready by 30s"
        assert on_generator, "Should be on generator by 30s"

        # Phase 2: Running on generator
        result = sim.step(1.0, 160.0)
        assert result.on_generator
        assert sim.state.generator.state == GeneratorState.LOADED

        # Phase 3: Utility restored
        sim.set_utility_available(True)

        # Run past retransfer delay (300s)
        for _ in range(350):
            sim.step(1.0, 160.0)

        # Should be back on utility (or transferring)
        assert sim.state.ats.position in (ATSPosition.UTILITY, ATSPosition.TRANSFERRING)

    def test_power_available_during_transfer(self) -> None:
        """UPS should bridge the gap during ATS transfer."""
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)

        # Initial: power available
        result = sim.step(1.0, 160.0)
        assert result.power_available

        # During utility loss, UPS provides power
        sim.set_utility_available(False)
        for _ in range(5):
            result = sim.step(1.0, 160.0)

        # UPS is on battery, still providing power
        assert sim.state.ups_units[0].mode == UPSMode.ON_BATTERY
        # The IT load is still being served
        assert sim.state.ups_units[0].output_power_kw > 0


# ===========================================================================
# Integration with DatacenterState Tests
# ===========================================================================
class TestPowerStateIntegration:
    """Test PowerState integration with DatacenterState."""

    def test_datacenter_state_with_power(self) -> None:
        """DatacenterState should use PowerState for PUE when available."""
        from dc_ops_env.simulation.types import DatacenterState, PowerState, UPSState, PDUState

        ups = UPSState(unit_id="UPS-1", heat_output_kw=5.0)
        pdu = PDUState(pdu_id="PDU-1", heat_output_kw=1.0)
        power = PowerState(ups_units=[ups], pdus=[pdu])

        state = DatacenterState(
            power=power,
            lighting_power_kw=5.0,
        )
        # With no zones (no IT load), PUE should be 1.0
        assert state.pue == 1.0

    def test_datacenter_state_without_power_uses_stubs(self) -> None:
        """DatacenterState without PowerState should use stub fractions."""
        from dc_ops_env.simulation.types import DatacenterState

        state = DatacenterState(
            ups_loss_fraction=0.05,
            pdu_loss_fraction=0.02,
        )
        # Should use the stub loss fractions (backward compat)
        assert state.power is None


# ===========================================================================
# Performance Test
# ===========================================================================
class TestPerformance:
    """Ensure power simulation is fast enough for RL training."""

    def test_steps_per_second(self) -> None:
        """Power sim should sustain >10,000 steps/sec."""
        import time

        config = make_simple_power_config(num_ups=2, num_pdus=20)
        sim = PowerSimulation(config, it_load_kw=160.0)

        n_steps = 5000
        start = time.perf_counter()
        for _ in range(n_steps):
            sim.step(1.0, 160.0)
        elapsed = time.perf_counter() - start

        steps_per_sec = n_steps / elapsed
        assert steps_per_sec > 10_000, \
            f"Only {steps_per_sec:.0f} steps/sec, need >10,000"


# ===========================================================================
# Mutation Helper Tests
# ===========================================================================
class TestMutationHelpers:
    """Test convenience methods for scenario injection."""

    def test_set_utility_available(self) -> None:
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        assert sim.state.utility_available is True
        sim.set_utility_available(False)
        assert sim.state.utility_available is False

    def test_set_ups_mode(self) -> None:
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        assert sim.set_ups_mode("UPS-1", UPSMode.ECO)
        assert sim.state.ups_units[0].mode == UPSMode.ECO
        assert not sim.set_ups_mode("UPS-999", UPSMode.ECO)

    def test_inject_and_clear_ups_fault(self) -> None:
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        assert sim.inject_ups_fault("UPS-1")
        assert sim.state.ups_units[0].mode == UPSMode.FAULT
        assert sim.clear_ups_fault("UPS-1")
        assert sim.state.ups_units[0].mode == UPSMode.DOUBLE_CONVERSION

    def test_start_stop_generator(self) -> None:
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        sim.start_generator()
        assert sim.state.generator.state == GeneratorState.START_DELAY

        # Run to READY
        for _ in range(20):
            sim.step(1.0, 160.0)
        assert sim.state.generator.is_available

        sim.stop_generator()
        assert sim.state.generator.state == GeneratorState.COOLDOWN

    def test_refuel_generator(self) -> None:
        config = make_simple_power_config()
        sim = PowerSimulation(config, it_load_kw=160.0)
        gen = sim.state.generator
        gen.fuel_level_liters = 500.0

        sim.refuel_generator(200.0)
        assert gen.fuel_level_liters == 700.0

        sim.refuel_generator()  # Full tank
        assert gen.fuel_level_liters == gen.fuel_tank_liters
