# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Power subsystem simulation: UPS, PDU, Generator, ATS.

Models the electrical power chain from utility/generator through UPS and PDU
to IT loads. Tracks efficiency losses, battery state-of-charge, generator
fuel consumption, and automatic transfer switching.

Physics references:
  - UPS quadratic loss model: APC White Paper 108
  - PDU three-phase power: P = √3 × V_LL × I_L × PF
  - Generator fuel: linear with load fraction + 10% idle
  - ATS transfer: mechanical switch timing (50-200 ms)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import (
    ATSConfig,
    GeneratorConfig,
    PDUConfig,
    PowerConfig,
    UPSConfig,
)
from .types import (
    ATSPosition,
    ATSState,
    GeneratorState,
    GensetState,
    PDUState,
    PowerState,
    UPSMode,
    UPSState,
)


# ---------------------------------------------------------------------------
# Power step result
# ---------------------------------------------------------------------------
@dataclass
class PowerAlarm:
    """A power subsystem alarm."""
    component: str        # e.g. "UPS-1", "PDU-A1", "GEN-1", "ATS-1"
    alarm_type: str       # e.g. "on_battery", "low_battery", "overload", "fuel_low"
    severity: str         # "warning", "critical"
    message: str
    value: float = 0.0    # Relevant numeric value (SOC, load%, fuel level, etc.)


@dataclass
class PowerStepResult:
    """Result of a single power simulation step."""
    total_ups_loss_kw: float = 0.0
    total_pdu_loss_kw: float = 0.0
    total_power_overhead_kw: float = 0.0
    generator_output_kw: float = 0.0
    generator_fuel_remaining_liters: float = 0.0
    utility_available: bool = True
    on_generator: bool = False
    power_available: bool = True
    alarms: list[PowerAlarm] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Power simulation
# ---------------------------------------------------------------------------
class PowerSimulation:
    """Simulates the datacenter power distribution chain.

    Power flow:
        Utility/Generator → ATS → UPS(es) → PDU(s) → IT Load

    Each step():
      1. ATS: detect utility loss/restoration, manage transfer
      2. Generator: state machine (off → start_delay → cranking → warming → ready → loaded)
      3. UPS: compute efficiency, manage battery SOC
      4. PDU: compute losses, check phase currents
    """

    def __init__(self, power_config: PowerConfig, it_load_kw: float = 160.0) -> None:
        self._config = power_config
        self._state = self._init_state(power_config)
        self._it_load_kw = it_load_kw

    @property
    def state(self) -> PowerState:
        return self._state

    @staticmethod
    def _init_state(config: PowerConfig) -> PowerState:
        """Initialize power state from configuration."""
        ups_units = []
        for uc in config.ups_units:
            ups = UPSState(
                unit_id=uc.unit_id,
                mode=UPSMode(uc.initial_mode),
                rated_capacity_kw=uc.rated_capacity_kw,
                loss_c0=uc.loss_c0,
                loss_c1=uc.loss_c1,
                loss_c2=uc.loss_c2,
                battery_capacity_kwh=uc.battery_capacity_kwh,
                battery_discharge_efficiency=uc.battery_discharge_efficiency,
                battery_aging_factor=uc.battery_aging_factor,
                recharge_rate_kw=uc.recharge_rate_kw,
                battery_soc=1.0,
            )
            ups_units.append(ups)

        pdus = []
        for pc in config.pdus:
            pdu = PDUState(
                pdu_id=pc.pdu_id,
                voltage_ll_v=pc.voltage_ll_v,
                max_current_per_phase_a=pc.max_current_per_phase_a,
                num_phases=pc.num_phases,
                breaker_rating_a=pc.breaker_rating_a,
                efficiency=pc.efficiency,
                continuous_derating=pc.continuous_derating,
            )
            pdus.append(pdu)

        gen_cfg = config.generator
        generator = GensetState(
            gen_id=gen_cfg.gen_id,
            rated_capacity_kw=gen_cfg.rated_capacity_kw,
            start_delay_s=gen_cfg.start_delay_s,
            crank_time_s=gen_cfg.crank_time_s,
            warmup_time_s=gen_cfg.warmup_time_s,
            cooldown_time_s=gen_cfg.cooldown_time_s,
            fuel_tank_liters=gen_cfg.fuel_tank_liters,
            fuel_level_liters=gen_cfg.fuel_tank_liters,
            consumption_lph_full=gen_cfg.consumption_lph_full,
        )

        ats_cfg = config.ats
        ats = ATSState(
            ats_id=ats_cfg.ats_id,
            transfer_time_ms=ats_cfg.transfer_time_ms,
            retransfer_delay_s=ats_cfg.retransfer_delay_s,
        )

        return PowerState(
            ups_units=ups_units,
            pdus=pdus,
            generator=generator,
            ats=ats,
            utility_available=config.utility_available,
            utility_voltage_v=config.utility_voltage_v,
        )

    def step(self, dt_s: float, it_load_kw: float) -> PowerStepResult:
        """Advance the power simulation by dt_s seconds.

        Args:
            dt_s: Timestep in seconds.
            it_load_kw: Total IT power demand in kW.

        Returns:
            PowerStepResult with losses, alarms, and status.
        """
        self._it_load_kw = it_load_kw
        alarms: list[PowerAlarm] = []

        # 1. ATS logic: detect utility state changes
        self._step_ats(dt_s, alarms)

        # 2. Generator state machine
        self._step_generator(dt_s, alarms)

        # 3. Determine if load-side power is available
        power_available = self._state.power_available

        # 4. UPS: efficiency, battery, losses
        total_ups_loss = self._step_ups_units(dt_s, it_load_kw, alarms)

        # 5. PDU: losses, phase currents
        total_pdu_loss = self._step_pdus(it_load_kw, alarms)

        return PowerStepResult(
            total_ups_loss_kw=total_ups_loss,
            total_pdu_loss_kw=total_pdu_loss,
            total_power_overhead_kw=total_ups_loss + total_pdu_loss,
            generator_output_kw=self._state.generator.output_power_kw,
            generator_fuel_remaining_liters=self._state.generator.fuel_level_liters,
            utility_available=self._state.utility_available,
            on_generator=self._state.on_generator,
            power_available=power_available,
            alarms=alarms,
        )

    # -------------------------------------------------------------------
    # ATS
    # -------------------------------------------------------------------
    def _step_ats(self, dt_s: float, alarms: list[PowerAlarm]) -> None:
        """Handle ATS transfer logic."""
        ats = self._state.ats
        gen = self._state.generator
        utility_ok = self._state.utility_available

        if ats.position == ATSPosition.UTILITY:
            if not utility_ok:
                # Utility lost — initiate transfer to generator
                ats.position = ATSPosition.TRANSFERRING
                ats.transfer_elapsed_ms = 0.0
                ats.retransfer_timer_s = 0.0
                # Start generator if not already running
                if gen.state == GeneratorState.OFF:
                    gen.state = GeneratorState.START_DELAY
                    gen.state_elapsed_s = 0.0
                alarms.append(PowerAlarm(
                    component=ats.ats_id,
                    alarm_type="utility_lost",
                    severity="critical",
                    message="Utility power lost, initiating transfer to generator",
                ))

        elif ats.position == ATSPosition.TRANSFERRING:
            ats.transfer_elapsed_ms += dt_s * 1000.0
            if ats.transfer_elapsed_ms >= ats.transfer_time_ms:
                # Transfer complete
                if utility_ok:
                    # Utility came back during transfer — go back to utility
                    ats.position = ATSPosition.UTILITY
                    ats.transfer_elapsed_ms = 0.0
                elif gen.is_available:
                    ats.position = ATSPosition.GENERATOR
                    ats.transfer_elapsed_ms = 0.0
                    alarms.append(PowerAlarm(
                        component=ats.ats_id,
                        alarm_type="on_generator",
                        severity="warning",
                        message="Load transferred to generator",
                    ))
                # else: stay transferring until generator is ready

        elif ats.position == ATSPosition.GENERATOR:
            if utility_ok:
                # Utility restored — wait retransfer delay before switching back
                ats.retransfer_timer_s += dt_s
                if ats.retransfer_timer_s >= ats.retransfer_delay_s:
                    ats.position = ATSPosition.TRANSFERRING
                    ats.transfer_elapsed_ms = 0.0
                    alarms.append(PowerAlarm(
                        component=ats.ats_id,
                        alarm_type="retransfer",
                        severity="warning",
                        message="Utility restored, initiating retransfer",
                    ))
            else:
                ats.retransfer_timer_s = 0.0

    # -------------------------------------------------------------------
    # Generator
    # -------------------------------------------------------------------
    def _step_generator(self, dt_s: float, alarms: list[PowerAlarm]) -> None:
        """Advance generator state machine."""
        gen = self._state.generator

        if gen.state == GeneratorState.OFF:
            gen.output_power_kw = 0.0
            gen.load_fraction = 0.0
            gen.fuel_consumption_lph = 0.0
            return

        gen.state_elapsed_s += dt_s

        if gen.state == GeneratorState.START_DELAY:
            if gen.state_elapsed_s >= gen.start_delay_s:
                gen.state = GeneratorState.CRANKING
                gen.state_elapsed_s = 0.0

        elif gen.state == GeneratorState.CRANKING:
            if gen.state_elapsed_s >= gen.crank_time_s:
                gen.state = GeneratorState.WARMING
                gen.state_elapsed_s = 0.0
                alarms.append(PowerAlarm(
                    component=gen.gen_id,
                    alarm_type="engine_started",
                    severity="warning",
                    message="Generator engine started, warming up",
                ))

        elif gen.state == GeneratorState.WARMING:
            # Idle fuel consumption during warmup
            gen.fuel_consumption_lph = gen.consumption_lph_full * 0.1
            self._consume_fuel(gen, dt_s)
            if gen.state_elapsed_s >= gen.warmup_time_s:
                gen.state = GeneratorState.READY
                gen.state_elapsed_s = 0.0
                alarms.append(PowerAlarm(
                    component=gen.gen_id,
                    alarm_type="ready",
                    severity="warning",
                    message="Generator ready to accept load",
                ))

        elif gen.state == GeneratorState.READY:
            gen.fuel_consumption_lph = gen.consumption_lph_full * 0.1
            self._consume_fuel(gen, dt_s)
            # If ATS has switched to generator, transition to loaded
            if self._state.ats.position == ATSPosition.GENERATOR:
                gen.state = GeneratorState.LOADED
                gen.state_elapsed_s = 0.0

        elif gen.state == GeneratorState.LOADED:
            gen.load_fraction = min(self._it_load_kw / gen.rated_capacity_kw, 1.0)
            gen.output_power_kw = min(self._it_load_kw, gen.rated_capacity_kw)
            gen.fuel_consumption_lph = gen.compute_fuel_consumption_lph()
            self._consume_fuel(gen, dt_s)

            # Check fuel level
            if gen.fuel_level_liters <= 0:
                gen.fuel_level_liters = 0.0
                gen.state = GeneratorState.OFF
                gen.output_power_kw = 0.0
                alarms.append(PowerAlarm(
                    component=gen.gen_id,
                    alarm_type="fuel_exhausted",
                    severity="critical",
                    message="Generator fuel exhausted — engine shutdown",
                ))
            elif gen.fuel_remaining_hours < 2.0:
                alarms.append(PowerAlarm(
                    component=gen.gen_id,
                    alarm_type="fuel_low",
                    severity="warning",
                    message=f"Generator fuel low: {gen.fuel_level_liters:.0f}L "
                            f"(~{gen.fuel_remaining_hours:.1f}h remaining)",
                    value=gen.fuel_level_liters,
                ))

            # If utility is back and ATS has switched away, go to cooldown
            if self._state.ats.position != ATSPosition.GENERATOR:
                gen.state = GeneratorState.COOLDOWN
                gen.state_elapsed_s = 0.0
                gen.output_power_kw = 0.0
                gen.load_fraction = 0.0

        elif gen.state == GeneratorState.COOLDOWN:
            gen.output_power_kw = 0.0
            gen.load_fraction = 0.0
            gen.fuel_consumption_lph = gen.consumption_lph_full * 0.1
            self._consume_fuel(gen, dt_s)
            if gen.state_elapsed_s >= gen.cooldown_time_s:
                gen.state = GeneratorState.OFF
                gen.state_elapsed_s = 0.0
                gen.fuel_consumption_lph = 0.0
                alarms.append(PowerAlarm(
                    component=gen.gen_id,
                    alarm_type="shutdown",
                    severity="warning",
                    message="Generator cooldown complete, engine off",
                ))

    @staticmethod
    def _consume_fuel(gen: GensetState, dt_s: float) -> None:
        """Consume fuel for the given timestep."""
        if gen.fuel_consumption_lph > 0:
            consumed = gen.fuel_consumption_lph * dt_s / 3600.0  # hours → seconds
            gen.fuel_level_liters = max(0.0, gen.fuel_level_liters - consumed)

    # -------------------------------------------------------------------
    # UPS
    # -------------------------------------------------------------------
    def _step_ups_units(
        self, dt_s: float, it_load_kw: float, alarms: list[PowerAlarm]
    ) -> float:
        """Step all UPS units and return total UPS losses in kW."""
        if not self._state.ups_units:
            return 0.0

        # Distribute IT load evenly across UPS units
        load_per_ups = it_load_kw / len(self._state.ups_units)
        total_loss = 0.0

        for ups in self._state.ups_units:
            loss = self._step_single_ups(ups, dt_s, load_per_ups, alarms)
            total_loss += loss

        return total_loss

    def _step_single_ups(
        self,
        ups: UPSState,
        dt_s: float,
        load_kw: float,
        alarms: list[PowerAlarm],
    ) -> float:
        """Step a single UPS unit. Returns loss in kW."""
        ups.output_power_kw = load_kw
        ups.load_fraction = load_kw / ups.rated_capacity_kw if ups.rated_capacity_kw > 0 else 0.0

        utility_ok = self._state.utility_available
        ats_ok = self._state.ats.load_powered

        # Mode transitions
        if ups.mode == UPSMode.FAULT:
            # Fault state: no output, no charging
            ups.efficiency = 0.0
            ups.heat_output_kw = 0.0
            ups.input_power_kw = 0.0
            ups.battery_power_kw = 0.0
            return 0.0

        if ups.mode == UPSMode.BYPASS:
            # Bypass: no UPS processing, minimal losses
            ups.efficiency = 1.0
            ups.heat_output_kw = 0.0
            ups.input_power_kw = load_kw
            ups.battery_power_kw = 0.0
            return 0.0

        # Check if we need to switch to battery
        source_ok = utility_ok and ats_ok
        if ups.mode == UPSMode.ON_BATTERY:
            if source_ok:
                # Source restored — switch back to normal mode
                ups.mode = UPSMode.DOUBLE_CONVERSION
                alarms.append(PowerAlarm(
                    component=ups.unit_id,
                    alarm_type="utility_restored",
                    severity="warning",
                    message=f"UPS {ups.unit_id} back on utility power",
                ))
        elif not source_ok and ups.mode in (
            UPSMode.DOUBLE_CONVERSION, UPSMode.LINE_INTERACTIVE, UPSMode.ECO
        ):
            ups.mode = UPSMode.ON_BATTERY
            alarms.append(PowerAlarm(
                component=ups.unit_id,
                alarm_type="on_battery",
                severity="critical",
                message=f"UPS {ups.unit_id} switched to battery",
                value=ups.battery_soc,
            ))

        # Compute efficiency based on mode
        if ups.mode == UPSMode.ECO:
            # Eco mode: ~99% efficiency (minimal processing)
            ups.efficiency = 0.99
        elif ups.mode == UPSMode.LINE_INTERACTIVE:
            # Line interactive: ~97% (some processing)
            ups.efficiency = min(0.97, ups.compute_efficiency() + 0.03)
        else:
            # Double conversion or on_battery: full quadratic model
            ups.efficiency = ups.compute_efficiency()

        # Compute losses
        if ups.efficiency > 0:
            ups_loss = load_kw * (1.0 / ups.efficiency - 1.0)
        else:
            ups_loss = ups.rated_capacity_kw * ups.loss_c0
        ups.heat_output_kw = ups_loss
        ups.input_power_kw = load_kw + ups_loss

        # Battery management
        if ups.mode == UPSMode.ON_BATTERY:
            # Discharging: SOC decreases
            # P_discharge = P_output / η_discharge (battery must supply more than output)
            p_discharge = load_kw / ups.battery_discharge_efficiency if ups.battery_discharge_efficiency > 0 else load_kw
            ups.battery_power_kw = p_discharge
            energy_used_kwh = p_discharge * dt_s / 3600.0
            effective_capacity = ups.battery_capacity_kwh * ups.battery_aging_factor
            if effective_capacity > 0:
                ups.battery_soc -= energy_used_kwh / effective_capacity
            ups.battery_soc = max(0.0, ups.battery_soc)
            ups.battery_time_remaining_s = ups.compute_battery_time_remaining_s()
            ups.input_power_kw = 0.0  # Not drawing from mains

            # Battery alarms
            if ups.battery_soc <= 0.0:
                ups.mode = UPSMode.FAULT
                alarms.append(PowerAlarm(
                    component=ups.unit_id,
                    alarm_type="battery_exhausted",
                    severity="critical",
                    message=f"UPS {ups.unit_id} battery exhausted — load unprotected",
                ))
            elif ups.battery_soc < 0.10:
                alarms.append(PowerAlarm(
                    component=ups.unit_id,
                    alarm_type="battery_critical",
                    severity="critical",
                    message=f"UPS {ups.unit_id} battery critical: {ups.battery_soc*100:.0f}%",
                    value=ups.battery_soc,
                ))
            elif ups.battery_soc < 0.25:
                alarms.append(PowerAlarm(
                    component=ups.unit_id,
                    alarm_type="battery_low",
                    severity="warning",
                    message=f"UPS {ups.unit_id} battery low: {ups.battery_soc*100:.0f}%",
                    value=ups.battery_soc,
                ))
        else:
            # On mains — charge battery if not full
            ups.battery_power_kw = 0.0
            ups.battery_time_remaining_s = float("inf")
            if ups.battery_soc < 1.0:
                charge_kw = min(ups.recharge_rate_kw, ups.rated_capacity_kw * 0.1)
                energy_charged_kwh = charge_kw * dt_s / 3600.0
                effective_capacity = ups.battery_capacity_kwh * ups.battery_aging_factor
                if effective_capacity > 0:
                    ups.battery_soc += energy_charged_kwh / effective_capacity
                ups.battery_soc = min(1.0, ups.battery_soc)
                ups.battery_power_kw = -charge_kw  # Negative = charging
                ups.input_power_kw += charge_kw  # Charging draws additional power

        # Overload alarm
        if ups.load_fraction > 1.0:
            alarms.append(PowerAlarm(
                component=ups.unit_id,
                alarm_type="overload",
                severity="critical",
                message=f"UPS {ups.unit_id} overloaded at {ups.load_fraction*100:.0f}%",
                value=ups.load_fraction,
            ))

        return ups_loss

    # -------------------------------------------------------------------
    # PDU
    # -------------------------------------------------------------------
    def _step_pdus(
        self, it_load_kw: float, alarms: list[PowerAlarm]
    ) -> float:
        """Step all PDUs and return total PDU losses in kW."""
        if not self._state.pdus:
            return 0.0

        # Distribute IT load evenly across PDUs
        load_per_pdu = it_load_kw / len(self._state.pdus)
        total_loss = 0.0

        for pdu in self._state.pdus:
            loss = self._step_single_pdu(pdu, load_per_pdu, alarms)
            total_loss += loss

        return total_loss

    def _step_single_pdu(
        self,
        pdu: PDUState,
        load_kw: float,
        alarms: list[PowerAlarm],
    ) -> float:
        """Step a single PDU. Returns loss in kW."""
        pdu.output_power_kw = load_kw
        pdu.input_power_kw = load_kw / pdu.efficiency if pdu.efficiency > 0 else load_kw
        pdu_loss = pdu.input_power_kw - pdu.output_power_kw
        pdu.heat_output_kw = pdu_loss

        # Compute per-phase currents (assume balanced load across phases)
        # P = √3 × V_LL × I_L × PF (assume PF = 1.0 for IT loads with PFC)
        if pdu.voltage_ll_v > 0:
            total_current = (load_kw * 1000.0) / (math.sqrt(3) * pdu.voltage_ll_v)
            per_phase = total_current / pdu.num_phases if pdu.num_phases > 0 else total_current
            pdu.phase_currents_a = [per_phase] * pdu.num_phases
        else:
            pdu.phase_currents_a = [0.0] * pdu.num_phases

        # Load fraction of derated capacity
        derated = pdu.derated_capacity_kw
        pdu.load_fraction = load_kw / derated if derated > 0 else 0.0

        # Phase imbalance (0 for balanced load — will be nonzero when
        # individual rack loads are modeled)
        pdu.phase_imbalance_pct = pdu.compute_phase_imbalance()

        # Check overload
        max_phase_current = max(pdu.phase_currents_a) if pdu.phase_currents_a else 0.0
        if max_phase_current > pdu.max_current_per_phase_a:
            pdu.overload = True
            alarms.append(PowerAlarm(
                component=pdu.pdu_id,
                alarm_type="phase_overcurrent",
                severity="critical",
                message=f"PDU {pdu.pdu_id} phase overcurrent: "
                        f"{max_phase_current:.1f}A > {pdu.max_current_per_phase_a:.0f}A",
                value=max_phase_current,
            ))
        else:
            pdu.overload = False

        # Breaker trip check (per-branch, simplified as aggregate)
        if max_phase_current > pdu.breaker_rating_a / pdu.continuous_derating:
            pdu.breaker_tripped = True
            alarms.append(PowerAlarm(
                component=pdu.pdu_id,
                alarm_type="breaker_trip",
                severity="critical",
                message=f"PDU {pdu.pdu_id} breaker tripped",
                value=max_phase_current,
            ))

        # Warn on high utilization
        if pdu.load_fraction > 0.80 and not pdu.overload:
            alarms.append(PowerAlarm(
                component=pdu.pdu_id,
                alarm_type="high_utilization",
                severity="warning",
                message=f"PDU {pdu.pdu_id} at {pdu.load_fraction*100:.0f}% of derated capacity",
                value=pdu.load_fraction,
            ))

        return pdu_loss

    # -------------------------------------------------------------------
    # Mutation helpers (for agent actions)
    # -------------------------------------------------------------------
    def set_utility_available(self, available: bool) -> None:
        """Set utility power availability (for scenario injection)."""
        self._state.utility_available = available

    def set_ups_mode(self, unit_id: str, mode: UPSMode) -> bool:
        """Manually set UPS operating mode. Returns True if found."""
        for ups in self._state.ups_units:
            if ups.unit_id == unit_id:
                ups.mode = mode
                return True
        return False

    def inject_ups_fault(self, unit_id: str) -> bool:
        """Put a UPS into fault mode. Returns True if found."""
        return self.set_ups_mode(unit_id, UPSMode.FAULT)

    def clear_ups_fault(self, unit_id: str) -> bool:
        """Restore a faulted UPS to double conversion. Returns True if found."""
        for ups in self._state.ups_units:
            if ups.unit_id == unit_id and ups.mode == UPSMode.FAULT:
                ups.mode = UPSMode.DOUBLE_CONVERSION
                return True
        return False

    def start_generator(self) -> None:
        """Manually start the generator."""
        gen = self._state.generator
        if gen.state == GeneratorState.OFF:
            gen.state = GeneratorState.START_DELAY
            gen.state_elapsed_s = 0.0

    def stop_generator(self) -> None:
        """Initiate generator cooldown/shutdown."""
        gen = self._state.generator
        if gen.state in (GeneratorState.READY, GeneratorState.LOADED):
            gen.state = GeneratorState.COOLDOWN
            gen.state_elapsed_s = 0.0
            gen.output_power_kw = 0.0
            gen.load_fraction = 0.0

    def refuel_generator(self, liters: float | None = None) -> None:
        """Refuel the generator (default: full tank)."""
        gen = self._state.generator
        if liters is None:
            gen.fuel_level_liters = gen.fuel_tank_liters
        else:
            gen.fuel_level_liters = min(
                gen.fuel_level_liters + liters,
                gen.fuel_tank_liters,
            )
