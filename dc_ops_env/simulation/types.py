# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Runtime state dataclasses for the datacenter simulation.

These are plain dataclasses (not Pydantic) for performance on the simulation
hot path. Pydantic models are only used at the API boundary (models.py).

All values in SI units:
  - Temperature: °C
  - Power/Heat: kW (for readability; converted to W in physics calculations)
  - Airflow: m³/s
  - Thermal capacitance: J/K
  - Thermal resistance: K/W
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CRACStatus(Enum):
    RUNNING = "running"
    STANDBY = "standby"
    FAULT = "fault"
    MAINTENANCE = "maintenance"


class CRACFaultType(Enum):
    NONE = "none"
    COMPRESSOR = "compressor"
    FAN = "fan"
    REFRIGERANT_LEAK = "refrigerant_leak"
    SENSOR = "sensor"
    ELECTRICAL = "electrical"


# ---------------------------------------------------------------------------
# Power subsystem enums
# ---------------------------------------------------------------------------
class UPSMode(Enum):
    """UPS operating mode."""
    DOUBLE_CONVERSION = "double_conversion"  # Normal: AC→DC→AC, full protection
    LINE_INTERACTIVE = "line_interactive"     # Reduced losses, slower transfer
    ECO = "eco"                              # Bypass with monitoring, minimal losses
    BYPASS = "bypass"                        # Manual bypass, no protection
    ON_BATTERY = "on_battery"               # Utility lost, discharging battery
    FAULT = "fault"                          # UPS fault, load on raw utility or dead


class GeneratorState(Enum):
    """Diesel generator state machine states."""
    OFF = "off"                  # Not running
    START_DELAY = "start_delay"  # Programmed delay before cranking
    CRANKING = "cranking"        # Engine cranking
    WARMING = "warming"          # Warm-up period before load acceptance
    READY = "ready"              # Running, ready to accept load
    LOADED = "loaded"            # Running under load
    COOLDOWN = "cooldown"        # Unloaded cool-down before shutdown


class ATSPosition(Enum):
    """ATS switch position."""
    UTILITY = "utility"
    GENERATOR = "generator"
    TRANSFERRING = "transferring"  # Mid-transfer (load momentarily interrupted)


@dataclass
class RackState:
    """Runtime state of a single server rack."""
    rack_id: str
    row: str
    position: int

    # Electrical / thermal load
    it_load_kw: float            # Current IT power draw (≈ heat dissipation)

    # Temperatures
    inlet_temp_c: float          # Cold aisle side (server intake)
    outlet_temp_c: float         # Hot aisle side (server exhaust)

    # Airflow
    airflow_m3s: float           # Total server fan airflow through this rack

    # Thermal inertia
    thermal_mass_jk: float       # Equipment thermal capacitance [J/K]

    def compute_outlet_temp(self) -> float:
        """Compute outlet temp from energy balance: Q = m_dot * c_p * dT.

        Returns outlet temperature in °C.
        """
        from ..config import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KGK

        m_dot = self.airflow_m3s * AIR_DENSITY_KG_M3  # kg/s
        if m_dot <= 0:
            # No airflow — temperature rises unboundedly in theory;
            # clamp to a high value to signal danger
            return self.inlet_temp_c + 50.0

        q_w = self.it_load_kw * 1000.0  # Convert kW → W
        delta_t = q_w / (m_dot * AIR_SPECIFIC_HEAT_J_KGK)
        return self.inlet_temp_c + delta_t


@dataclass
class CRACState:
    """Runtime state of a CRAC/CRAH cooling unit."""
    unit_id: str

    # Operating status
    status: CRACStatus = CRACStatus.RUNNING
    fault_type: CRACFaultType = CRACFaultType.NONE

    # Setpoints and actuals
    setpoint_c: float = 18.0          # Desired supply air temperature
    supply_temp_c: float = 18.0       # Actual supply air temperature (lags setpoint)
    fan_speed_pct: float = 100.0      # 0-100

    # Rated specifications (from config, immutable during episode)
    max_airflow_m3s: float = 0.0      # At 100% fan speed
    rated_capacity_kw: float = 70.0   # At rated return temp
    rated_return_temp_c: float = 24.0  # Return temp for rated capacity
    capacity_slope_per_c: float = 0.03 # Fractional capacity change per °C
    fan_rated_power_kw: float = 5.0
    cop_rated: float = 3.5
    cop_degradation_per_c: float = 0.04
    supply_temp_lag_s: float = 30.0    # First-order lag time constant

    @property
    def current_airflow_m3s(self) -> float:
        """Actual airflow based on fan speed and status."""
        if self.status != CRACStatus.RUNNING:
            return 0.0
        if self.fault_type == CRACFaultType.FAN:
            return 0.0
        return self.max_airflow_m3s * (self.fan_speed_pct / 100.0)

    def compute_cooling_output_kw(self, return_air_temp_c: float) -> float:
        """Compute actual cooling output [kW].

        Cooling capacity depends on return air temperature:
            Q_actual = Q_rated × [1 + α × (T_return - T_rated)]

        But is also limited by airflow × deltaT:
            Q_airflow = m_dot × c_p × (T_return - T_supply)

        The actual output is the minimum of both limits.
        """
        from ..config import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KGK

        if self.status != CRACStatus.RUNNING:
            return 0.0
        if self.fault_type in (CRACFaultType.COMPRESSOR, CRACFaultType.REFRIGERANT_LEAK):
            return 0.0

        # Capacity limit (refrigeration cycle capacity)
        delta_return = return_air_temp_c - self.rated_return_temp_c
        q_capacity = self.rated_capacity_kw * (1.0 + self.capacity_slope_per_c * delta_return)
        q_capacity = max(q_capacity, 0.0)

        # Airflow limit
        m_dot = self.current_airflow_m3s * AIR_DENSITY_KG_M3  # kg/s
        if m_dot <= 0:
            return 0.0
        delta_t = return_air_temp_c - self.supply_temp_c
        if delta_t <= 0:
            return 0.0
        q_airflow = m_dot * AIR_SPECIFIC_HEAT_J_KGK * delta_t / 1000.0  # W → kW

        return min(q_capacity, q_airflow)

    def compute_power_consumption_kw(
        self, cooling_output_kw: float, outside_temp_c: float
    ) -> float:
        """Compute CRAC electrical power consumption [kW].

        Fan power: cubic relationship with speed (affinity laws).
            P_fan = P_rated × (speed/100)³

        Compressor power: Q_cooling / COP
            COP degrades at higher outside temperatures.
        """
        if self.status != CRACStatus.RUNNING:
            return 0.0

        # Fan power (affinity law: power ∝ speed³)
        speed_frac = self.fan_speed_pct / 100.0
        p_fan = self.fan_rated_power_kw * (speed_frac ** 3)

        # Compressor power
        cop = self.cop_rated
        if outside_temp_c > 35.0:
            # COP degrades linearly above 35°C
            cop *= max(0.3, 1.0 - self.cop_degradation_per_c * (outside_temp_c - 35.0))

        if self.fault_type in (CRACFaultType.COMPRESSOR, CRACFaultType.REFRIGERANT_LEAK):
            p_compressor = 0.0
        elif cop > 0 and cooling_output_kw > 0:
            p_compressor = cooling_output_kw / cop
        else:
            p_compressor = 0.0

        return p_fan + p_compressor

    def update_supply_temp(self, dt_s: float) -> None:
        """First-order lag: supply temp approaches setpoint with time constant.

        T_supply(t+dt) = T_supply(t) + (T_setpoint - T_supply(t)) × (1 - e^(-dt/τ))

        For small dt/τ this approximates: T += (T_set - T) × dt/τ
        """
        import math

        if self.status != CRACStatus.RUNNING:
            return
        if self.fault_type == CRACFaultType.COMPRESSOR:
            # Compressor fault: supply temp drifts toward return air (no cooling)
            return
        if self.supply_temp_lag_s <= 0:
            self.supply_temp_c = self.setpoint_c
            return

        alpha = 1.0 - math.exp(-dt_s / self.supply_temp_lag_s)
        self.supply_temp_c += (self.setpoint_c - self.supply_temp_c) * alpha


@dataclass
class ZoneState:
    """Runtime state of a thermal zone (a section of the datacenter)."""
    zone_id: str

    # Temperatures
    cold_aisle_temp_c: float = 20.0
    hot_aisle_temp_c: float = 35.0

    # Humidity (tracked, not yet fully modeled psychrometrically)
    humidity_rh: float = 0.45          # Fraction 0-1

    # Containment / recirculation
    recirculation_factor: float = 0.08  # 0 = perfect containment

    # Equipment
    racks: list[RackState] = field(default_factory=list)
    crac_units: list[CRACState] = field(default_factory=list)

    # Zone thermal properties
    air_volume_m3: float = 500.0
    envelope_r_kw: float = 0.02       # Thermal resistance to outside [K/W]

    # ASHRAE class for this zone
    ashrae_class: str = "A2"

    @property
    def total_it_load_kw(self) -> float:
        return sum(r.it_load_kw for r in self.racks)

    @property
    def total_rack_airflow_m3s(self) -> float:
        return sum(r.airflow_m3s for r in self.racks)

    @property
    def total_crac_airflow_m3s(self) -> float:
        return sum(c.current_airflow_m3s for c in self.crac_units)

    @property
    def max_inlet_temp_c(self) -> float:
        if not self.racks:
            return self.cold_aisle_temp_c
        return max(r.inlet_temp_c for r in self.racks)

    def compute_thermal_capacitance_jk(self) -> float:
        """Total thermal capacitance of this zone [J/K].

        C_total = C_air + C_equipment

        C_air = ρ × V × c_p  (~1-2 kJ/K for typical zone)
        C_equipment = Σ rack thermal masses  (dominant term, ~100+ kJ/K)
        """
        from ..config import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KGK

        c_air = AIR_DENSITY_KG_M3 * self.air_volume_m3 * AIR_SPECIFIC_HEAT_J_KGK
        c_equipment = sum(r.thermal_mass_jk for r in self.racks)
        return c_air + c_equipment


# ---------------------------------------------------------------------------
# Power subsystem state
# ---------------------------------------------------------------------------
@dataclass
class UPSState:
    """Runtime state of a UPS unit."""
    unit_id: str

    # Operating mode
    mode: UPSMode = UPSMode.DOUBLE_CONVERSION

    # Load
    input_power_kw: float = 0.0       # Power entering UPS from utility/generator
    output_power_kw: float = 0.0      # Power delivered to IT load
    load_fraction: float = 0.0        # output / rated_capacity (0-1)

    # Efficiency and losses
    efficiency: float = 0.97          # Current operating efficiency
    heat_output_kw: float = 0.0       # Waste heat = input - output

    # Battery
    battery_soc: float = 1.0          # State of charge (0-1)
    battery_power_kw: float = 0.0     # Positive = discharging, negative = charging
    battery_time_remaining_s: float = 0.0  # Estimated time at current draw

    # Rated specs (from config, immutable during episode)
    rated_capacity_kw: float = 500.0
    loss_c0: float = 0.013
    loss_c1: float = 0.006
    loss_c2: float = 0.011
    battery_capacity_kwh: float = 8.3
    battery_discharge_efficiency: float = 0.90
    battery_aging_factor: float = 0.85
    recharge_rate_kw: float = 5.0

    def compute_efficiency(self) -> float:
        """UPS efficiency using quadratic loss model.

        η(x) = x / (x + c_0 + c_1·x + c_2·x²)
        where x = load_fraction.

        At very low loads (x < 0.01), efficiency is undefined / near-zero.
        """
        x = self.load_fraction
        if x < 0.01:
            return 0.0
        denominator = x + self.loss_c0 + self.loss_c1 * x + self.loss_c2 * x * x
        if denominator <= 0:
            return 0.0
        return x / denominator

    def compute_losses_kw(self) -> float:
        """Power losses at current load.

        P_loss = P_output × (1/η - 1)
        """
        eta = self.compute_efficiency()
        if eta <= 0:
            # No-load losses: just transformer/control board idle draw
            return self.rated_capacity_kw * self.loss_c0
        return self.output_power_kw * (1.0 / eta - 1.0)

    def compute_battery_time_remaining_s(self) -> float:
        """Estimate remaining battery runtime at current discharge rate.

        t = (SOC × E_battery × η_discharge × aging) / P_discharge
        """
        if self.battery_power_kw <= 0:
            return float("inf")
        usable_kwh = (
            self.battery_soc
            * self.battery_capacity_kwh
            * self.battery_discharge_efficiency
            * self.battery_aging_factor
        )
        return usable_kwh / self.battery_power_kw * 3600.0  # hours → seconds


@dataclass
class PDUState:
    """Runtime state of a three-phase PDU."""
    pdu_id: str

    # Per-phase currents [A]
    phase_currents_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Power
    input_power_kw: float = 0.0
    output_power_kw: float = 0.0
    heat_output_kw: float = 0.0       # Transformer losses

    # Utilization
    load_fraction: float = 0.0        # Of derated capacity
    phase_imbalance_pct: float = 0.0  # Max deviation from average phase current

    # Alarms
    breaker_tripped: bool = False
    overload: bool = False

    # Rated specs (from config)
    voltage_ll_v: float = 208.0
    max_current_per_phase_a: float = 24.0
    num_phases: int = 3
    breaker_rating_a: float = 20.0
    efficiency: float = 0.98
    continuous_derating: float = 0.80

    @property
    def nameplate_capacity_kw(self) -> float:
        """Total nameplate capacity: P = √3 × V_LL × I_phase × num_phases_factor."""
        import math
        return math.sqrt(3) * self.voltage_ll_v * self.max_current_per_phase_a / 1000.0

    @property
    def derated_capacity_kw(self) -> float:
        """NEC 80% continuous derating applied."""
        return self.nameplate_capacity_kw * self.continuous_derating

    def compute_phase_imbalance(self) -> float:
        """Phase imbalance as percentage deviation from average.

        imbalance = max(|I_phase - I_avg|) / I_avg × 100
        Returns 0 if no load.
        """
        if not self.phase_currents_a:
            return 0.0
        avg = sum(self.phase_currents_a) / len(self.phase_currents_a)
        if avg <= 0:
            return 0.0
        max_dev = max(abs(i - avg) for i in self.phase_currents_a)
        return max_dev / avg * 100.0

    def compute_heat_output_kw(self) -> float:
        """PDU transformer losses."""
        return self.input_power_kw * (1.0 - self.efficiency)


@dataclass
class GensetState:
    """Runtime state of a diesel standby generator."""
    gen_id: str

    # State machine
    state: GeneratorState = GeneratorState.OFF
    state_elapsed_s: float = 0.0      # Time in current state

    # Output
    output_power_kw: float = 0.0
    load_fraction: float = 0.0        # output / rated_capacity

    # Fuel
    fuel_level_liters: float = 2000.0
    fuel_consumption_lph: float = 0.0  # Current consumption rate

    # Timing specs (from config)
    rated_capacity_kw: float = 750.0
    start_delay_s: float = 4.0
    crank_time_s: float = 5.0
    warmup_time_s: float = 8.0
    cooldown_time_s: float = 300.0
    fuel_tank_liters: float = 2000.0
    consumption_lph_full: float = 180.0

    @property
    def is_available(self) -> bool:
        """Generator is ready to accept load."""
        return self.state in (GeneratorState.READY, GeneratorState.LOADED)

    @property
    def fuel_remaining_hours(self) -> float:
        """Estimated hours of fuel at current consumption rate."""
        if self.fuel_consumption_lph <= 0:
            return float("inf")
        return self.fuel_level_liters / self.fuel_consumption_lph

    def compute_fuel_consumption_lph(self) -> float:
        """Fuel consumption scales roughly linearly with load.

        Includes ~10% idle consumption when running unloaded.
        """
        if self.state == GeneratorState.OFF:
            return 0.0
        if self.state in (GeneratorState.CRANKING, GeneratorState.START_DELAY):
            return 0.0  # Not yet burning fuel
        # Idle + proportional: consumption = full × (0.1 + 0.9 × load_fraction)
        return self.consumption_lph_full * (0.1 + 0.9 * self.load_fraction)


@dataclass
class ATSState:
    """Runtime state of an Automatic Transfer Switch."""
    ats_id: str

    position: ATSPosition = ATSPosition.UTILITY
    transfer_elapsed_ms: float = 0.0   # Progress through transfer

    # Specs (from config)
    transfer_time_ms: float = 100.0
    retransfer_delay_s: float = 300.0

    # Timer for retransfer delay (counts up when utility is restored)
    retransfer_timer_s: float = 0.0

    @property
    def load_powered(self) -> bool:
        """Whether the load side has power (False only during transfer gap)."""
        return self.position != ATSPosition.TRANSFERRING


@dataclass
class PowerState:
    """Aggregated power subsystem state."""
    ups_units: list[UPSState] = field(default_factory=list)
    pdus: list[PDUState] = field(default_factory=list)
    generator: GensetState = field(default_factory=lambda: GensetState(gen_id="GEN-1"))
    ats: ATSState = field(default_factory=lambda: ATSState(ats_id="ATS-1"))

    # Utility
    utility_available: bool = True
    utility_voltage_v: float = 480.0

    @property
    def total_ups_loss_kw(self) -> float:
        return sum(u.heat_output_kw for u in self.ups_units)

    @property
    def total_pdu_loss_kw(self) -> float:
        return sum(p.heat_output_kw for p in self.pdus)

    @property
    def total_power_overhead_kw(self) -> float:
        """Total electrical overhead from power distribution."""
        return self.total_ups_loss_kw + self.total_pdu_loss_kw

    @property
    def on_generator(self) -> bool:
        return self.ats.position == ATSPosition.GENERATOR

    @property
    def power_available(self) -> bool:
        """Whether load-side power is available (from any source)."""
        if not self.ats.load_powered:
            return False
        if self.ats.position == ATSPosition.UTILITY:
            return self.utility_available
        if self.ats.position == ATSPosition.GENERATOR:
            return self.generator.is_available
        return False


@dataclass
class DatacenterState:
    """Top-level simulation state aggregating all subsystems."""
    zones: list[ZoneState] = field(default_factory=list)

    # Power subsystem (None = use stub loss fractions for backward compat)
    power: PowerState | None = None

    # Environment
    outside_temp_c: float = 35.0
    outside_humidity_rh: float = 0.40

    # Facility overhead
    lighting_power_kw: float = 5.0       # Total lighting load

    # Power distribution stub losses (fractions of IT load)
    # Used only when power subsystem is not initialized
    ups_loss_fraction: float = 0.05
    pdu_loss_fraction: float = 0.02

    # Simulation clock
    sim_time_s: float = 0.0

    @property
    def total_it_load_kw(self) -> float:
        return sum(z.total_it_load_kw for z in self.zones)

    @property
    def total_cooling_power_kw(self) -> float:
        total = 0.0
        for zone in self.zones:
            for crac in zone.crac_units:
                q_cool = crac.compute_cooling_output_kw(zone.hot_aisle_temp_c)
                total += crac.compute_power_consumption_kw(q_cool, self.outside_temp_c)
        return total

    @property
    def pue(self) -> float:
        """Dynamic PUE = Total Facility Power / IT Power.

        When power subsystem is active, uses real UPS/PDU losses.
        Otherwise falls back to stub loss fractions.
        """
        p_it = self.total_it_load_kw
        if p_it <= 0:
            return 1.0

        p_cooling = self.total_cooling_power_kw

        if self.power is not None:
            p_distribution_loss = self.power.total_power_overhead_kw
        else:
            p_distribution_loss = p_it * (self.ups_loss_fraction + self.pdu_loss_fraction)

        p_total = p_it + p_cooling + p_distribution_loss + self.lighting_power_kw
        return p_total / p_it
