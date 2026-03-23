# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Physical constants, ASHRAE thermal guidelines, and unit conversion utilities.

All internal simulation values use SI units:
  - Temperature: °C (Celsius)
  - Power/Heat: W (Watts)
  - Energy: J (Joules)
  - Airflow: m³/s
  - Thermal capacitance: J/K
  - Thermal resistance: K/W
  - Time: s (seconds)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Air properties (dry air at standard conditions: ~20 °C, 101.325 kPa)
# ---------------------------------------------------------------------------
AIR_DENSITY_KG_M3 = 1.2
AIR_SPECIFIC_HEAT_J_KGK = 1005.0
AIR_RHO_CP = AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KGK  # 1206.0 J/(m³·K)

# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
CFM_TO_M3S = 4.71947e-4       # 1 CFM = 4.71947 × 10⁻⁴ m³/s
M3S_TO_CFM = 1.0 / CFM_TO_M3S  # ≈ 2118.88
TONS_TO_KW = 3.517             # 1 ton of refrigeration = 3.517 kW thermal
KW_TO_TONS = 1.0 / TONS_TO_KW
BTU_HR_TO_W = 0.29307107      # 1 BTU/hr = 0.293 W
W_TO_BTU_HR = 1.0 / BTU_HR_TO_W  # ≈ 3.412


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def cfm_to_m3s(cfm: float) -> float:
    return cfm * CFM_TO_M3S


def m3s_to_cfm(m3s: float) -> float:
    return m3s * M3S_TO_CFM


# ---------------------------------------------------------------------------
# ASHRAE TC 9.9 Thermal Guidelines, 5th Edition (2021)
#
# Each class defines recommended and allowable operating envelopes for
# server inlet temperatures and humidity.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ASHRAEClass:
    """ASHRAE thermal envelope for a given equipment class."""
    name: str
    recommended_min_c: float
    recommended_max_c: float
    allowable_min_c: float
    allowable_max_c: float
    max_dew_point_c: float
    max_rh: float              # Fraction, e.g. 0.80 = 80%
    description: str = ""


ASHRAE_A1 = ASHRAEClass(
    name="A1",
    recommended_min_c=18.0,
    recommended_max_c=27.0,
    allowable_min_c=15.0,
    allowable_max_c=32.0,
    max_dew_point_c=17.0,
    max_rh=0.80,
    description="Enterprise servers, storage",
)

ASHRAE_A2 = ASHRAEClass(
    name="A2",
    recommended_min_c=18.0,
    recommended_max_c=27.0,
    allowable_min_c=10.0,
    allowable_max_c=35.0,
    max_dew_point_c=21.0,
    max_rh=0.80,
    description="Volume servers",
)

ASHRAE_A3 = ASHRAEClass(
    name="A3",
    recommended_min_c=18.0,
    recommended_max_c=27.0,
    allowable_min_c=5.0,
    allowable_max_c=40.0,
    max_dew_point_c=24.0,
    max_rh=0.85,
    description="Extended temperature range",
)

ASHRAE_A4 = ASHRAEClass(
    name="A4",
    recommended_min_c=18.0,
    recommended_max_c=27.0,
    allowable_min_c=5.0,
    allowable_max_c=45.0,
    max_dew_point_c=24.0,
    max_rh=0.90,
    description="Maximum temperature flexibility",
)

ASHRAE_H1 = ASHRAEClass(
    name="H1",
    recommended_min_c=18.0,
    recommended_max_c=22.0,
    allowable_min_c=5.0,
    allowable_max_c=25.0,
    max_dew_point_c=17.0,
    max_rh=0.80,
    description="High-density / AI / HPC",
)

ASHRAE_CLASSES: dict[str, ASHRAEClass] = {
    "A1": ASHRAE_A1,
    "A2": ASHRAE_A2,
    "A3": ASHRAE_A3,
    "A4": ASHRAE_A4,
    "H1": ASHRAE_H1,
}

# Minimum humidity boundary (all classes):
# Higher of dew point -12 °C OR 8% RH
ASHRAE_MIN_DEW_POINT_C = -12.0
ASHRAE_MIN_RH = 0.08

# Rate-of-change limits
ASHRAE_RATE_LIMIT_SOLID_STATE_C_PER_HR = 20.0    # °C/hr max
ASHRAE_RATE_LIMIT_SOLID_STATE_C_PER_15MIN = 5.0  # °C per 15 min max

# Sensor accuracy
ASHRAE_SENSOR_ACCURACY_STANDARD_C = 0.5
ASHRAE_SENSOR_ACCURACY_HIGH_DENSITY_C = 0.3


# ---------------------------------------------------------------------------
# Default datacenter configuration
# ---------------------------------------------------------------------------
@dataclass
class CRACConfig:
    """Configuration for a single CRAC/CRAH unit."""
    unit_id: str = "CRAC-1"
    rated_capacity_kw: float = 70.0        # Nominal cooling capacity at rated conditions
    rated_return_temp_c: float = 24.0       # Return air temp at which capacity is rated
    capacity_slope_per_c: float = 0.03      # Fractional capacity increase per °C above rated return
    max_airflow_cfm: float = 12000.0        # Maximum airflow at 100% fan speed
    fan_rated_power_kw: float = 5.0         # Fan power at 100% speed
    cop_rated: float = 3.5                  # Coefficient of performance at design conditions
    cop_degradation_per_c: float = 0.04     # COP fractional decrease per °C outside temp above 35°C
    initial_setpoint_c: float = 18.0        # Default supply air setpoint
    initial_fan_speed_pct: float = 100.0    # Default fan speed
    supply_temp_lag_s: float = 30.0         # Time constant for supply temp to reach setpoint


@dataclass
class RackConfig:
    """Configuration for a single server rack."""
    rack_id: str = "A-01"
    row: str = "A"
    position: int = 1
    it_load_kw: float = 8.0                # IT power draw
    num_servers_2u: int = 20               # Number of 2U servers
    server_thermal_mass_jk: float = 11100.0  # 11.1 kJ/K per 2U server (measured experimentally)
    airflow_cfm_per_kw: float = 160.0      # Server fan airflow per kW IT load


@dataclass
class ZoneConfig:
    """Configuration for a thermal zone (section of datacenter)."""
    zone_id: str = "zone_a"
    racks: list[RackConfig] = field(default_factory=list)
    crac_units: list[CRACConfig] = field(default_factory=list)
    containment_type: str = "cold_aisle"    # "cold_aisle", "hot_aisle", "none"
    recirculation_factor: float = 0.08      # 0 = perfect containment, 0.3 = none
    air_volume_m3: float = 500.0            # Zone air volume
    envelope_r_kw: float = 0.02            # Thermal resistance to outside (K/W)
    initial_cold_aisle_temp_c: float = 20.0
    initial_humidity_rh: float = 0.45
    ashrae_class: str = "A2"


# ---------------------------------------------------------------------------
# Power distribution configuration
# ---------------------------------------------------------------------------
@dataclass
class UPSConfig:
    """Configuration for a UPS unit.

    Efficiency model (quadratic loss):
        η(x) = x / (x + c_0 + c_1·x + c_2·x²)
    where x = load_fraction (0 to 1).

    Default coefficients from APC White Paper 108 (modern double-conversion):
        c_0 = 0.013  (no-load: transformers, logic boards)
        c_1 = 0.006  (proportional: conduction losses)
        c_2 = 0.011  (square-law: I²R in conductors)
    """
    unit_id: str = "UPS-1"
    rated_capacity_kw: float = 500.0
    # Quadratic loss coefficients (fractions of rated capacity)
    loss_c0: float = 0.013                 # No-load losses
    loss_c1: float = 0.006                 # Proportional losses
    loss_c2: float = 0.011                 # Square-law losses
    # Battery
    battery_capacity_kwh: float = 8.3      # ~10 min at full load
    battery_discharge_efficiency: float = 0.90
    battery_aging_factor: float = 0.85     # End-of-life derating
    battery_temp_c: float = 25.0           # Battery room temperature
    # Recharge: ~10× discharge time
    recharge_rate_kw: float = 5.0          # Max recharge rate
    # Operating mode
    initial_mode: str = "double_conversion"  # "double_conversion", "line_interactive", "eco", "bypass"


@dataclass
class PDUConfig:
    """Configuration for a three-phase PDU.

    US standard: 208V L-L / 120V L-N, 24A per phase.
    Total nameplate: √3 × 208 × 24 ≈ 8,646 W.
    80% NEC continuous derating: 6,917 W.

    European: 400V L-L / 230V L-N, 32A per phase.
    Total nameplate: √3 × 400 × 32 ≈ 22,170 W.
    """
    pdu_id: str = "PDU-A1"
    voltage_ll_v: float = 208.0            # Line-to-line voltage
    max_current_per_phase_a: float = 24.0
    num_phases: int = 3
    breaker_rating_a: float = 20.0         # Per-branch circuit breaker
    num_outlets: int = 48
    efficiency: float = 0.98               # Transformer efficiency (2% losses)
    continuous_derating: float = 0.80      # NEC 80% rule for continuous loads


@dataclass
class GeneratorConfig:
    """Configuration for a diesel standby generator.

    Startup sequence (NFPA 110 Type 10):
        Start delay → cranking → warm-up → ready to accept load
        Total: 10-20 seconds
    """
    gen_id: str = "GEN-1"
    rated_capacity_kw: float = 750.0
    # Startup timing
    start_delay_s: float = 4.0             # Programmed delay before crank
    crank_time_s: float = 5.0              # Engine cranking duration
    warmup_time_s: float = 8.0             # Warm-up before load acceptance
    # Fuel
    fuel_tank_liters: float = 2000.0
    consumption_lph_full: float = 180.0    # Liters/hour at full load
    # Cool-down
    cooldown_time_s: float = 300.0         # 5-min unloaded cool-down


@dataclass
class ATSConfig:
    """Configuration for an Automatic Transfer Switch."""
    ats_id: str = "ATS-1"
    transfer_time_ms: float = 100.0        # Mechanical transfer time
    retransfer_delay_s: float = 300.0      # Wait before transferring back to utility


@dataclass
class PowerConfig:
    """Aggregated power infrastructure configuration."""
    ups_units: list[UPSConfig] = field(default_factory=list)
    pdus: list[PDUConfig] = field(default_factory=list)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    ats: ATSConfig = field(default_factory=ATSConfig)
    utility_voltage_v: float = 480.0       # Main utility feed voltage
    utility_available: bool = True


@dataclass
class DatacenterConfig:
    """Full datacenter configuration."""
    name: str = "DC-OPS Facility"
    zones: list[ZoneConfig] = field(default_factory=list)
    power: PowerConfig = field(default_factory=PowerConfig)
    outside_temp_c: float = 35.0
    outside_humidity_rh: float = 0.40
    lighting_w_per_m2: float = 10.0         # Typical 10 W/m²
    floor_area_m2: float = 500.0
    simulation_dt_s: float = 1.0            # Integration timestep
    # Kept for backward compatibility with Phase 1 thermal sim
    ups_loss_fraction: float = 0.05
    pdu_loss_fraction: float = 0.02


def make_default_datacenter_config() -> DatacenterConfig:
    """Create a realistic default datacenter: 2 zones, 10 racks each, 4 CRACs total.

    Power infrastructure:
      - 2× UPS (N+1 redundant, 500 kW each for 160 kW total IT load)
      - 20× PDUs (one per rack, US 3-phase 208V/24A)
      - 1× diesel generator (750 kW)
      - 1× ATS
    """
    zone_a_racks = [
        RackConfig(rack_id=f"A-{i:02d}", row="A", position=i, it_load_kw=8.0)
        for i in range(1, 11)
    ]
    zone_a_cracs = [
        CRACConfig(unit_id="CRAC-1"),
        CRACConfig(unit_id="CRAC-2"),
    ]

    zone_b_racks = [
        RackConfig(rack_id=f"B-{i:02d}", row="B", position=i, it_load_kw=8.0)
        for i in range(1, 11)
    ]
    zone_b_cracs = [
        CRACConfig(unit_id="CRAC-3"),
        CRACConfig(unit_id="CRAC-4"),
    ]

    # Power infrastructure
    ups_units = [
        UPSConfig(unit_id="UPS-1", rated_capacity_kw=500.0),
        UPSConfig(unit_id="UPS-2", rated_capacity_kw=500.0),
    ]
    pdus = [
        PDUConfig(pdu_id=f"PDU-{rack.rack_id}")
        for rack in zone_a_racks + zone_b_racks
    ]
    power = PowerConfig(
        ups_units=ups_units,
        pdus=pdus,
        generator=GeneratorConfig(gen_id="GEN-1", rated_capacity_kw=750.0),
        ats=ATSConfig(ats_id="ATS-1"),
    )

    return DatacenterConfig(
        name="DC-OPS Default Facility",
        zones=[
            ZoneConfig(
                zone_id="zone_a",
                racks=zone_a_racks,
                crac_units=zone_a_cracs,
                air_volume_m3=600.0,
            ),
            ZoneConfig(
                zone_id="zone_b",
                racks=zone_b_racks,
                crac_units=zone_b_cracs,
                air_volume_m3=600.0,
            ),
        ],
        power=power,
        outside_temp_c=35.0,
        outside_humidity_rh=0.40,
        floor_area_m2=1200.0,
    )
