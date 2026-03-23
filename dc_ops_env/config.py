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
from pathlib import Path
from typing import Union


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


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).parent / "data" / "datacenter_configs"

# Built-in config names (resolved relative to this package)
BUILTIN_CONFIGS: dict[str, Path] = {
    "default": _CONFIG_DIR / "default.yaml",
    "small": _CONFIG_DIR / "small_facility.yaml",
    "large": _CONFIG_DIR / "large_facility.yaml",
}


def load_datacenter_config(source: Union[str, Path]) -> DatacenterConfig:
    """Load a DatacenterConfig from a YAML file or built-in name.

    Args:
        source: Either a built-in name ("default", "small", "large"),
                or a path to a YAML file.

    Returns:
        Fully constructed DatacenterConfig.

    Examples:
        config = load_datacenter_config("small")
        config = load_datacenter_config("/path/to/custom.yaml")
    """
    import yaml

    # Resolve source to a file path
    if isinstance(source, str) and source in BUILTIN_CONFIGS:
        path = BUILTIN_CONFIGS[source]
    else:
        path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _dict_to_datacenter_config(data)


def _dict_to_datacenter_config(data: dict) -> DatacenterConfig:
    """Convert a raw YAML dict into a DatacenterConfig."""
    zones = [_dict_to_zone_config(z) for z in data.get("zones", [])]
    power = _dict_to_power_config(data.get("power", {}))

    return DatacenterConfig(
        name=data.get("name", "DC-OPS Facility"),
        zones=zones,
        power=power,
        outside_temp_c=data.get("outside_temp_c", 35.0),
        outside_humidity_rh=data.get("outside_humidity_rh", 0.40),
        lighting_w_per_m2=data.get("lighting_w_per_m2", 10.0),
        floor_area_m2=data.get("floor_area_m2", 500.0),
        simulation_dt_s=data.get("simulation_dt_s", 1.0),
        ups_loss_fraction=data.get("ups_loss_fraction", 0.05),
        pdu_loss_fraction=data.get("pdu_loss_fraction", 0.02),
    )


def _dict_to_zone_config(data: dict) -> ZoneConfig:
    """Convert a raw dict into a ZoneConfig."""
    racks = [_dict_to_rack_config(r) for r in data.get("racks", [])]
    cracs = [_dict_to_crac_config(c) for c in data.get("crac_units", [])]

    return ZoneConfig(
        zone_id=data.get("zone_id", "zone_a"),
        racks=racks,
        crac_units=cracs,
        containment_type=data.get("containment_type", "cold_aisle"),
        recirculation_factor=data.get("recirculation_factor", 0.08),
        air_volume_m3=data.get("air_volume_m3", 500.0),
        envelope_r_kw=data.get("envelope_r_kw", 0.02),
        initial_cold_aisle_temp_c=data.get("initial_cold_aisle_temp_c", 20.0),
        initial_humidity_rh=data.get("initial_humidity_rh", 0.45),
        ashrae_class=data.get("ashrae_class", "A2"),
    )


def _dict_to_rack_config(data: dict) -> RackConfig:
    """Convert a raw dict into a RackConfig."""
    return RackConfig(
        rack_id=data.get("rack_id", "A-01"),
        row=data.get("row", "A"),
        position=data.get("position", 1),
        it_load_kw=data.get("it_load_kw", 8.0),
        num_servers_2u=data.get("num_servers_2u", 20),
        server_thermal_mass_jk=data.get("server_thermal_mass_jk", 11100.0),
        airflow_cfm_per_kw=data.get("airflow_cfm_per_kw", 160.0),
    )


def _dict_to_crac_config(data: dict) -> CRACConfig:
    """Convert a raw dict into a CRACConfig."""
    return CRACConfig(
        unit_id=data.get("unit_id", "CRAC-1"),
        rated_capacity_kw=data.get("rated_capacity_kw", 70.0),
        rated_return_temp_c=data.get("rated_return_temp_c", 24.0),
        capacity_slope_per_c=data.get("capacity_slope_per_c", 0.03),
        max_airflow_cfm=data.get("max_airflow_cfm", 12000.0),
        fan_rated_power_kw=data.get("fan_rated_power_kw", 5.0),
        cop_rated=data.get("cop_rated", 3.5),
        cop_degradation_per_c=data.get("cop_degradation_per_c", 0.04),
        initial_setpoint_c=data.get("initial_setpoint_c", 18.0),
        initial_fan_speed_pct=data.get("initial_fan_speed_pct", 100.0),
        supply_temp_lag_s=data.get("supply_temp_lag_s", 30.0),
    )


def _dict_to_power_config(data: dict) -> PowerConfig:
    """Convert a raw dict into a PowerConfig."""
    ups = [_dict_to_ups_config(u) for u in data.get("ups_units", [])]
    pdus = [_dict_to_pdu_config(p) for p in data.get("pdus", [])]
    gen_data = data.get("generator", {})
    ats_data = data.get("ats", {})

    return PowerConfig(
        ups_units=ups,
        pdus=pdus,
        generator=GeneratorConfig(
            gen_id=gen_data.get("gen_id", "GEN-1"),
            rated_capacity_kw=gen_data.get("rated_capacity_kw", 750.0),
            start_delay_s=gen_data.get("start_delay_s", 4.0),
            crank_time_s=gen_data.get("crank_time_s", 5.0),
            warmup_time_s=gen_data.get("warmup_time_s", 8.0),
            fuel_tank_liters=gen_data.get("fuel_tank_liters", 2000.0),
            consumption_lph_full=gen_data.get("consumption_lph_full", 180.0),
            cooldown_time_s=gen_data.get("cooldown_time_s", 300.0),
        ),
        ats=ATSConfig(
            ats_id=ats_data.get("ats_id", "ATS-1"),
            transfer_time_ms=ats_data.get("transfer_time_ms", 100.0),
            retransfer_delay_s=ats_data.get("retransfer_delay_s", 300.0),
        ),
        utility_voltage_v=data.get("utility_voltage_v", 480.0),
        utility_available=data.get("utility_available", True),
    )


def _dict_to_ups_config(data: dict) -> UPSConfig:
    """Convert a raw dict into a UPSConfig."""
    return UPSConfig(
        unit_id=data.get("unit_id", "UPS-1"),
        rated_capacity_kw=data.get("rated_capacity_kw", 500.0),
        loss_c0=data.get("loss_c0", 0.013),
        loss_c1=data.get("loss_c1", 0.006),
        loss_c2=data.get("loss_c2", 0.011),
        battery_capacity_kwh=data.get("battery_capacity_kwh", 8.3),
        battery_discharge_efficiency=data.get("battery_discharge_efficiency", 0.90),
        battery_aging_factor=data.get("battery_aging_factor", 0.85),
        battery_temp_c=data.get("battery_temp_c", 25.0),
        recharge_rate_kw=data.get("recharge_rate_kw", 5.0),
        initial_mode=data.get("initial_mode", "double_conversion"),
    )


def _dict_to_pdu_config(data: dict) -> PDUConfig:
    """Convert a raw dict into a PDUConfig."""
    return PDUConfig(
        pdu_id=data.get("pdu_id", "PDU-A1"),
        voltage_ll_v=data.get("voltage_ll_v", 208.0),
        max_current_per_phase_a=data.get("max_current_per_phase_a", 24.0),
        num_phases=data.get("num_phases", 3),
        breaker_rating_a=data.get("breaker_rating_a", 20.0),
        num_outlets=data.get("num_outlets", 48),
        efficiency=data.get("efficiency", 0.98),
        continuous_derating=data.get("continuous_derating", 0.80),
    )
