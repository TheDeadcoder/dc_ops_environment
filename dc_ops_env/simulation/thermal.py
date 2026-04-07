# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RC thermal network simulation for datacenter zones.

Physics model (lumped-capacitance, per zone):

    C_zone × dT_zone/dt = Q_IT - Q_cooling + Q_envelope + Q_internal

Where:
    C_zone  = C_air + C_equipment                        [J/K]
    Q_IT    = sum of rack IT loads × 1000                 [W]
    Q_cool  = sum of CRAC cooling outputs × 1000          [W]
    Q_env   = (T_outside - T_zone) / R_envelope           [W]
    Q_int   = UPS losses + PDU losses + lighting          [W]

Cold aisle temperature accounts for hot-air recirculation:
    T_cold_effective = (1-r) × T_supply_weighted + r × T_hot_aisle

where r is the recirculation factor (0 = perfect containment).

Hot aisle temperature from server energy balance:
    T_hot = T_cold + Q_IT / (m_dot_rack × c_p)

Integration: Forward Euler with configurable dt (default 1.0 s).
Target: <1 ms per step for a 20-rack, 4-CRAC datacenter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import (
    AIR_DENSITY_KG_M3,
    AIR_SPECIFIC_HEAT_J_KGK,
    ASHRAE_CLASSES,
    DatacenterConfig,
    RackConfig,
    CRACConfig,
    ZoneConfig,
    cfm_to_m3s,
    make_default_datacenter_config,
)
from .types import (
    CRACFaultType,
    CRACState,
    CRACStatus,
    DatacenterState,
    RackState,
    ZoneState,
)


@dataclass
class ThermalAlarm:
    """An active thermal alarm."""
    rack_id: str
    zone_id: str
    inlet_temp_c: float
    threshold_c: float
    severity: str  # "warning" (recommended exceeded) or "critical" (allowable exceeded)


@dataclass
class ThermalStepResult:
    """Result of a single simulation step."""
    state: DatacenterState
    alarms: list[ThermalAlarm] = field(default_factory=list)
    total_cooling_output_kw: float = 0.0
    total_cooling_power_kw: float = 0.0
    energy_consumed_kwh: float = 0.0  # Energy consumed in this step


class ThermalSimulation:
    """Multi-zone RC thermal network simulation.

    Owns the DatacenterState and advances it forward in time.
    Each call to step() integrates the thermal ODEs by dt seconds.
    """

    def __init__(self, config: DatacenterConfig | None = None):
        if config is None:
            config = make_default_datacenter_config()
        self._config = config
        self._state = self._build_initial_state(config)
        self._dt = config.simulation_dt_s

    @property
    def state(self) -> DatacenterState:
        return self._state

    @property
    def config(self) -> DatacenterConfig:
        return self._config

    @property
    def dt(self) -> float:
        return self._dt

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _build_initial_state(config: DatacenterConfig) -> DatacenterState:
        """Construct the initial DatacenterState from configuration."""
        zones: list[ZoneState] = []
        for zc in config.zones:
            racks = ThermalSimulation._build_racks(zc, zc.initial_cold_aisle_temp_c)
            cracs = ThermalSimulation._build_cracs(zc)
            zone = ZoneState(
                zone_id=zc.zone_id,
                cold_aisle_temp_c=zc.initial_cold_aisle_temp_c,
                hot_aisle_temp_c=zc.initial_cold_aisle_temp_c + 15.0,  # Initial estimate
                humidity_rh=zc.initial_humidity_rh,
                recirculation_factor=zc.recirculation_factor,
                racks=racks,
                crac_units=cracs,
                air_volume_m3=zc.air_volume_m3,
                envelope_r_kw=zc.envelope_r_kw,
                ashrae_class=zc.ashrae_class,
            )
            zones.append(zone)

        state = DatacenterState(
            zones=zones,
            outside_temp_c=config.outside_temp_c,
            outside_humidity_rh=config.outside_humidity_rh,
            lighting_power_kw=config.lighting_w_per_m2 * config.floor_area_m2 / 1000.0,
            ups_loss_fraction=config.ups_loss_fraction,
            pdu_loss_fraction=config.pdu_loss_fraction,
            sim_time_s=0.0,
        )

        # Run a few settling steps so initial temps are physically consistent
        sim = ThermalSimulation.__new__(ThermalSimulation)
        sim._state = state
        sim._config = make_default_datacenter_config()
        sim._dt = 1.0
        for _ in range(300):
            sim._integrate_step(1.0)

        return state

    @staticmethod
    def _build_racks(zone_config: ZoneConfig, initial_temp_c: float) -> list[RackState]:
        racks: list[RackState] = []
        for rc in zone_config.racks:
            airflow_cfm = rc.airflow_cfm_per_kw * rc.it_load_kw
            airflow_m3s = cfm_to_m3s(airflow_cfm)
            thermal_mass = rc.num_servers_2u * rc.server_thermal_mass_jk

            rack = RackState(
                rack_id=rc.rack_id,
                row=rc.row,
                position=rc.position,
                it_load_kw=rc.it_load_kw,
                inlet_temp_c=initial_temp_c,
                outlet_temp_c=initial_temp_c + 15.0,  # Will be corrected by settling
                airflow_m3s=airflow_m3s,
                thermal_mass_jk=thermal_mass,
            )
            racks.append(rack)
        return racks

    @staticmethod
    def _build_cracs(zone_config: ZoneConfig) -> list[CRACState]:
        cracs: list[CRACState] = []
        for cc in zone_config.crac_units:
            crac = CRACState(
                unit_id=cc.unit_id,
                setpoint_c=cc.initial_setpoint_c,
                supply_temp_c=cc.initial_setpoint_c,
                fan_speed_pct=cc.initial_fan_speed_pct,
                max_airflow_m3s=cfm_to_m3s(cc.max_airflow_cfm),
                rated_capacity_kw=cc.rated_capacity_kw,
                rated_return_temp_c=cc.rated_return_temp_c,
                capacity_slope_per_c=cc.capacity_slope_per_c,
                fan_rated_power_kw=cc.fan_rated_power_kw,
                cop_rated=cc.cop_rated,
                cop_degradation_per_c=cc.cop_degradation_per_c,
                supply_temp_lag_s=cc.supply_temp_lag_s,
            )
            cracs.append(crac)
        return cracs

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, dt: float | None = None) -> ThermalStepResult:
        """Advance the simulation by dt seconds.

        Returns a ThermalStepResult with updated state, alarms, and energy metrics.
        """
        if dt is None:
            dt = self._dt

        result = self._integrate_step(dt)
        self._state.sim_time_s += dt
        return result

    def step_n(self, n: int, dt: float | None = None) -> ThermalStepResult:
        """Advance simulation by n steps. Returns result of the last step."""
        result = ThermalStepResult(state=self._state)
        for _ in range(n):
            result = self.step(dt)
        return result

    def _integrate_step(self, dt: float) -> ThermalStepResult:
        """Core integration: one Forward Euler step across all zones.

        Physics model — **cold aisle energy balance** (not total-zone):

        The cold aisle is a mixing volume. Heat flows into/out of it:
          q_crac   = m_dot_crac × c_p × (T_supply − T_cold)   [cooling from CRACs]
          q_recirc = r × m_dot_crac × c_p × (T_hot − T_cold)  [recirculated hot air]
          q_env    = (T_outside − T_cold) / R_envelope          [building heat gain]
          q_int    = UPS losses + PDU losses + lighting          [internal gains]

        IT heat does NOT appear directly — servers move cold air to the hot
        aisle, raising T_hot.  IT heat affects the cold aisle only through
        recirculation (hot air leaking back) and indirectly via CRAC return
        temperature.

        Hot aisle temperature (algebraic, not ODE):
          T_hot = T_cold + Q_IT / (m_dot_rack × c_p)

        CRAC return air temperature accounts for bypass airflow:
          When CRAC airflow > rack airflow, excess cold air bypasses servers
          and returns directly to the CRAC at T_cold, lowering the effective
          return air temperature and thus CRAC cooling output.
          T_return = (1 − bypass) × T_hot + bypass × T_cold
        """
        state = self._state
        alarms: list[ThermalAlarm] = []
        total_cooling_output_kw = 0.0
        total_cooling_power_kw = 0.0
        total_power_kw = 0.0

        for zone in state.zones:
            # 1. Update CRAC supply temperatures (first-order lag toward setpoint)
            for crac in zone.crac_units:
                crac.update_supply_temp(dt)

            # 2. Airflow quantities
            q_it_w = zone.total_it_load_kw * 1000.0
            m_dot_rack = zone.total_rack_airflow_m3s * AIR_DENSITY_KG_M3   # kg/s
            m_dot_crac = zone.total_crac_airflow_m3s * AIR_DENSITY_KG_M3   # kg/s

            # Server temperature rise [°C]
            if m_dot_rack > 0:
                dt_server = q_it_w / (m_dot_rack * AIR_SPECIFIC_HEAT_J_KGK)
            else:
                dt_server = 50.0  # No airflow → extreme rise
            t_hot = zone.cold_aisle_temp_c + dt_server

            # 3. Bypass fraction: excess CRAC airflow that bypasses servers
            if m_dot_crac > 0 and m_dot_rack > 0:
                bypass_frac = max(0.0, 1.0 - m_dot_rack / m_dot_crac)
            else:
                bypass_frac = 0.0

            # CRAC return air temp (mixed hot exhaust + bypassed cold air)
            t_return = (1.0 - bypass_frac) * t_hot + bypass_frac * zone.cold_aisle_temp_c

            # 4. CRAC cooling output (based on bypass-corrected return temp)
            q_cooling_extracted_w = 0.0
            zone_cooling_power_kw = 0.0
            for crac in zone.crac_units:
                q_crac_kw = crac.compute_cooling_output_kw(t_return)
                q_cooling_extracted_w += q_crac_kw * 1000.0
                total_cooling_output_kw += q_crac_kw

                p_crac_kw = crac.compute_power_consumption_kw(q_crac_kw, state.outside_temp_c)
                zone_cooling_power_kw += p_crac_kw
                total_cooling_power_kw += p_crac_kw

            # 5. Cold aisle energy balance [all in Watts]

            # CRAC supply mixing: each CRAC injects air into the cold aisle.
            # Running CRACs inject air at their supply temp (near setpoint).
            # Compressor-faulted CRACs with fans running inject air at the
            # return air temp (air passes through the inactive coil unconditioned).
            q_crac_mixing_w = 0.0
            for crac in zone.crac_units:
                crac_flow = crac.current_airflow_m3s * AIR_DENSITY_KG_M3
                if crac_flow <= 0:
                    continue
                if crac.fault_type in (CRACFaultType.COMPRESSOR, CRACFaultType.REFRIGERANT_LEAK):
                    effective_supply = t_return  # No cooling — just recirculating
                else:
                    effective_supply = crac.supply_temp_c
                q_crac_mixing_w += crac_flow * AIR_SPECIFIC_HEAT_J_KGK * (
                    effective_supply - zone.cold_aisle_temp_c
                )

            # Hot air entering cold aisle from two mechanisms:
            #
            # (a) Containment recirculation: fraction r of air leaks through
            #     containment gaps regardless of CRAC flow balance.
            #     Uses max(m_dot_rack, m_dot_crac) — recirculation is driven
            #     by pressure differentials from whichever airflow is dominant.
            #     When CRACs are off, server fans still drive leakage.
            r = zone.recirculation_factor
            m_dot_dominant = max(m_dot_rack, m_dot_crac)
            q_recirc_w = r * m_dot_dominant * AIR_SPECIFIC_HEAT_J_KGK * dt_server

            # (b) Natural return: when CRAC airflow < rack airflow, servers
            #     exhaust more hot air than CRACs can capture. The uncaptured
            #     fraction returns to the cold aisle via natural convection.
            #     When CRACs are completely off, ALL server exhaust returns
            #     (= Q_IT returns to cold aisle as heat).
            if m_dot_rack > 0 and m_dot_crac < m_dot_rack:
                natural_return_frac = 1.0 - m_dot_crac / m_dot_rack
                q_natural_return_w = (
                    natural_return_frac * m_dot_rack * AIR_SPECIFIC_HEAT_J_KGK * dt_server
                )
            else:
                q_natural_return_w = 0.0

            # Envelope heat gain
            if zone.envelope_r_kw > 0:
                q_envelope_w = (state.outside_temp_c - zone.cold_aisle_temp_c) / zone.envelope_r_kw
            else:
                q_envelope_w = 0.0

            # Internal gains (UPS/PDU losses + lighting)
            q_ups_w = zone.total_it_load_kw * state.ups_loss_fraction * 1000.0
            q_pdu_w = zone.total_it_load_kw * state.pdu_loss_fraction * 1000.0
            num_zones = len(state.zones) if state.zones else 1
            q_lighting_w = state.lighting_power_kw * 1000.0 / num_zones
            q_internal_w = q_ups_w + q_pdu_w + q_lighting_w

            # 6. Net heat into cold aisle [W]
            q_net_w = (
                q_crac_mixing_w + q_recirc_w + q_natural_return_w
                + q_envelope_w + q_internal_w
            )

            # 7. Forward Euler integration
            c_total = zone.compute_thermal_capacitance_jk()
            if c_total > 0:
                dT = q_net_w * dt / c_total
                zone.cold_aisle_temp_c += dT

            # 8. Update hot aisle (algebraic: T_hot = T_cold + server ΔT)
            if m_dot_rack > 0:
                zone.hot_aisle_temp_c = (
                    zone.cold_aisle_temp_c
                    + q_it_w / (m_dot_rack * AIR_SPECIFIC_HEAT_J_KGK)
                )
            else:
                zone.hot_aisle_temp_c = zone.cold_aisle_temp_c + 50.0

            # 9. Update individual rack inlet/outlet temperatures
            for rack in zone.racks:
                rack.inlet_temp_c = zone.cold_aisle_temp_c
                rack.outlet_temp_c = rack.compute_outlet_temp()

            # 10. Check ASHRAE alarms
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if ashrae:
                for rack in zone.racks:
                    if rack.inlet_temp_c > ashrae.allowable_max_c:
                        alarms.append(ThermalAlarm(
                            rack_id=rack.rack_id,
                            zone_id=zone.zone_id,
                            inlet_temp_c=rack.inlet_temp_c,
                            threshold_c=ashrae.allowable_max_c,
                            severity="critical",
                        ))
                    elif rack.inlet_temp_c > ashrae.recommended_max_c:
                        alarms.append(ThermalAlarm(
                            rack_id=rack.rack_id,
                            zone_id=zone.zone_id,
                            inlet_temp_c=rack.inlet_temp_c,
                            threshold_c=ashrae.recommended_max_c,
                            severity="warning",
                        ))

            total_power_kw += zone.total_it_load_kw

        # Energy consumed in this step [kWh]
        total_facility_kw = total_power_kw + total_cooling_power_kw + (
            total_power_kw * (state.ups_loss_fraction + state.pdu_loss_fraction)
            + state.lighting_power_kw
        )
        energy_kwh = total_facility_kw * dt / 3600.0

        return ThermalStepResult(
            state=state,
            alarms=alarms,
            total_cooling_output_kw=total_cooling_output_kw,
            total_cooling_power_kw=total_cooling_power_kw,
            energy_consumed_kwh=energy_kwh,
        )

    @staticmethod
    def _compute_weighted_supply_temp(zone: ZoneState) -> float | None:
        """Flow-weighted average of CRAC supply temperatures.

        T_supply_weighted = Σ(T_supply_i × m_dot_i) / Σ(m_dot_i)

        Returns None if no CRACs are producing airflow.
        """
        total_flow = 0.0
        weighted_temp = 0.0
        for crac in zone.crac_units:
            flow = crac.current_airflow_m3s
            if flow > 0:
                weighted_temp += crac.supply_temp_c * flow
                total_flow += flow

        if total_flow <= 0:
            return None
        return weighted_temp / total_flow

    # ------------------------------------------------------------------
    # Mutation helpers (used by action parser in later phases)
    # ------------------------------------------------------------------

    def set_crac_setpoint(self, unit_id: str, setpoint_c: float) -> bool:
        """Adjust a CRAC unit's supply air temperature setpoint. Returns success."""
        crac = self._find_crac(unit_id)
        if crac is None:
            return False
        crac.setpoint_c = setpoint_c
        return True

    def set_crac_fan_speed(self, unit_id: str, speed_pct: float) -> bool:
        """Set CRAC fan speed (0-100%). Returns success."""
        crac = self._find_crac(unit_id)
        if crac is None:
            return False
        crac.fan_speed_pct = max(0.0, min(100.0, speed_pct))
        return True

    def set_crac_status(self, unit_id: str, status: CRACStatus) -> bool:
        """Change CRAC operating status. Returns success."""
        crac = self._find_crac(unit_id)
        if crac is None:
            return False
        crac.status = status
        return True

    def inject_crac_fault(
        self, unit_id: str, fault_type: CRACFaultType
    ) -> bool:
        """Inject a fault into a CRAC unit. Returns success."""
        crac = self._find_crac(unit_id)
        if crac is None:
            return False
        crac.status = CRACStatus.FAULT
        crac.fault_type = fault_type
        return True

    def clear_crac_fault(self, unit_id: str) -> bool:
        """Clear a CRAC fault and return to running. Returns success."""
        crac = self._find_crac(unit_id)
        if crac is None:
            return False
        crac.status = CRACStatus.RUNNING
        crac.fault_type = CRACFaultType.NONE
        return True

    def set_rack_load(self, rack_id: str, load_kw: float) -> bool:
        """Change a rack's IT load. Returns success."""
        rack = self._find_rack(rack_id)
        if rack is None:
            return False
        rack.it_load_kw = max(0.0, load_kw)
        # Update airflow proportionally (servers spin fans with load)
        from ..config import RackConfig
        default_cfm_per_kw = RackConfig().airflow_cfm_per_kw
        rack.airflow_m3s = cfm_to_m3s(default_cfm_per_kw * rack.it_load_kw)
        return True

    def set_outside_temp(self, temp_c: float) -> None:
        """Set outside temperature."""
        self._state.outside_temp_c = temp_c

    def _find_crac(self, unit_id: str) -> CRACState | None:
        target = unit_id.lower()
        for zone in self._state.zones:
            for crac in zone.crac_units:
                if crac.unit_id.lower() == target:
                    return crac
        return None

    def _find_rack(self, rack_id: str) -> RackState | None:
        target = rack_id.lower()
        for zone in self._state.zones:
            for rack in zone.racks:
                if rack.rack_id.lower() == target:
                    return rack
        return None

    def find_zone_for_crac(self, unit_id: str) -> ZoneState | None:
        """Find the zone containing a given CRAC unit."""
        for zone in self._state.zones:
            for crac in zone.crac_units:
                if crac.unit_id == unit_id:
                    return zone
        return None

    def find_zone_for_rack(self, rack_id: str) -> ZoneState | None:
        """Find the zone containing a given rack."""
        for zone in self._state.zones:
            for rack in zone.racks:
                if rack.rack_id == rack_id:
                    return zone
        return None
