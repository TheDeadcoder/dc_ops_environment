# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Renders simulation state into a text-based monitoring dashboard.

The dashboard mimics what a real datacenter operator would see on their
NOC (Network Operations Center) screens. It is the primary observation
for the LLM agent.
"""

from __future__ import annotations

from ..config import ASHRAE_CLASSES, m3s_to_cfm
from ..simulation.types import (
    CRACFaultType,
    CRACState,
    CRACStatus,
    DatacenterState,
    PowerState,
    ZoneState,
)


def render_dashboard(
    state: DatacenterState,
    *,
    alert: str = "",
    step: int = 0,
    max_steps: int = 15,
    scenario_type: str = "",
) -> str:
    """Render the full monitoring dashboard as a text string.

    Args:
        state: Current datacenter simulation state.
        alert: Active alert message to display prominently.
        step: Current step number in the episode.
        max_steps: Maximum steps in the episode.
        scenario_type: Type of scenario being run.

    Returns:
        Multi-line string formatted as a monitoring dashboard.
    """
    w = 68  # Inner width of the dashboard frame
    lines: list[str] = []

    def hline(char: str = "═") -> str:
        return f"╠{char * w}╣"

    def row(text: str) -> str:
        return f"║ {text:<{w - 2}} ║"

    # Header
    lines.append(f"╔{'═' * w}╗")
    title = "DC-OPS MONITORING DASHBOARD"
    lines.append(f"║{title:^{w}}║")
    sim_min = state.sim_time_s / 60.0
    status_line = f"Sim Time: {sim_min:.1f} min    Step: {step}/{max_steps}"
    if scenario_type:
        status_line += f"    [{scenario_type}]"
    lines.append(row(status_line))

    # Alert section
    if alert:
        lines.append(hline())
        # Split long alerts across lines
        alert_prefix = "!! ALERT: "
        remaining = w - 2 - len(alert_prefix)
        if len(alert) <= remaining:
            lines.append(row(f"{alert_prefix}{alert}"))
        else:
            lines.append(row(f"{alert_prefix}{alert[:remaining]}"))
            # Continuation lines
            for i in range(remaining, len(alert), w - 4):
                lines.append(row(f"  {alert[i:i + w - 4]}"))

    # Cooling Units
    lines.append(hline())
    lines.append(row("COOLING UNITS"))
    lines.append(row(f"{'Unit':<10} {'Status':<12} {'Setpoint':>8} {'Supply':>8} {'Fan%':>5} {'CFM':>7} {'kW':>6}"))
    lines.append(row("-" * (w - 2)))

    for zone in state.zones:
        for crac in zone.crac_units:
            lines.append(row(_format_crac_row(crac, state.outside_temp_c, zone.hot_aisle_temp_c)))

    # Zone Temperatures
    lines.append(hline())
    lines.append(row("ZONE TEMPERATURES"))
    lines.append(row(f"{'Zone':<8} {'Cold Aisle':>10} {'Hot Aisle':>10} {'Max Inlet':>10} {'IT Load':>8} {'Class':>6}"))
    lines.append(row("-" * (w - 2)))

    for zone in state.zones:
        lines.append(row(_format_zone_row(zone)))

    # Rack Detail (per zone, show max-temp racks)
    lines.append(hline())
    lines.append(row("RACK TEMPERATURES (top 5 hottest)"))
    lines.append(row(f"{'Rack':<8} {'Inlet':>8} {'Outlet':>8} {'Load kW':>8} {'CFM':>7}"))
    lines.append(row("-" * (w - 2)))

    # Collect all racks, sort by inlet temp descending
    all_racks = []
    for zone in state.zones:
        all_racks.extend(zone.racks)
    all_racks.sort(key=lambda r: r.inlet_temp_c, reverse=True)
    for rack in all_racks[:5]:
        cfm = m3s_to_cfm(rack.airflow_m3s)
        lines.append(row(
            f"{rack.rack_id:<8} {rack.inlet_temp_c:>7.1f}°C {rack.outlet_temp_c:>7.1f}°C "
            f"{rack.it_load_kw:>7.1f} {cfm:>7.0f}"
        ))

    # Power Section
    lines.append(hline())
    lines.append(row("POWER"))

    p_it = state.total_it_load_kw
    p_cooling = state.total_cooling_power_kw
    pue = state.pue

    lines.append(row(
        f"IT Load: {p_it:.1f} kW | Cooling: {p_cooling:.1f} kW | PUE: {pue:.2f}"
    ))

    if state.power is not None:
        lines.append(row(_format_power_section(state.power)))
        lines.append(row(_format_ups_summary(state.power)))
    else:
        lines.append(row("UPS: N/A | Generator: N/A"))

    # Environment
    lines.append(hline())
    lines.append(row("ENVIRONMENT"))
    lines.append(row(
        f"Outside: {state.outside_temp_c:.1f}°C | "
        f"Humidity: {state.outside_humidity_rh * 100:.0f}% RH"
    ))

    # Footer
    lines.append(f"╚{'═' * w}╝")

    return "\n".join(lines)


def _format_crac_row(crac: CRACState, outside_temp_c: float, hot_aisle_temp_c: float) -> str:
    """Format a single CRAC row for the dashboard."""
    # Status display
    if crac.status == CRACStatus.FAULT:
        fault_label = crac.fault_type.value.upper() if crac.fault_type != CRACFaultType.NONE else "FAULT"
        status_str = f"!! {fault_label}"
    elif crac.status == CRACStatus.MAINTENANCE:
        status_str = "MAINT"
    elif crac.status == CRACStatus.STANDBY:
        status_str = "STANDBY"
    else:
        status_str = "RUNNING"

    # Supply temp display
    if crac.status != CRACStatus.RUNNING:
        supply_str = "---"
    else:
        supply_str = f"{crac.supply_temp_c:.1f}°C"

    # CFM
    cfm = m3s_to_cfm(crac.current_airflow_m3s)

    # Power consumption
    q_cool = crac.compute_cooling_output_kw(hot_aisle_temp_c)
    p_kw = crac.compute_power_consumption_kw(q_cool, outside_temp_c)

    return (
        f"{crac.unit_id:<10} {status_str:<12} {crac.setpoint_c:>7.1f}°C "
        f"{supply_str:>8} {crac.fan_speed_pct:>5.0f} {cfm:>7.0f} {p_kw:>6.1f}"
    )


def _format_zone_row(zone: ZoneState) -> str:
    """Format a single zone row for the dashboard."""
    ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
    max_inlet = zone.max_inlet_temp_c

    # Mark if exceeding ASHRAE recommended
    inlet_marker = ""
    if ashrae and max_inlet > ashrae.recommended_max_c:
        inlet_marker = "*"
    if ashrae and max_inlet > ashrae.allowable_max_c:
        inlet_marker = "!!"

    return (
        f"{zone.zone_id:<8} {zone.cold_aisle_temp_c:>9.1f}°C "
        f"{zone.hot_aisle_temp_c:>9.1f}°C {max_inlet:>8.1f}°C{inlet_marker:<2}"
        f"{zone.total_it_load_kw:>7.1f} {zone.ashrae_class:>6}"
    )


def _format_power_section(power: PowerState) -> str:
    """Format power source status line."""
    parts: list[str] = []

    # Utility / generator status
    if power.utility_available:
        parts.append("Utility: NORMAL")
    else:
        parts.append("Utility: DOWN")

    from ..simulation.types import GeneratorState as GS
    gen = power.generator
    if gen.state == GS.OFF:
        parts.append("Gen: OFF")
    elif gen.state == GS.LOADED:
        fuel_hrs = gen.fuel_remaining_hours
        fuel_str = f"{fuel_hrs:.1f}h" if fuel_hrs < 100 else ">100h"
        parts.append(f"Gen: LOADED {gen.load_fraction * 100:.0f}% (fuel: {fuel_str})")
    elif gen.state in (GS.START_DELAY, GS.CRANKING, GS.WARMING):
        parts.append(f"Gen: STARTING ({gen.state.value})")
    elif gen.state == GS.READY:
        parts.append("Gen: READY")
    elif gen.state == GS.COOLDOWN:
        parts.append("Gen: COOLDOWN")

    # ATS position
    from ..simulation.types import ATSPosition
    ats = power.ats
    if ats.position == ATSPosition.UTILITY:
        parts.append("ATS: UTILITY")
    elif ats.position == ATSPosition.GENERATOR:
        parts.append("ATS: GENERATOR")
    elif ats.position == ATSPosition.TRANSFERRING:
        parts.append("ATS: TRANSFERRING")

    return " | ".join(parts)


def _format_ups_summary(power: PowerState) -> str:
    """Format UPS status summary line."""
    if not power.ups_units:
        return "UPS: N/A"

    parts: list[str] = []
    for ups in power.ups_units:
        soc_pct = ups.battery_soc * 100
        mode_str = ups.mode.value.upper().replace("_", " ")
        load_pct = ups.load_fraction * 100
        eta_pct = ups.efficiency * 100

        if ups.mode.value == "on_battery":
            time_str = ""
            if ups.battery_time_remaining_s < float("inf"):
                mins = ups.battery_time_remaining_s / 60.0
                time_str = f" {mins:.0f}min"
            parts.append(f"{ups.unit_id}: BATTERY {soc_pct:.0f}%{time_str}")
        elif ups.mode.value == "fault":
            parts.append(f"{ups.unit_id}: FAULT")
        else:
            parts.append(f"{ups.unit_id}: {mode_str} {load_pct:.0f}% η{eta_pct:.0f}%")

    return "UPS: " + " | ".join(parts)
