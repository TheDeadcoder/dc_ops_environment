# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic action parser for operator commands.

Parses natural-language commands from the LLM agent into simulation mutations.
Uses regex matching for speed and testability — no LLM-in-the-loop.

Command format: command_name [target] [value]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..simulation.thermal import ThermalSimulation
from ..simulation.power import PowerSimulation
from ..simulation.types import (
    CRACFaultType,
    CRACStatus,
    UPSMode,
)


@dataclass
class CommandResult:
    """Result of parsing and executing a command."""
    success: bool
    message: str
    command_name: str = ""
    target: str = ""


# ---------------------------------------------------------------------------
# Available commands for the agent
# ---------------------------------------------------------------------------
AVAILABLE_ACTIONS: list[str] = [
    "diagnose <unit_id>           — Inspect a CRAC/UPS/PDU for faults",
    "adjust_setpoint <crac_id> <temp_c> — Change CRAC supply air setpoint",
    "set_fan_speed <crac_id> <pct>  — Set CRAC fan speed (0-100%)",
    "set_rack_load <rack_id> <kw>   — Adjust rack IT load (migrate workload)",
    "start_crac <crac_id>          — Start a standby CRAC unit",
    "stop_crac <crac_id>           — Put a CRAC into standby",
    "start_generator               — Manually start the diesel generator",
    "stop_generator                — Initiate generator cooldown",
    "set_ups_mode <ups_id> <mode>  — Set UPS mode (eco/double_conversion/bypass)",
    "refuel_generator [liters]     — Refuel (default: full tank)",
    "acknowledge_alarm             — Acknowledge current alert",
    "check_status                  — Request full status report",
    "escalate                      — Escalate to senior engineer",
    "wait                          — Take no action this step",
]


def parse_command(
    command: str,
    thermal_sim: ThermalSimulation,
    power_sim: PowerSimulation | None = None,
) -> CommandResult:
    """Parse and execute an operator command.

    Args:
        command: Raw command string from the agent.
        thermal_sim: Thermal simulation to mutate.
        power_sim: Power simulation to mutate (optional).

    Returns:
        CommandResult with success status and feedback message.
    """
    cmd = command.strip()
    if not cmd:
        return CommandResult(False, "Empty command. Use 'check_status' or see available actions.")

    # Try each handler in order
    for pattern, handler in _COMMAND_TABLE:
        match = re.match(pattern, cmd, re.IGNORECASE)
        if match:
            return handler(match, thermal_sim, power_sim)

    return CommandResult(
        False,
        f"Unknown command: '{cmd}'. Use 'check_status' for available actions.",
        command_name="unknown",
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def _handle_diagnose(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    """Inspect a unit for faults and report status."""
    target = match.group(1)

    # Check CRACs
    for zone in thermal.state.zones:
        for crac in zone.crac_units:
            if crac.unit_id.lower() == target.lower():
                lines = [
                    f"=== Diagnostic Report: {crac.unit_id} ===",
                    f"Status: {crac.status.value}",
                    f"Fault: {crac.fault_type.value}",
                    f"Setpoint: {crac.setpoint_c:.1f}°C",
                    f"Supply Temp: {crac.supply_temp_c:.1f}°C",
                    f"Fan Speed: {crac.fan_speed_pct:.0f}%",
                    f"Airflow: {crac.current_airflow_m3s:.3f} m³/s",
                ]
                if crac.fault_type != CRACFaultType.NONE:
                    lines.append(f">> FAULT DETECTED: {crac.fault_type.value}")
                    lines.append(">> Recommended: repair or replace component")
                else:
                    lines.append(">> No faults detected. Unit operating normally.")
                return CommandResult(True, "\n".join(lines), "diagnose", target)

    # Check UPS
    if power:
        for ups in power.state.ups_units:
            if ups.unit_id.lower() == target.lower():
                lines = [
                    f"=== Diagnostic Report: {ups.unit_id} ===",
                    f"Mode: {ups.mode.value}",
                    f"Load: {ups.load_fraction * 100:.1f}%",
                    f"Efficiency: {ups.efficiency * 100:.1f}%",
                    f"Battery SOC: {ups.battery_soc * 100:.0f}%",
                    f"Output: {ups.output_power_kw:.1f} kW",
                    f"Losses: {ups.heat_output_kw:.1f} kW",
                ]
                return CommandResult(True, "\n".join(lines), "diagnose", target)

    return CommandResult(False, f"Unit '{target}' not found.", "diagnose", target)


def _handle_adjust_setpoint(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    target = match.group(1)
    try:
        value = float(match.group(2))
    except (ValueError, IndexError):
        return CommandResult(False, "Invalid temperature value.", "adjust_setpoint", target)

    if value < 10.0 or value > 35.0:
        return CommandResult(
            False,
            f"Setpoint {value:.1f}°C out of safe range (10-35°C).",
            "adjust_setpoint", target,
        )

    if thermal.set_crac_setpoint(target, value):
        return CommandResult(
            True,
            f"Setpoint for {target} adjusted to {value:.1f}°C. "
            "Supply temp will converge over ~30 seconds.",
            "adjust_setpoint", target,
        )
    return CommandResult(False, f"CRAC '{target}' not found.", "adjust_setpoint", target)


def _handle_set_fan_speed(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    target = match.group(1)
    try:
        value = float(match.group(2))
    except (ValueError, IndexError):
        return CommandResult(False, "Invalid fan speed value.", "set_fan_speed", target)

    if value < 0 or value > 100:
        return CommandResult(
            False, f"Fan speed {value:.0f}% out of range (0-100%).",
            "set_fan_speed", target,
        )

    if thermal.set_crac_fan_speed(target, value):
        return CommandResult(
            True,
            f"Fan speed for {target} set to {value:.0f}%.",
            "set_fan_speed", target,
        )
    return CommandResult(False, f"CRAC '{target}' not found.", "set_fan_speed", target)


def _handle_set_rack_load(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    target = match.group(1)
    try:
        value = float(match.group(2))
    except (ValueError, IndexError):
        return CommandResult(False, "Invalid load value.", "set_rack_load", target)

    if value < 0 or value > 30:
        return CommandResult(
            False, f"Load {value:.1f} kW out of range (0-30 kW).",
            "set_rack_load", target,
        )

    if thermal.set_rack_load(target, value):
        return CommandResult(
            True,
            f"IT load for rack {target} set to {value:.1f} kW.",
            "set_rack_load", target,
        )
    return CommandResult(False, f"Rack '{target}' not found.", "set_rack_load", target)


def _handle_start_crac(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    target = match.group(1)
    for zone in thermal.state.zones:
        for crac in zone.crac_units:
            if crac.unit_id.lower() == target.lower():
                if crac.status == CRACStatus.RUNNING:
                    return CommandResult(False, f"{target} is already running.", "start_crac", target)
                if crac.fault_type != CRACFaultType.NONE:
                    return CommandResult(
                        False,
                        f"{target} has an active fault ({crac.fault_type.value}). "
                        "Clear the fault before starting.",
                        "start_crac", target,
                    )
                crac.status = CRACStatus.RUNNING
                return CommandResult(True, f"{target} started.", "start_crac", target)
    return CommandResult(False, f"CRAC '{target}' not found.", "start_crac", target)


def _handle_stop_crac(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    target = match.group(1)
    for zone in thermal.state.zones:
        for crac in zone.crac_units:
            if crac.unit_id.lower() == target.lower():
                if crac.status == CRACStatus.STANDBY:
                    return CommandResult(False, f"{target} is already in standby.", "stop_crac", target)
                crac.status = CRACStatus.STANDBY
                return CommandResult(True, f"{target} placed in standby.", "stop_crac", target)
    return CommandResult(False, f"CRAC '{target}' not found.", "stop_crac", target)


def _handle_start_generator(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    if power is None:
        return CommandResult(False, "Power subsystem not available.", "start_generator")
    power.start_generator()
    return CommandResult(True, "Generator start sequence initiated.", "start_generator")


def _handle_stop_generator(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    if power is None:
        return CommandResult(False, "Power subsystem not available.", "stop_generator")
    power.stop_generator()
    return CommandResult(True, "Generator cooldown initiated.", "stop_generator")


def _handle_set_ups_mode(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    if power is None:
        return CommandResult(False, "Power subsystem not available.", "set_ups_mode")
    target = match.group(1)
    mode_str = match.group(2).lower().strip()

    mode_map = {
        "double_conversion": UPSMode.DOUBLE_CONVERSION,
        "eco": UPSMode.ECO,
        "line_interactive": UPSMode.LINE_INTERACTIVE,
        "bypass": UPSMode.BYPASS,
    }
    mode = mode_map.get(mode_str)
    if mode is None:
        valid = ", ".join(mode_map.keys())
        return CommandResult(False, f"Unknown UPS mode '{mode_str}'. Valid: {valid}", "set_ups_mode", target)

    if power.set_ups_mode(target, mode):
        return CommandResult(True, f"{target} set to {mode_str} mode.", "set_ups_mode", target)
    return CommandResult(False, f"UPS '{target}' not found.", "set_ups_mode", target)


def _handle_refuel_generator(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    if power is None:
        return CommandResult(False, "Power subsystem not available.", "refuel_generator")
    liters_str = match.group(1) if match.group(1) else None
    if liters_str:
        try:
            liters = float(liters_str)
        except ValueError:
            return CommandResult(False, "Invalid liters value.", "refuel_generator")
        power.refuel_generator(liters)
        return CommandResult(True, f"Added {liters:.0f}L to generator.", "refuel_generator")
    else:
        power.refuel_generator()
        return CommandResult(True, "Generator refueled to full tank.", "refuel_generator")


def _handle_acknowledge_alarm(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    return CommandResult(True, "Alarm acknowledged.", "acknowledge_alarm")


def _handle_check_status(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    return CommandResult(True, "Full status displayed in dashboard.", "check_status")


def _handle_escalate(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    return CommandResult(
        True,
        "Incident escalated to senior datacenter engineer. Episode ending.",
        "escalate",
    )


def _handle_wait(
    match: re.Match, thermal: ThermalSimulation, power: PowerSimulation | None
) -> CommandResult:
    return CommandResult(True, "Waiting. No action taken.", "wait")


# ---------------------------------------------------------------------------
# Command table: (regex_pattern, handler_function)
# Order matters — first match wins.
# ---------------------------------------------------------------------------
_COMMAND_TABLE: list[tuple[re.Pattern | str, Any]] = [
    (r"diagnose\s+(\S+)", _handle_diagnose),
    (r"adjust_setpoint\s+(\S+)\s+([\d.]+)", _handle_adjust_setpoint),
    (r"set_fan_speed\s+(\S+)\s+([\d.]+)", _handle_set_fan_speed),
    (r"(?:set_rack_load|migrate_workload)\s+(\S+)\s+([\d.]+)", _handle_set_rack_load),
    (r"start_crac\s+(\S+)", _handle_start_crac),
    (r"stop_crac\s+(\S+)", _handle_stop_crac),
    (r"start_generator\b", _handle_start_generator),
    (r"stop_generator\b", _handle_stop_generator),
    (r"set_ups_mode\s+(\S+)\s+(\S+)", _handle_set_ups_mode),
    (r"refuel_generator\s*([\d.]*)", _handle_refuel_generator),
    (r"acknowledge_alarm\b", _handle_acknowledge_alarm),
    (r"check_status\b", _handle_check_status),
    (r"escalate\b", _handle_escalate),
    (r"wait\b", _handle_wait),
]
