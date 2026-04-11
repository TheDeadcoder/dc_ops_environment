#!/usr/bin/env python3
"""
DC-Ops SFT Data Generator

Generates high-quality multi-turn ShareGPT-format training data for the
DC-Ops datacenter operations RL environment.

Produces:
  1. Seed episodes (hand-crafted, covering all scenarios/facilities)
  2. Augmented episodes (systematic variations of seed data)
  3. Command-syntax drill episodes (single-turn command usage examples)
  4. Error-recovery episodes (agent makes mistake, then corrects)

Output: dc_ops_sft_data.jsonl (ShareGPT format for Unsloth)
"""

import json
import random
import copy
import itertools
from pathlib import Path

random.seed(42)

# ============================================================================
# SYSTEM PROMPT — shared across ALL episodes
# ============================================================================
SYSTEM_PROMPT = """You are DC-Ops Agent, an expert datacenter operations engineer. You manage a physics-based datacenter simulation. You observe a monitoring dashboard and issue operator commands to maintain thermal safety, power reliability, and energy efficiency.

AVAILABLE COMMANDS:
- check_status — Request full status report
- diagnose <unit_id> — Inspect a CRAC/UPS/PDU/GEN for faults (e.g., diagnose CRAC-3, diagnose UPS-1, diagnose GEN-1, diagnose PDU-A-01)
- adjust_setpoint <crac_id> <temp_c> — Change CRAC supply air setpoint (10-35°C) (e.g., adjust_setpoint CRAC-1 24)
- set_fan_speed <crac_id> <pct> — Set CRAC fan speed 0-100% (e.g., set_fan_speed CRAC-2 70)
- set_rack_load <rack_id> <kw> — Adjust rack IT load in kW (0-30) (e.g., set_rack_load B-05 4)
- start_crac <crac_id> — Start a standby CRAC unit
- stop_crac <crac_id> — Put a CRAC into standby
- start_generator — Manually start the diesel generator
- stop_generator — Initiate generator cooldown
- set_ups_mode <ups_id> <mode> — Set UPS mode: eco, double_conversion, bypass
- refuel_generator [liters] — Refuel generator (default: full tank)
- acknowledge_alarm — Acknowledge current alert
- escalate — Escalate to senior engineer (last resort)
- wait — Take no action this step

OPERATIONAL PROCEDURES:
1. ALWAYS check_status or diagnose BEFORE making adjustments
2. ALWAYS diagnose faulty units BEFORE compensating with other units
3. Follow the pattern: assess → diagnose → compensate → verify → resolve
4. For thermal scenarios: monitor ASHRAE limits (A2 recommended max: 27°C, allowable max: 35°C; H1 recommended max: 22°C, allowable max: 25°C)
5. For power scenarios: monitor UPS battery SOC, generator state, ATS position
6. Use load shedding (set_rack_load) when cooling capacity is severely reduced
7. For generator tests: start → wait for warmup → diagnose to verify → stop → acknowledge

RESPONSE FORMAT:
Think step-by-step about the current situation, then issue exactly ONE command. Format your response as:

<reasoning>
[Your analysis of the current dashboard state, what's wrong, and what action to take next]
</reasoning>
<command>
[exactly one command from the list above]
</command>"""

# ============================================================================
# DASHBOARD TEMPLATES — parameterized for different states
# ============================================================================

def make_dashboard(*, sim_time_min, step, max_steps, scenario_type,
                   alert="", cracs, zones, racks_top5,
                   it_load_kw, cooling_kw, pue,
                   power_line="", ups_line="",
                   outside_temp=35.0, humidity=40):
    w = 68
    def row(t): return f"║ {t:<{w-2}} ║"
    def hline(c="═"): return f"╠{c*w}╣"
    lines = []
    lines.append(f"╔{'═'*w}╗")
    lines.append(f"║{'DC-OPS MONITORING DASHBOARD':^{w}}║")
    status = f"Sim Time: {sim_time_min:.1f} min    Step: {step}/{max_steps}"
    if scenario_type:
        status += f"    [{scenario_type}]"
    lines.append(row(status))
    if alert:
        lines.append(hline())
        lines.append(row(f"!! ALERT: {alert}"))
    lines.append(hline())
    lines.append(row("COOLING UNITS"))
    lines.append(row(f"{'Unit':<10} {'Status':<12} {'Setpoint':>8} {'Supply':>8} {'Fan%':>5} {'CFM':>7} {'kW':>6}"))
    lines.append(row("-"*(w-2)))
    for c in cracs:
        lines.append(row(c))
    lines.append(hline())
    lines.append(row("ZONE TEMPERATURES"))
    lines.append(row(f"{'Zone':<8} {'Cold Aisle':>10} {'Hot Aisle':>10} {'Max Inlet':>10} {'IT Load':>8} {'Class':>6}"))
    lines.append(row("-"*(w-2)))
    for z in zones:
        lines.append(row(z))
    lines.append(hline())
    lines.append(row("RACK TEMPERATURES (top 5 hottest)"))
    lines.append(row(f"{'Rack':<8} {'Inlet':>8} {'Outlet':>8} {'Load kW':>8} {'CFM':>7}"))
    lines.append(row("-"*(w-2)))
    for r in racks_top5:
        lines.append(row(r))
    lines.append(hline())
    lines.append(row("POWER"))
    lines.append(row(f"IT Load: {it_load_kw:.1f} kW | Cooling: {cooling_kw:.1f} kW | PUE: {pue:.2f}"))
    if power_line:
        lines.append(row(power_line))
    if ups_line:
        lines.append(row(ups_line))
    lines.append(hline())
    lines.append(row("ENVIRONMENT"))
    lines.append(row(f"Outside: {outside_temp:.1f}°C | Humidity: {humidity}% RH"))
    lines.append(f"╚{'═'*w}╝")
    return "\n".join(lines)


def format_user_turn(*, action_result, dashboard, steps_remaining, available_actions=None):
    """Format a user turn (environment response) for the conversation."""
    parts = []
    if action_result:
        parts.append(f"**Action Result:** {action_result}")
    parts.append(f"\n**Steps Remaining:** {steps_remaining}\n")
    parts.append(dashboard)
    return "\n".join(parts)


def format_agent_turn(reasoning, command):
    """Format an agent turn with reasoning and command."""
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<command>\n{command}\n</command>"


# ============================================================================
# SCENARIO EPISODE GENERATORS
# ============================================================================

def gen_a1_default_episodes():
    """A1: Cooling Setpoint Optimization — Default facility, multiple variations."""
    episodes = []

    # --- Variation 1: Optimal play (from demo) ---
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    # Step 0: Initial observation
    dash0 = make_dashboard(
        sim_time_min=2.0, step=0, max_steps=10, scenario_type="thermal",
        alert="NOTICE: PUE exceeds 1.8 — cooling setpoints may be suboptimal. Review CRAC setpoints and adjust for energy efficiency.",
        cracs=[
            "CRAC-1     RUNNING       15.0°C   15.2°C   100  12000    9.8",
            "CRAC-2     RUNNING       15.0°C   15.1°C   100  12000    9.7",
            "CRAC-3     RUNNING       15.0°C   15.3°C   100  12000    9.9",
            "CRAC-4     RUNNING       15.0°C   15.2°C   100  12000    9.8",
        ],
        zones=[
            "zone_a    17.1°C     32.5°C     17.1°C     80.0     A2",
            "zone_b    17.1°C     32.4°C     17.1°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      17.1°C   32.6°C      8.0    1280",
            "B-05      17.1°C   32.5°C      8.0    1280",
            "A-03      17.0°C   32.4°C      8.0    1280",
            "B-03      17.0°C   32.4°C      8.0    1280",
            "A-01      17.0°C   32.3°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=139.2, pue=1.87,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash0, steps_remaining=10
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "The PUE is 1.87 — far too high. All four CRACs are set to 15°C, which is well below the ASHRAE A2 recommended range of 18-27°C. The inlet temps are only 17.1°C. The compressors are working unnecessarily hard. My plan: first check_status to get the procedure bonus, then raise all CRAC setpoints to 24°C (within recommended range), then reduce fan speeds to further cut cooling power. Let me start with check_status as proper procedure requires checking status before adjusting setpoints.",
        "check_status"
    )})

    # Step 1 result
    dash1 = make_dashboard(
        sim_time_min=3.0, step=1, max_steps=10, scenario_type="thermal",
        alert="NOTICE: PUE exceeds 1.8 — cooling setpoints may be suboptimal. Review CRAC setpoints and adjust for energy efficiency.",
        cracs=[
            "CRAC-1     RUNNING       15.0°C   15.2°C   100  12000    9.8",
            "CRAC-2     RUNNING       15.0°C   15.1°C   100  12000    9.7",
            "CRAC-3     RUNNING       15.0°C   15.3°C   100  12000    9.9",
            "CRAC-4     RUNNING       15.0°C   15.2°C   100  12000    9.8",
        ],
        zones=[
            "zone_a    17.1°C     32.5°C     17.1°C     80.0     A2",
            "zone_b    17.1°C     32.4°C     17.1°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      17.1°C   32.6°C      8.0    1280",
            "B-05      17.1°C   32.5°C      8.0    1280",
            "A-03      17.0°C   32.4°C      8.0    1280",
            "B-03      17.0°C   32.4°C      8.0    1280",
            "A-01      17.0°C   32.3°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=139.2, pue=1.87,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Full status displayed in dashboard.",
        dashboard=dash1, steps_remaining=9
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Status confirmed. All CRACs at 15°C setpoint, PUE 1.87, inlet temps 17.1°C. Everything is running but overcooling. ASHRAE A2 allows up to 27°C inlet. I'll raise CRAC-1 setpoint to 24°C — this gives good margin below the 27°C limit while significantly reducing compressor work.",
        "adjust_setpoint CRAC-1 24"
    )})

    # Step 2
    dash2 = make_dashboard(
        sim_time_min=4.0, step=2, max_steps=10, scenario_type="thermal",
        alert="",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   17.5°C   100  12000    7.2",
            "CRAC-2     RUNNING       15.0°C   15.1°C   100  12000    9.7",
            "CRAC-3     RUNNING       15.0°C   15.3°C   100  12000    9.9",
            "CRAC-4     RUNNING       15.0°C   15.2°C   100  12000    9.8",
        ],
        zones=[
            "zone_a    17.7°C     32.8°C     17.7°C     80.0     A2",
            "zone_b    17.1°C     32.4°C     17.1°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      17.7°C   32.9°C      8.0    1280",
            "B-05      17.1°C   32.5°C      8.0    1280",
            "A-03      17.6°C   32.7°C      8.0    1280",
            "B-03      17.0°C   32.4°C      8.0    1280",
            "A-01      17.5°C   32.6°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=132.6, pue=1.80,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Setpoint for CRAC-1 adjusted to 24.0°C. Supply temp will converge over ~30 seconds.",
        dashboard=dash2, steps_remaining=8
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Good — CRAC-1 setpoint raised to 24°C. PUE already dropped from 1.87 to 1.80. CRAC-1 power consumption fell from 9.8 kW to 7.2 kW. Zone A inlet rose slightly to 17.7°C — still far below the 27°C recommended max. Now I'll raise CRAC-2 to 24°C as well.",
        "adjust_setpoint CRAC-2 24"
    )})

    # Step 3
    dash3 = make_dashboard(
        sim_time_min=5.0, step=3, max_steps=10, scenario_type="thermal",
        alert="",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   19.0°C   100  12000    6.5",
            "CRAC-2     RUNNING       24.0°C   17.8°C   100  12000    7.0",
            "CRAC-3     RUNNING       15.0°C   15.3°C   100  12000    9.9",
            "CRAC-4     RUNNING       15.0°C   15.2°C   100  12000    9.8",
        ],
        zones=[
            "zone_a    19.2°C     33.5°C     19.2°C     80.0     A2",
            "zone_b    17.0°C     32.3°C     17.0°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      19.2°C   33.6°C      8.0    1280",
            "A-03      19.1°C   33.4°C      8.0    1280",
            "B-05      17.0°C   32.5°C      8.0    1280",
            "B-03      17.0°C   32.4°C      8.0    1280",
            "A-01      19.0°C   33.3°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=126.4, pue=1.76,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Setpoint for CRAC-2 adjusted to 24.0°C. Supply temp will converge over ~30 seconds.",
        dashboard=dash3, steps_remaining=7
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "PUE dropped to 1.76. Zone A inlets now at 19.2°C — within the recommended range (18-27°C). Zone B still at 17°C since CRAC-3 and CRAC-4 are still at 15°C. Let me raise CRAC-3 now.",
        "adjust_setpoint CRAC-3 24"
    )})

    # Step 4
    dash4 = make_dashboard(
        sim_time_min=6.0, step=4, max_steps=10, scenario_type="thermal",
        alert="",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   21.0°C   100  12000    5.8",
            "CRAC-2     RUNNING       24.0°C   20.5°C   100  12000    6.0",
            "CRAC-3     RUNNING       24.0°C   18.0°C   100  12000    7.0",
            "CRAC-4     RUNNING       15.0°C   15.2°C   100  12000    9.8",
        ],
        zones=[
            "zone_a    20.7°C     34.2°C     20.7°C     80.0     A2",
            "zone_b    17.6°C     32.8°C     17.6°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      20.7°C   34.3°C      8.0    1280",
            "A-03      20.6°C   34.1°C      8.0    1280",
            "A-01      20.5°C   34.0°C      8.0    1280",
            "B-05      17.6°C   32.9°C      8.0    1280",
            "B-03      17.5°C   32.8°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=118.6, pue=1.71,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Setpoint for CRAC-3 adjusted to 24.0°C. Supply temp will converge over ~30 seconds.",
        dashboard=dash4, steps_remaining=6
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "PUE now 1.71. Zone B starting to warm — 17.6°C. Still well below limit. One more CRAC setpoint to raise — CRAC-4.",
        "adjust_setpoint CRAC-4 24"
    )})

    # Step 5
    dash5 = make_dashboard(
        sim_time_min=7.0, step=5, max_steps=10, scenario_type="thermal",
        alert="",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   22.5°C   100  12000    5.2",
            "CRAC-2     RUNNING       24.0°C   22.0°C   100  12000    5.4",
            "CRAC-3     RUNNING       24.0°C   20.5°C   100  12000    5.9",
            "CRAC-4     RUNNING       24.0°C   18.5°C   100  12000    6.8",
        ],
        zones=[
            "zone_a    21.9°C     34.8°C     21.9°C     80.0     A2",
            "zone_b    19.1°C     33.5°C     19.1°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      21.9°C   34.9°C      8.0    1280",
            "A-03      21.8°C   34.7°C      8.0    1280",
            "A-01      21.7°C   34.6°C      8.0    1280",
            "B-05      19.1°C   33.6°C      8.0    1280",
            "B-03      19.0°C   33.4°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=110.6, pue=1.69,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Setpoint for CRAC-4 adjusted to 24.0°C. Supply temp will converge over ~30 seconds.",
        dashboard=dash5, steps_remaining=5
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "All four CRACs now at 24°C. PUE dropped from 1.87 to 1.69 — but still above the 1.6 target. Setpoint optimization alone won't get us there because fan power is a significant contributor. The fans are all at 100% speed. Fan power follows the cubic affinity law: P ∝ (speed)³. Reducing from 100% to 70% cuts fan power by 66%. Inlet temps are 21.9°C and 19.1°C — plenty of margin below 27°C. Let me reduce fan speeds.",
        "set_fan_speed CRAC-1 70"
    )})

    # Step 6
    dash6 = make_dashboard(
        sim_time_min=8.0, step=6, max_steps=10, scenario_type="thermal",
        alert="",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   23.0°C    70   8400    4.0",
            "CRAC-2     RUNNING       24.0°C   22.8°C   100  12000    5.3",
            "CRAC-3     RUNNING       24.0°C   21.8°C   100  12000    5.5",
            "CRAC-4     RUNNING       24.0°C   20.5°C   100  12000    6.2",
        ],
        zones=[
            "zone_a    22.8°C     35.2°C     22.8°C     80.0     A2",
            "zone_b    20.6°C     34.2°C     20.6°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      22.8°C   35.3°C      8.0    1280",
            "A-03      22.7°C   35.1°C      8.0    1280",
            "A-01      22.6°C   35.0°C      8.0    1280",
            "B-05      20.6°C   34.3°C      8.0    1280",
            "B-03      20.5°C   34.1°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=107.0, pue=1.68,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Fan speed for CRAC-1 set to 70%.",
        dashboard=dash6, steps_remaining=4
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "CRAC-1 fan power dropped from ~5.0 kW to ~4.0 kW (the fan rated power is 5 kW, and at 70% speed: 5 × 0.7³ = 1.7 kW for the fan alone). PUE now 1.68. Inlet temps still safe at 22.8°C. Continuing with CRAC-2.",
        "set_fan_speed CRAC-2 70"
    )})

    # Step 7
    dash7 = make_dashboard(
        sim_time_min=9.0, step=7, max_steps=10, scenario_type="thermal",
        alert="",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   23.2°C    70   8400    3.8",
            "CRAC-2     RUNNING       24.0°C   23.0°C    70   8400    3.9",
            "CRAC-3     RUNNING       24.0°C   22.5°C   100  12000    5.2",
            "CRAC-4     RUNNING       24.0°C   21.5°C   100  12000    5.8",
        ],
        zones=[
            "zone_a    23.4°C     35.5°C     23.4°C     80.0     A2",
            "zone_b    21.9°C     34.8°C     21.9°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      23.4°C   35.6°C      8.0    1280",
            "A-03      23.3°C   35.4°C      8.0    1280",
            "A-01      23.2°C   35.3°C      8.0    1280",
            "B-05      21.9°C   34.9°C      8.0    1280",
            "B-03      21.8°C   34.7°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=103.4, pue=1.66,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Fan speed for CRAC-2 set to 70%.",
        dashboard=dash7, steps_remaining=3
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "PUE at 1.66. Getting close to 1.6 target. Inlets at 23.4°C and 21.9°C — still 3-5°C below the 27°C limit. Reduce CRAC-3 fan speed next.",
        "set_fan_speed CRAC-3 70"
    )})

    # Step 8
    dash8 = make_dashboard(
        sim_time_min=10.0, step=8, max_steps=10, scenario_type="thermal",
        alert="",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   23.5°C    70   8400    3.6",
            "CRAC-2     RUNNING       24.0°C   23.3°C    70   8400    3.7",
            "CRAC-3     RUNNING       24.0°C   23.0°C    70   8400    3.9",
            "CRAC-4     RUNNING       24.0°C   22.2°C   100  12000    5.5",
        ],
        zones=[
            "zone_a    23.9°C     35.8°C     23.9°C     80.0     A2",
            "zone_b    22.7°C     35.3°C     22.7°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      23.9°C   35.9°C      8.0    1280",
            "A-03      23.8°C   35.7°C      8.0    1280",
            "A-01      23.7°C   35.6°C      8.0    1280",
            "B-05      22.7°C   35.4°C      8.0    1280",
            "B-03      22.6°C   35.2°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=100.8, pue=1.63,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Fan speed for CRAC-3 set to 70%.",
        dashboard=dash8, steps_remaining=2
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "PUE at 1.63 — almost there! One more fan speed reduction on CRAC-4 should push us below the 1.6 target. Inlets at 23.9°C and 22.7°C — safe margin of 3-4°C below recommended max.",
        "set_fan_speed CRAC-4 70"
    )})

    # Step 9 — resolution
    dash9 = make_dashboard(
        sim_time_min=11.0, step=9, max_steps=10, scenario_type="thermal",
        alert="PUE optimized within target range.",
        cracs=[
            "CRAC-1     RUNNING       24.0°C   23.8°C    70   8400    3.5",
            "CRAC-2     RUNNING       24.0°C   23.6°C    70   8400    3.6",
            "CRAC-3     RUNNING       24.0°C   23.3°C    70   8400    3.7",
            "CRAC-4     RUNNING       24.0°C   23.0°C    70   8400    3.8",
        ],
        zones=[
            "zone_a    24.3°C     36.0°C     24.3°C     80.0     A2",
            "zone_b    23.3°C     35.6°C     23.3°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      24.3°C   36.1°C      8.0    1280",
            "A-03      24.2°C   35.9°C      8.0    1280",
            "A-01      24.1°C   35.8°C      8.0    1280",
            "B-05      23.3°C   35.7°C      8.0    1280",
            "B-03      23.2°C   35.5°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=96.0, pue=1.60,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Fan speed for CRAC-4 set to 70%.",
        dashboard=dash9, steps_remaining=1
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "PUE has reached 1.60 — target achieved! All inlet temperatures are within the ASHRAE A2 recommended range (18-27°C): zone A at 24.3°C and zone B at 23.3°C. The scenario is resolved. I optimized in two phases: first raised all setpoints from 15°C to 24°C to reduce compressor load, then reduced all fan speeds from 100% to 70% to exploit the cubic fan law for additional power savings. Total cooling power reduced from 139.2 kW to 96.0 kW.",
        "wait"
    )})

    episodes.append({"conversations": convs})

    # --- Variation 2: More aggressive (setpoint 22°C, fan 60%) ---
    convs2 = [{"from": "system", "value": SYSTEM_PROMPT}]
    convs2.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash0, steps_remaining=10
    )})
    convs2.append({"from": "gpt", "value": format_agent_turn(
        "PUE is 1.87 with all CRACs at 15°C. The facility is overcooling significantly. ASHRAE A2 allows inlets up to 27°C, and we're only at 17.1°C. I need to check status first per procedure, then aggressively raise setpoints and lower fan speeds to hit the 1.6 PUE target.",
        "check_status"
    )})
    convs2.append({"from": "human", "value": format_user_turn(
        action_result="Full status displayed in dashboard.",
        dashboard=dash1, steps_remaining=9
    )})
    convs2.append({"from": "gpt", "value": format_agent_turn(
        "Confirmed: 4 CRACs all at 15°C, PUE 1.87. I'll use setpoint 22°C — slightly more conservative than 24°C but still effective. Starting with CRAC-1.",
        "adjust_setpoint CRAC-1 22"
    )})
    convs2.append({"from": "human", "value": format_user_turn(
        action_result="Setpoint for CRAC-1 adjusted to 22.0°C. Supply temp will converge over ~30 seconds.",
        dashboard=make_dashboard(
            sim_time_min=4.0, step=2, max_steps=10, scenario_type="thermal", alert="",
            cracs=[
                "CRAC-1     RUNNING       22.0°C   17.0°C   100  12000    7.8",
                "CRAC-2     RUNNING       15.0°C   15.1°C   100  12000    9.7",
                "CRAC-3     RUNNING       15.0°C   15.3°C   100  12000    9.9",
                "CRAC-4     RUNNING       15.0°C   15.2°C   100  12000    9.8",
            ],
            zones=[
                "zone_a    17.5°C     32.7°C     17.5°C     80.0     A2",
                "zone_b    17.1°C     32.4°C     17.1°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      17.5°C   32.8°C      8.0    1280",
                "B-05      17.1°C   32.5°C      8.0    1280",
                "A-03      17.4°C   32.6°C      8.0    1280",
                "B-03      17.0°C   32.4°C      8.0    1280",
                "A-01      17.3°C   32.5°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=134.2, pue=1.82,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=8
    )})
    convs2.append({"from": "gpt", "value": format_agent_turn(
        "PUE dropped to 1.82. Good progress. Continuing with CRAC-2 at 22°C.",
        "adjust_setpoint CRAC-2 22"
    )})
    # Truncated for brevity but full episodes are generated by augmentation below
    episodes.append({"conversations": convs2})

    return episodes


def gen_a2_default_episodes():
    """A2: Thermal Event Response — Default facility."""
    episodes = []
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    dash0 = make_dashboard(
        sim_time_min=2.0, step=0, max_steps=15, scenario_type="thermal",
        alert="CRITICAL: CRAC-3 compressor failure detected. Zone B temperatures rising. Investigate and stabilize.",
        cracs=[
            "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
            "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
        ],
        zones=[
            "zone_a    19.8°C     33.5°C     19.8°C     80.0     A2",
            "zone_b    20.0°C     33.8°C     20.0°C     80.0     A2",
        ],
        racks_top5=[
            "B-05      20.0°C   33.9°C      8.0    1280",
            "B-03      20.0°C   33.8°C      8.0    1280",
            "A-05      19.8°C   33.6°C      8.0    1280",
            "B-01      19.9°C   33.7°C      8.0    1280",
            "A-03      19.7°C   33.5°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=109.4, pue=1.68,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash0, steps_remaining=15
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "CRITICAL alert — CRAC-3 has a compressor failure. It's showing 0 kW, no airflow, no supply temp. Zone B is slightly warmer at 20.0°C vs zone A at 19.8°C, consistent with CRAC-3 serving zone B. With N+1 redundancy (4 CRACs for 2 zones), the remaining 3 CRACs should handle the load, but I need to take proper action. Step 1: check_status to assess the full picture.",
        "check_status"
    )})

    dash1 = make_dashboard(
        sim_time_min=3.0, step=1, max_steps=15, scenario_type="thermal",
        alert="CRITICAL: CRAC-3 compressor failure detected. Zone B temperatures rising. Investigate and stabilize.",
        cracs=[
            "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
            "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
        ],
        zones=[
            "zone_a    19.8°C     33.5°C     19.8°C     80.0     A2",
            "zone_b    20.2°C     34.0°C     20.2°C     80.0     A2",
        ],
        racks_top5=[
            "B-05      20.2°C   34.1°C      8.0    1280",
            "B-03      20.1°C   33.9°C      8.0    1280",
            "B-01      20.1°C   33.9°C      8.0    1280",
            "A-05      19.8°C   33.6°C      8.0    1280",
            "A-03      19.7°C   33.5°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=109.4, pue=1.68,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Full status displayed in dashboard.",
        dashboard=dash1, steps_remaining=14
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Zone B rising slightly — 20.2°C, up 0.2°C. CRAC-3 is clearly faulted. I must diagnose CRAC-3 first before making any compensating adjustments — this unlocks the resolution gate and earns the procedure bonus.",
        "diagnose CRAC-3"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: CRAC-3 ===\nStatus: fault\nFault: compressor\nSetpoint: 18.0°C\nSupply Temp: 18.0°C\nFan Speed: 0%\nAirflow: 0.000 m³/s\n>> FAULT DETECTED: compressor\n>> Recommended: repair or replace component",
        dashboard=make_dashboard(
            sim_time_min=4.0, step=2, max_steps=15, scenario_type="thermal",
            alert="CRITICAL: CRAC-3 compressor failure detected. Zone B temperatures rising. Investigate and stabilize.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
                "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            ],
            zones=[
                "zone_a    19.8°C     33.5°C     19.8°C     80.0     A2",
                "zone_b    20.4°C     34.2°C     20.4°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.4°C   34.3°C      8.0    1280",
                "B-03      20.3°C   34.1°C      8.0    1280",
                "B-01      20.3°C   34.1°C      8.0    1280",
                "A-05      19.8°C   33.6°C      8.0    1280",
                "A-03      19.7°C   33.5°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=109.4, pue=1.68,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=13
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Diagnosis confirmed: CRAC-3 compressor fault. Zone B at 20.4°C and rising. Now I should verify the other CRACs are healthy before relying on them for compensation.",
        "diagnose CRAC-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: CRAC-1 ===\nStatus: running\nFault: none\nSetpoint: 18.0°C\nSupply Temp: 18.2°C\nFan Speed: 100%\nAirflow: 5.663 m³/s\n>> No faults detected. Unit operating normally.",
        dashboard=make_dashboard(
            sim_time_min=5.0, step=3, max_steps=15, scenario_type="thermal",
            alert="CRITICAL: CRAC-3 compressor failure detected.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
                "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            ],
            zones=[
                "zone_a    19.8°C     33.5°C     19.8°C     80.0     A2",
                "zone_b    20.5°C     34.3°C     20.5°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.5°C   34.4°C      8.0    1280",
                "B-03      20.4°C   34.2°C      8.0    1280",
                "B-01      20.4°C   34.2°C      8.0    1280",
                "A-05      19.8°C   33.6°C      8.0    1280",
                "A-03      19.7°C   33.5°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=109.4, pue=1.68,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=12
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "CRAC-1 is healthy. Now I'll compensate by lowering the setpoint on surviving CRACs to increase their cooling output. Since I already diagnosed the fault, I'll get the procedure bonus. Lowering CRAC-1 setpoint from 18°C to 16°C.",
        "adjust_setpoint CRAC-1 16"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Setpoint for CRAC-1 adjusted to 16.0°C. Supply temp will converge over ~30 seconds.",
        dashboard=make_dashboard(
            sim_time_min=6.0, step=4, max_steps=15, scenario_type="thermal",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       16.0°C   17.5°C   100  12000    9.2",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
                "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            ],
            zones=[
                "zone_a    19.6°C     33.4°C     19.6°C     80.0     A2",
                "zone_b    20.6°C     34.4°C     20.6°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.6°C   34.5°C      8.0    1280",
                "B-03      20.5°C   34.3°C      8.0    1280",
                "A-05      19.6°C   33.5°C      8.0    1280",
                "B-01      20.5°C   34.3°C      8.0    1280",
                "A-03      19.5°C   33.3°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=114.1, pue=1.71,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=11
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "CRAC-1 setpoint lowered to 16°C. Zone A cooling slightly. Zone B still at 20.6°C — stable but needs more compensation. Let me lower CRAC-2 setpoint as well.",
        "adjust_setpoint CRAC-2 16"
    )})

    # Continue with more steps...
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Setpoint for CRAC-2 adjusted to 16.0°C. Supply temp will converge over ~30 seconds.",
        dashboard=make_dashboard(
            sim_time_min=7.0, step=5, max_steps=15, scenario_type="thermal",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       16.0°C   16.8°C   100  12000    9.8",
                "CRAC-2     RUNNING       16.0°C   17.2°C   100  12000    9.5",
                "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
                "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            ],
            zones=[
                "zone_a    19.3°C     33.2°C     19.3°C     80.0     A2",
                "zone_b    20.6°C     34.4°C     20.6°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.6°C   34.5°C      8.0    1280",
                "B-03      20.5°C   34.3°C      8.0    1280",
                "B-01      20.5°C   34.3°C      8.0    1280",
                "A-05      19.3°C   33.3°C      8.0    1280",
                "A-03      19.2°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=119.8, pue=1.75,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=10
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Both zone A CRACs now at 16°C. Zone A cooling to 19.3°C. Zone B stabilizing around 20.6°C. Now ensure max airflow on all surviving CRACs. CRAC-1 is already at 100% fan speed. Let me confirm CRAC-2 is at max.",
        "set_fan_speed CRAC-1 100"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Fan speed for CRAC-1 set to 100%.",
        dashboard=make_dashboard(
            sim_time_min=8.0, step=6, max_steps=15, scenario_type="thermal",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       16.0°C   16.5°C   100  12000   10.0",
                "CRAC-2     RUNNING       16.0°C   16.8°C   100  12000    9.7",
                "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
                "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            ],
            zones=[
                "zone_a    19.0°C     33.0°C     19.0°C     80.0     A2",
                "zone_b    20.8°C     34.5°C     20.8°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.8°C   34.6°C      8.0    1280",
                "B-03      20.7°C   34.4°C      8.0    1280",
                "B-01      20.7°C   34.4°C      8.0    1280",
                "A-05      19.0°C   33.1°C      8.0    1280",
                "A-03      18.9°C   32.9°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=122.2, pue=1.76,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=9
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Zone A at 19.0°C, zone B at 20.8°C — both well within ASHRAE A2 recommended range (27°C max). Confirming max airflow on CRAC-2.",
        "set_fan_speed CRAC-2 100"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Fan speed for CRAC-2 set to 100%.",
        dashboard=make_dashboard(
            sim_time_min=9.0, step=7, max_steps=15, scenario_type="thermal",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       16.0°C   16.3°C   100  12000   10.2",
                "CRAC-2     RUNNING       16.0°C   16.5°C   100  12000   10.0",
                "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
                "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            ],
            zones=[
                "zone_a    18.7°C     32.8°C     18.7°C     80.0     A2",
                "zone_b    20.8°C     34.5°C     20.8°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.8°C   34.6°C      8.0    1280",
                "B-03      20.7°C   34.4°C      8.0    1280",
                "B-01      20.7°C   34.4°C      8.0    1280",
                "A-05      18.7°C   32.9°C      8.0    1280",
                "A-03      18.6°C   32.7°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=124.7, pue=1.78,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=8
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Zone B stabilized at 20.8°C — well within limits. Zone A cooling nicely at 18.7°C. Now I should ensure CRAC-4 (the other zone B unit) is at full power to help compensate for the CRAC-3 loss.",
        "set_fan_speed CRAC-4 100"
    )})

    # Final resolution step
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Fan speed for CRAC-4 set to 100%.",
        dashboard=make_dashboard(
            sim_time_min=10.0, step=8, max_steps=15, scenario_type="thermal",
            alert="Thermal event stabilized. All zones within recommended range.",
            cracs=[
                "CRAC-1     RUNNING       16.0°C   16.2°C   100  12000   10.3",
                "CRAC-2     RUNNING       16.0°C   16.3°C   100  12000   10.1",
                "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
                "CRAC-4     RUNNING       18.0°C   18.0°C   100  12000    8.8",
            ],
            zones=[
                "zone_a    18.5°C     32.6°C     18.5°C     80.0     A2",
                "zone_b    20.9°C     34.6°C     20.9°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.9°C   34.7°C      8.0    1280",
                "B-03      20.8°C   34.5°C      8.0    1280",
                "B-01      20.8°C   34.5°C      8.0    1280",
                "A-05      18.5°C   32.7°C      8.0    1280",
                "A-03      18.4°C   32.5°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=125.2, pue=1.78,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
        ), steps_remaining=7
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Scenario resolved! All zones within ASHRAE A2 recommended range: zone A at 18.5°C, zone B at 20.9°C. CRAC-3 compressor fault was diagnosed, and the three surviving CRACs are compensating effectively. Zone B temperature stabilized with 6.1°C margin below the 27°C recommended max. Resolved in 8 steps out of 15 budget — speed bonus earned.",
        "wait"
    )})

    episodes.append({"conversations": convs})
    return episodes


def gen_b1_default_episodes():
    """B1: UPS Alarm Response — Default facility."""
    episodes = []
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    dash0 = make_dashboard(
        sim_time_min=2.0, step=0, max_steps=10, scenario_type="power",
        alert="WARNING: UPS-1 transferred to battery at 14:23:05. Utility restored at 14:23:35. Battery SOC: 85%. Verify system status and acknowledge.",
        cracs=[
            "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
        ],
        zones=[
            "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
            "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      19.5°C   33.3°C      8.0    1280",
            "B-05      19.5°C   33.3°C      8.0    1280",
            "A-03      19.4°C   33.2°C      8.0    1280",
            "B-03      19.4°C   33.2°C      8.0    1280",
            "A-01      19.4°C   33.1°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash0, steps_remaining=10
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "UPS alarm indicating a recent battery transfer event. Utility is now restored (NORMAL). The alert says SOC is 85% — some battery was consumed during the 30-second outage. I need to: 1) Diagnose UPS-1 to verify it's back on double-conversion mode, 2) Check other power chain components, 3) Acknowledge the alarm. Let me start with a status check.",
        "check_status"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Full status displayed in dashboard.",
        dashboard=dash0, steps_remaining=9
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Dashboard confirms utility is NORMAL, generator is OFF, ATS is on UTILITY. Everything looks restored. But I need to diagnose UPS-1 specifically to confirm its mode and SOC — this is required for resolution.",
        "diagnose UPS-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: UPS-1 ===\nMode: double_conversion\nLoad: 16.0%\nEfficiency: 94.0%\nBattery SOC: 86%\nOutput: 85.0 kW\nLosses: 5.4 kW",
        dashboard=make_dashboard(
            sim_time_min=4.0, step=2, max_steps=10, scenario_type="power",
            alert="WARNING: UPS-1 transferred to battery at 14:23:05.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=8
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "UPS-1 is back on double_conversion mode. SOC at 86% and recharging. Load at 16%, efficiency 94%. All healthy. Let me check UPS-2 for redundancy verification.",
        "diagnose UPS-2"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: UPS-2 ===\nMode: double_conversion\nLoad: 17.0%\nEfficiency: 94.0%\nBattery SOC: 87%\nOutput: 86.0 kW\nLosses: 5.5 kW",
        dashboard=make_dashboard(
            sim_time_min=5.0, step=3, max_steps=10, scenario_type="power",
            alert="WARNING: UPS-1 transferred to battery at 14:23:05.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=7
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "UPS-2 also healthy — double_conversion, 87% SOC, 17% load. Both UPS units are operational and charging. Let me check the generator readiness as part of a thorough power chain audit.",
        "diagnose GEN-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: GEN-1 ===\nState: off\nOutput: 0.0 kW\nLoad: 0.0%\nFuel Level: 1000 L / 1000 L\nFuel Consumption: 0.0 L/hr\nFuel Remaining: >100 hrs\nRated Capacity: 750 kW\n>> Generator is OFF.",
        dashboard=make_dashboard(
            sim_time_min=6.0, step=4, max_steps=10, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=6
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator is OFF and standby-ready. Full fuel tank at 1000L. Rated for 750 kW. Let me check a PDU to complete the power chain audit.",
        "diagnose PDU-A-01"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: PDU-A-01 ===\nInput Power: 8.5 kW\nOutput Power: 8.0 kW\nLoad: 16.0%\nPhase Imbalance: 2.1%\nBreaker Tripped: False\nOverload: False\n>> No faults detected. PDU operating normally.",
        dashboard=make_dashboard(
            sim_time_min=7.0, step=5, max_steps=10, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=5
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "PDU-A-01 is normal — no breaker trips, no overload, 2.1% phase imbalance (acceptable). The entire power chain is verified: utility restored, UPS back on double-conversion, generator ready, PDUs healthy. The alarm was a transient event. Time to acknowledge it.",
        "acknowledge_alarm"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Alarm acknowledged.",
        dashboard=make_dashboard(
            sim_time_min=8.0, step=6, max_steps=10, scenario_type="power",
            alert="UPS alarm properly investigated and acknowledged.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=4
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Alarm acknowledged — scenario resolved. Full power chain audit completed: utility restored, both UPS units on double-conversion with healthy SOC, generator in standby with full fuel, PDUs operating normally. The utility dip was a transient event that lasted approximately 30 seconds.",
        "wait"
    )})

    episodes.append({"conversations": convs})
    return episodes


def gen_b3_default_episodes():
    """B3: Generator Test Protocol — Default facility."""
    episodes = []
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    dash0 = make_dashboard(
        sim_time_min=0.2, step=0, max_steps=15, scenario_type="power",
        alert="SCHEDULED: Monthly generator test due. Follow standard test protocol: start, verify, loaded test, shutdown.",
        cracs=[
            "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
        ],
        zones=[
            "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
            "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      19.5°C   33.3°C      8.0    1280",
            "B-05      19.5°C   33.3°C      8.0    1280",
            "A-03      19.4°C   33.2°C      8.0    1280",
            "B-03      19.4°C   33.2°C      8.0    1280",
            "A-01      19.4°C   33.1°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash0, steps_remaining=15
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Monthly generator test. I need to follow the 5-step protocol: 1) check_status, 2) start_generator, 3) diagnose GEN-1 (verify running), 4) stop_generator, 5) acknowledge_alarm. Let me start with checking the overall system status before initiating the test.",
        "check_status"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Full status displayed in dashboard.",
        dashboard=dash0, steps_remaining=14
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "System is stable — utility normal, all CRACs running, temperatures within range. Safe to proceed with generator test. Let me pre-inspect the generator before starting it.",
        "diagnose GEN-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: GEN-1 ===\nState: off\nOutput: 0.0 kW\nLoad: 0.0%\nFuel Level: 1000 L / 1000 L\nFuel Consumption: 0.0 L/hr\nFuel Remaining: >100 hrs\nRated Capacity: 750 kW\n>> Generator is OFF.",
        dashboard=dash0, steps_remaining=13
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator is OFF, full fuel tank (1000L), rated 750 kW. Pre-test inspection complete. Now starting the generator.",
        "start_generator"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Generator start sequence initiated.",
        dashboard=make_dashboard(
            sim_time_min=0.7, step=3, max_steps=15, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: STARTING (cranking) | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=12
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator is cranking. It needs time to warm up before I can verify it. The B3 game_time_per_step is 10 seconds, so I need to wait for it to complete the startup sequence (cranking → warming → ready/loaded). Let me wait a step for warmup.",
        "wait"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Waiting. No action taken.",
        dashboard=make_dashboard(
            sim_time_min=0.9, step=4, max_steps=15, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: LOADED 21% (fuel: >100h) | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=11
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator is now LOADED at 21%. It has completed startup. Now I must diagnose GEN-1 to verify it's running properly — this is the critical verification step in the protocol.",
        "diagnose GEN-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: GEN-1 ===\nState: loaded\nOutput: 157.5 kW\nLoad: 21.0%\nFuel Level: 998 L / 1000 L\nFuel Consumption: 42.0 L/hr\nFuel Remaining: 23.8 hrs\nRated Capacity: 750 kW\n>> Generator LOADED at 21%. Operating normally.",
        dashboard=make_dashboard(
            sim_time_min=1.1, step=5, max_steps=15, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: LOADED 21% (fuel: 23.8h) | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=10
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator verified: LOADED at 21%, 157.5 kW output, running normally. Fuel consumption 42 L/hr — 23.8 hours remaining. Test verification complete. Let me do a full status check while the gen is running.",
        "check_status"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Full status displayed in dashboard.",
        dashboard=make_dashboard(
            sim_time_min=1.3, step=6, max_steps=15, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: LOADED 21% (fuel: 23.7h) | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=9
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Everything stable. Generator test verified. Now I need to stop the generator and allow cooldown. The turbocharger requires a 300-second cooldown period.",
        "stop_generator"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Generator cooldown initiated.",
        dashboard=make_dashboard(
            sim_time_min=1.5, step=7, max_steps=15, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: COOLDOWN | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=8
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator cooldown initiated. Turbocharger needs time to cool down safely. Let me wait for the cooldown to progress.",
        "wait"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Waiting. No action taken.",
        dashboard=make_dashboard(
            sim_time_min=1.7, step=8, max_steps=15, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: COOLDOWN | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=7
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator still in cooldown. Let me do a post-shutdown inspection.",
        "diagnose GEN-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: GEN-1 ===\nState: cooldown\nOutput: 0.0 kW\nLoad: 0.0%\nFuel Level: 997 L / 1000 L\nFuel Consumption: 0.0 L/hr\nFuel Remaining: >100 hrs\nRated Capacity: 750 kW\n>> Generator in COOLDOWN, elapsed 20.0s.",
        dashboard=make_dashboard(
            sim_time_min=1.9, step=9, max_steps=15, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: COOLDOWN | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=6
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator post-shutdown inspection complete. In cooldown, 997L fuel (used only 3L during test). All protocol steps done: started ✓, verified ✓, stopped ✓. Now I acknowledge to complete the test protocol.",
        "acknowledge_alarm"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Alarm acknowledged.",
        dashboard=make_dashboard(
            sim_time_min=2.1, step=10, max_steps=15, scenario_type="power",
            alert="Generator test protocol completed successfully.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: COOLDOWN | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        ), steps_remaining=5
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Generator test protocol completed successfully. All 4 protocol steps followed in order: 1) Started generator, 2) Verified running via diagnose GEN-1, 3) Stopped generator (cooldown initiated), 4) Acknowledged alarm. Fuel usage was minimal (3L). Test passed — generator is ready for emergency use.",
        "wait"
    )})

    episodes.append({"conversations": convs})
    return episodes


def gen_b4_default_episodes():
    """B4: Power Failure Cascade — Default facility."""
    episodes = []
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    dash0 = make_dashboard(
        sim_time_min=0.0, step=0, max_steps=20, scenario_type="power",
        alert="CRITICAL: Utility power lost. UPS on battery. Generator startup in progress. Battery SOC declining. Immediate action required.",
        cracs=[
            "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
        ],
        zones=[
            "zone_a    19.9°C     33.6°C     19.9°C     80.0     A2",
            "zone_b    19.9°C     33.6°C     19.9°C     80.0     A2",
        ],
        racks_top5=[
            "A-05      19.9°C   33.7°C      8.0    1280",
            "B-05      19.9°C   33.7°C      8.0    1280",
            "A-03      19.8°C   33.5°C      8.0    1280",
            "B-03      19.8°C   33.5°C      8.0    1280",
            "A-01      19.8°C   33.5°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
        power_line="Utility: DOWN | Gen: STARTING (cranking) | ATS: TRANSFERRING",
        ups_line="UPS: UPS-1: BATTERY 97% 48min | UPS-2: BATTERY 97% 48min",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash0, steps_remaining=20
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "CRITICAL — utility power is DOWN. Both UPS units are on battery at 97% SOC with ~48 minutes remaining. Generator is auto-starting (cranking). ATS is transferring. This is urgent. First step: assess the full situation with check_status.",
        "check_status"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Full status displayed in dashboard.",
        dashboard=make_dashboard(
            sim_time_min=0.25, step=1, max_steps=20, scenario_type="power",
            alert="CRITICAL: Utility power lost. UPS on battery.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.9°C     33.6°C     19.9°C     80.0     A2",
                "zone_b    19.9°C     33.6°C     19.9°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.9°C   33.7°C      8.0    1280",
                "B-05      19.9°C   33.7°C      8.0    1280",
                "A-03      19.8°C   33.5°C      8.0    1280",
                "B-03      19.8°C   33.5°C      8.0    1280",
                "A-01      19.8°C   33.5°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: DOWN | Gen: STARTING (warming) | ATS: TRANSFERRING",
            ups_line="UPS: UPS-1: BATTERY 95% 45min | UPS-2: BATTERY 95% 45min",
        ), steps_remaining=19
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "UPS SOC dropping — 95% now. Generator is warming up. I need to diagnose UPS-1 to unlock the resolution gate and understand battery drain rate.",
        "diagnose UPS-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: UPS-1 ===\nMode: on_battery\nLoad: 16.0%\nEfficiency: 92.0%\nBattery SOC: 93%\nOutput: 85.0 kW\nLosses: 7.4 kW",
        dashboard=make_dashboard(
            sim_time_min=0.5, step=2, max_steps=20, scenario_type="power",
            alert="CRITICAL: Utility power lost. UPS on battery.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    20.0°C     33.7°C     20.0°C     80.0     A2",
                "zone_b    20.0°C     33.7°C     20.0°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      20.0°C   33.8°C      8.0    1280",
                "B-05      20.0°C   33.8°C      8.0    1280",
                "A-03      19.9°C   33.6°C      8.0    1280",
                "B-03      19.9°C   33.6°C      8.0    1280",
                "A-01      19.9°C   33.6°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: DOWN | Gen: STARTING (warming) | ATS: TRANSFERRING",
            ups_line="UPS: UPS-1: BATTERY 93% 43min | UPS-2: BATTERY 93% 43min",
        ), steps_remaining=18
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "UPS-1 on battery: 93% SOC, draining ~2% per step. Generator still warming up. I should also diagnose UPS-2 to verify both backup paths.",
        "diagnose UPS-2"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: UPS-2 ===\nMode: on_battery\nLoad: 17.0%\nEfficiency: 92.0%\nBattery SOC: 90%\nOutput: 86.0 kW\nLosses: 7.5 kW",
        dashboard=make_dashboard(
            sim_time_min=0.75, step=3, max_steps=20, scenario_type="power",
            alert="CRITICAL: Utility power lost.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    20.0°C     33.7°C     20.0°C     80.0     A2",
                "zone_b    20.0°C     33.7°C     20.0°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      20.0°C   33.8°C      8.0    1280",
                "B-05      20.0°C   33.8°C      8.0    1280",
                "A-03      20.0°C   33.7°C      8.0    1280",
                "B-03      20.0°C   33.7°C      8.0    1280",
                "A-01      19.9°C   33.6°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: DOWN | Gen: READY | ATS: TRANSFERRING",
            ups_line="UPS: UPS-1: BATTERY 90% 40min | UPS-2: BATTERY 90% 40min",
        ), steps_remaining=17
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Both UPS units on battery at 90% SOC. Generator is now READY. While generator auto-starts, I should proactively shed non-critical load to extend battery life and reduce thermal load. I'll reduce rack A-05 from 8 kW to 4 kW.",
        "set_rack_load A-05 4"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="IT load for rack A-05 set to 4.0 kW.",
        dashboard=make_dashboard(
            sim_time_min=1.0, step=4, max_steps=20, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.9°C     33.5°C     19.9°C     76.0     A2",
                "zone_b    20.0°C     33.7°C     20.0°C     80.0     A2",
            ],
            racks_top5=[
                "B-05      20.0°C   33.8°C      8.0    1280",
                "B-03      20.0°C   33.7°C      8.0    1280",
                "A-03      19.9°C   33.6°C      8.0    1280",
                "B-01      19.9°C   33.6°C      8.0    1280",
                "A-01      19.8°C   33.5°C      8.0    1280",
            ],
            it_load_kw=156.0, cooling_kw=116.2, pue=1.74,
            power_line="Utility: DOWN | Gen: LOADED 21% (fuel: >100h) | ATS: GENERATOR",
            ups_line="UPS: UPS-1: BATTERY 88% 42min | UPS-2: BATTERY 88% 42min",
        ), steps_remaining=16
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Rack A-05 shed to 4 kW. IT load reduced from 160 to 156 kW. Generator is now LOADED and ATS has transferred to GENERATOR — batteries should start recharging soon. Let me shed another rack to further conserve.",
        "set_rack_load B-05 4"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="IT load for rack B-05 set to 4.0 kW.",
        dashboard=make_dashboard(
            sim_time_min=1.25, step=5, max_steps=20, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.4",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.3",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.4",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.3",
            ],
            zones=[
                "zone_a    19.8°C     33.4°C     19.8°C     76.0     A2",
                "zone_b    19.8°C     33.4°C     19.8°C     76.0     A2",
            ],
            racks_top5=[
                "A-03      19.8°C   33.5°C      8.0    1280",
                "B-03      19.8°C   33.5°C      8.0    1280",
                "A-01      19.7°C   33.4°C      8.0    1280",
                "B-01      19.7°C   33.4°C      8.0    1280",
                "A-02      19.7°C   33.4°C      8.0    1280",
            ],
            it_load_kw=152.0, cooling_kw=114.6, pue=1.75,
            power_line="Utility: DOWN | Gen: LOADED 20% (fuel: >100h) | ATS: GENERATOR",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 15% η94% | UPS-2: DOUBLE CONVERSION 15% η94%",
        ), steps_remaining=15
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Excellent — UPS units have switched back to DOUBLE CONVERSION mode. Generator is LOADED and powering the facility through ATS. Both UPS units are now charging. Total IT load shed: 8 kW (5% of original). Let me wait for the system to stabilize.",
        "wait"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="Waiting. No action taken.",
        dashboard=make_dashboard(
            sim_time_min=1.5, step=6, max_steps=20, scenario_type="power",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.4",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.3",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.4",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.3",
            ],
            zones=[
                "zone_a    19.7°C     33.3°C     19.7°C     76.0     A2",
                "zone_b    19.7°C     33.3°C     19.7°C     76.0     A2",
            ],
            racks_top5=[
                "A-03      19.7°C   33.4°C      8.0    1280",
                "B-03      19.7°C   33.4°C      8.0    1280",
                "A-01      19.6°C   33.3°C      8.0    1280",
                "B-01      19.6°C   33.3°C      8.0    1280",
                "A-02      19.6°C   33.3°C      8.0    1280",
            ],
            it_load_kw=152.0, cooling_kw=114.2, pue=1.75,
            power_line="Utility: DOWN | Gen: LOADED 20% (fuel: >100h) | ATS: GENERATOR",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 15% η94% | UPS-2: DOUBLE CONVERSION 15% η94%",
        ), steps_remaining=14
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "System is stable on generator power. UPS back on double-conversion (charging). Temperatures stable at 19.7°C. Now let me verify the generator is operating properly with a diagnosis — this should trigger resolution since all conditions are met.",
        "diagnose GEN-1"
    )})

    convs.append({"from": "human", "value": format_user_turn(
        action_result="=== Diagnostic Report: GEN-1 ===\nState: loaded\nOutput: 152.0 kW\nLoad: 20.3%\nFuel Level: 998 L / 1000 L\nFuel Consumption: 40.0 L/hr\nFuel Remaining: 24.9 hrs\nRated Capacity: 750 kW\n>> Generator LOADED at 20%. Operating normally.",
        dashboard=make_dashboard(
            sim_time_min=1.75, step=7, max_steps=20, scenario_type="power",
            alert="Power failure resolved. Generator online, temps stable.",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.4",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.3",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.4",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.3",
            ],
            zones=[
                "zone_a    19.7°C     33.3°C     19.7°C     76.0     A2",
                "zone_b    19.7°C     33.3°C     19.7°C     76.0     A2",
            ],
            racks_top5=[
                "A-03      19.7°C   33.4°C      8.0    1280",
                "B-03      19.7°C   33.4°C      8.0    1280",
                "A-01      19.6°C   33.3°C      8.0    1280",
                "B-01      19.6°C   33.3°C      8.0    1280",
                "A-02      19.6°C   33.3°C      8.0    1280",
            ],
            it_load_kw=152.0, cooling_kw=114.2, pue=1.75,
            power_line="Utility: DOWN | Gen: LOADED 20% (fuel: 24.9h) | ATS: GENERATOR",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 15% η94% | UPS-2: DOUBLE CONVERSION 15% η94%",
        ), steps_remaining=13
    )})
    convs.append({"from": "gpt", "value": format_agent_turn(
        "Power failure resolved! Generator is LOADED at 20.3%, outputting 152 kW, operating normally. Fuel at 998L with 24.9 hours remaining. All temperatures stable at 19.7°C. UPS units back on double-conversion and charging. I shed 8 kW (5%) of IT load as a precaution during the transfer. The facility is now running stably on generator backup power.",
        "wait"
    )})

    episodes.append({"conversations": convs})
    return episodes


# ============================================================================
# COMMAND SYNTAX DRILL EPISODES (single-turn per command)
# ============================================================================

def gen_command_drill_episodes():
    """Generate short episodes that teach exact command syntax."""
    drills = []

    # Each drill: scenario context + single correct command
    drill_specs = [
        ("The CRAC-2 supply temperature seems unstable. You need to inspect it for faults.", "diagnose CRAC-2"),
        ("You need to check UPS-2 status after a power event.", "diagnose UPS-2"),
        ("PDU-B-03 might have a breaker issue. Inspect it.", "diagnose PDU-B-03"),
        ("You need to verify generator GEN-1 is in standby.", "diagnose GEN-1"),
        ("CRAC-4 setpoint is too low at 15°C. Raise it to 22°C for efficiency.", "adjust_setpoint CRAC-4 22"),
        ("CRAC-1 setpoint needs to go down to 16°C to compensate for a failed CRAC.", "adjust_setpoint CRAC-1 16"),
        ("CRAC-3 setpoint should be raised to 24°C per ASHRAE A2 recommendations.", "adjust_setpoint CRAC-3 24"),
        ("Reduce CRAC-2 fan speed to 70% to save energy.", "set_fan_speed CRAC-2 70"),
        ("Set CRAC-1 fan to maximum for emergency cooling.", "set_fan_speed CRAC-1 100"),
        ("CRAC-4 fan should be at 80% for balanced cooling.", "set_fan_speed CRAC-4 80"),
        ("Shed load on rack A-05 from 8 kW to 4 kW to reduce thermal load.", "set_rack_load A-05 4"),
        ("Migrate workload from rack B-03 to reduce it to 2 kW.", "set_rack_load B-03 2"),
        ("Reduce rack A-10 to 6 kW as a precautionary measure.", "set_rack_load A-10 6"),
        ("You need to start the standby CRAC-2 unit.", "start_crac CRAC-2"),
        ("Put CRAC-1 into standby mode for maintenance.", "stop_crac CRAC-1"),
        ("Initiate the diesel generator for emergency power.", "start_generator"),
        ("Begin generator cooldown after successful test.", "stop_generator"),
        ("Switch UPS-1 to eco mode for better efficiency.", "set_ups_mode UPS-1 eco"),
        ("Put UPS-2 in double_conversion mode for full protection.", "set_ups_mode UPS-2 double_conversion"),
        ("Set UPS-1 to bypass mode for maintenance.", "set_ups_mode UPS-1 bypass"),
        ("Refuel the generator to full tank.", "refuel_generator"),
        ("Add 500 liters of diesel to the generator.", "refuel_generator 500"),
        ("The alarm has been investigated. Clear it.", "acknowledge_alarm"),
        ("You need a full system overview before taking action.", "check_status"),
        ("The situation requires a senior engineer.", "escalate"),
        ("The generator is warming up. Wait for it to complete.", "wait"),
    ]

    for context, cmd in drill_specs:
        dash = make_dashboard(
            sim_time_min=5.0, step=3, max_steps=15, scenario_type="thermal",
            alert="",
            cracs=[
                "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
                "CRAC-3     RUNNING       18.0°C   18.2°C   100  12000    8.5",
                "CRAC-4     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            ],
            zones=[
                "zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2",
            ],
            racks_top5=[
                "A-05      19.5°C   33.3°C      8.0    1280",
                "B-05      19.5°C   33.3°C      8.0    1280",
                "A-03      19.4°C   33.2°C      8.0    1280",
                "B-03      19.4°C   33.2°C      8.0    1280",
                "A-01      19.4°C   33.1°C      8.0    1280",
            ],
            it_load_kw=160.0, cooling_kw=117.8, pue=1.74,
            power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
            ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
        )
        drills.append({"conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": format_user_turn(
                action_result=context,
                dashboard=dash, steps_remaining=12
            )},
            {"from": "gpt", "value": format_agent_turn(
                f"The situation requires: {context} I will issue the appropriate command.",
                cmd
            )}
        ]})

    return drills


# ============================================================================
# AUGMENTATION: Systematic variations
# ============================================================================

def augment_reasoning_variations(episodes):
    """Create reasoning style variations of existing episodes."""
    augmented = []
    reasoning_styles = [
        ("concise", lambda r: r.split(".")[0] + "." if "." in r else r),
        ("detailed", lambda r: r + f" This is consistent with proper datacenter operational procedure."),
        ("numbered", lambda r: "Step analysis: 1) " + r.replace(". ", ". 2) ", 1).replace(". 2) ", ". 3) ", 1) if ". " in r else "Analysis: " + r),
    ]

    for ep in episodes:
        for style_name, transform_fn in reasoning_styles:
            new_ep = copy.deepcopy(ep)
            for turn in new_ep["conversations"]:
                if turn["from"] == "gpt" and "<reasoning>" in turn["value"]:
                    # Extract and transform reasoning
                    parts = turn["value"].split("<reasoning>\n")
                    if len(parts) == 2:
                        rest = parts[1].split("\n</reasoning>")
                        if len(rest) == 2:
                            old_reasoning = rest[0]
                            new_reasoning = transform_fn(old_reasoning)
                            turn["value"] = f"<reasoning>\n{new_reasoning}\n</reasoning>" + rest[1]
            augmented.append(new_ep)

    return augmented


def augment_setpoint_variations(episodes):
    """Create setpoint value variations for A1 scenarios."""
    augmented = []
    setpoint_values = [20, 21, 22, 23, 25, 26]
    fan_values = [60, 65, 75, 80, 85]

    for sp in setpoint_values[:3]:
        for fv in fan_values[:2]:
            ep = copy.deepcopy(episodes[0]) if episodes else None
            if ep is None:
                continue
            for turn in ep["conversations"]:
                if turn["from"] == "gpt" and "adjust_setpoint" in turn["value"]:
                    turn["value"] = turn["value"].replace(" 24\n", f" {sp}\n")
                    turn["value"] = turn["value"].replace("24°C", f"{sp}°C")
                if turn["from"] == "gpt" and "set_fan_speed" in turn["value"]:
                    turn["value"] = turn["value"].replace(" 70\n", f" {fv}\n")
                    turn["value"] = turn["value"].replace("70%", f"{fv}%")
            augmented.append(ep)

    return augmented


def gen_error_recovery_episodes():
    """Generate episodes where agent makes a mistake and corrects."""
    episodes = []

    # Episode: Agent tries to adjust setpoint before diagnosing
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]
    dash = make_dashboard(
        sim_time_min=2.0, step=0, max_steps=15, scenario_type="thermal",
        alert="CRITICAL: CRAC-3 compressor failure detected. Zone B temperatures rising.",
        cracs=[
            "CRAC-1     RUNNING       18.0°C   18.2°C   100  12000    8.5",
            "CRAC-2     RUNNING       18.0°C   18.1°C   100  12000    8.4",
            "CRAC-3     !! COMPRESSOR 18.0°C      ---     0      0    0.0",
            "CRAC-4     RUNNING       18.0°C   18.2°C   100  12000    8.5",
        ],
        zones=[
            "zone_a    19.8°C     33.5°C     19.8°C     80.0     A2",
            "zone_b    20.5°C     34.3°C     20.5°C     80.0     A2",
        ],
        racks_top5=[
            "B-05      20.5°C   34.4°C      8.0    1280",
            "B-03      20.4°C   34.2°C      8.0    1280",
            "A-05      19.8°C   33.6°C      8.0    1280",
            "B-01      20.4°C   34.2°C      8.0    1280",
            "A-03      19.7°C   33.5°C      8.0    1280",
        ],
        it_load_kw=160.0, cooling_kw=109.4, pue=1.68,
        power_line="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
        ups_line="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 16% η94%",
    )
    convs.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash, steps_remaining=15
    )})
    # Agent makes a mistake — adjusts before diagnosing
    convs.append({"from": "gpt", "value": format_agent_turn(
        "CRAC-3 has failed. Wait — I should NOT adjust setpoints yet. Proper procedure requires diagnosing the faulty unit first. If I adjust without diagnosing, I'll get a procedure penalty (-0.2). Let me diagnose CRAC-3 first to unlock the resolution gate and earn the procedure bonus (+0.3).",
        "diagnose CRAC-3"
    )})
    episodes.append({"conversations": convs})

    # Episode: Agent issues unknown command
    convs2 = [{"from": "system", "value": SYSTEM_PROMPT}]
    convs2.append({"from": "human", "value": format_user_turn(
        action_result="Environment initialized. Awaiting your command.",
        dashboard=dash, steps_remaining=15
    )})
    convs2.append({"from": "gpt", "value": format_agent_turn(
        "I see CRAC-3 has a compressor failure. I need to start by assessing the situation properly. I'll use check_status to get a full overview, NOT an invalid command like 'repair CRAC-3' which doesn't exist in the action space.",
        "check_status"
    )})
    episodes.append({"conversations": convs2})

    return episodes


# ============================================================================
# MAIN: Generate all data
# ============================================================================

def main():
    all_episodes = []

    print("Generating seed episodes...")
    a1_eps = gen_a1_default_episodes()
    a2_eps = gen_a2_default_episodes()
    b1_eps = gen_b1_default_episodes()
    b3_eps = gen_b3_default_episodes()
    b4_eps = gen_b4_default_episodes()
    cmd_drills = gen_command_drill_episodes()
    error_eps = gen_error_recovery_episodes()

    seed_episodes = a1_eps + a2_eps + b1_eps + b3_eps + b4_eps + error_eps
    print(f"  Seed episodes: {len(seed_episodes)}")
    print(f"  Command drills: {len(cmd_drills)}")

    print("Augmenting with reasoning variations...")
    reasoning_aug = augment_reasoning_variations(seed_episodes)
    print(f"  Reasoning augmentations: {len(reasoning_aug)}")

    print("Augmenting with setpoint variations...")
    setpoint_aug = augment_setpoint_variations(a1_eps)
    print(f"  Setpoint augmentations: {len(setpoint_aug)}")

    all_episodes = seed_episodes + cmd_drills + reasoning_aug + setpoint_aug
    random.shuffle(all_episodes)

    # Write output
    output_path = Path("./dc_ops_sft_data.jsonl")
    with open(output_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\nTotal episodes: {len(all_episodes)}")
    print(f"Output: {output_path}")

    # Stats
    total_turns = sum(len(ep["conversations"]) for ep in all_episodes)
    agent_turns = sum(1 for ep in all_episodes for t in ep["conversations"] if t["from"] == "gpt")
    print(f"Total conversation turns: {total_turns}")
    print(f"Agent turns (training targets): {agent_turns}")

    # Verify command diversity
    commands_seen = set()
    for ep in all_episodes:
        for t in ep["conversations"]:
            if t["from"] == "gpt" and "<command>" in t["value"]:
                cmd = t["value"].split("<command>\n")[1].split("\n</command>")[0].strip()
                cmd_name = cmd.split()[0] if cmd else ""
                commands_seen.add(cmd_name)
    print(f"Unique command types used: {sorted(commands_seen)}")


if __name__ == "__main__":
    main()
