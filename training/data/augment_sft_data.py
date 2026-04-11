#!/usr/bin/env python3
"""
DC-Ops Synthetic Data Augmentation Engine

Takes the seed data from generate_sft_data.py and produces 500+ additional
high-quality training episodes through systematic variation of:
  - Facility configs (default/small/large)
  - Temperature values
  - Rack/CRAC IDs
  - Reasoning depth and style
  - Action ordering (valid alternatives)
  - Partial episodes (teach mid-episode recovery)
  - Multi-scenario mixes

Output: dc_ops_sft_augmented.jsonl
"""

import json
import random
import copy
from pathlib import Path

random.seed(123)

# ============================================================================
# SYSTEM PROMPT (same as seed data)
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


def fmt_agent(reasoning, command):
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<command>\n{command}\n</command>"


def make_dash(*, sim_min, step, max_steps, stype, alert="",
              cracs, zones, racks, it_kw, cool_kw, pue,
              pline="Utility: NORMAL | Gen: OFF | ATS: UTILITY",
              uline="UPS: UPS-1: DOUBLE CONVERSION 16% η94% | UPS-2: DOUBLE CONVERSION 17% η94%",
              outside=35.0):
    w = 68
    def r(t): return f"║ {t:<{w-2}} ║"
    def h(c="═"): return f"╠{c*w}╣"
    L = []
    L.append(f"╔{'═'*w}╗")
    L.append(f"║{'DC-OPS MONITORING DASHBOARD':^{w}}║")
    s = f"Sim Time: {sim_min:.1f} min    Step: {step}/{max_steps}    [{stype}]"
    L.append(r(s))
    if alert:
        L.append(h()); L.append(r(f"!! ALERT: {alert}"))
    L.append(h()); L.append(r("COOLING UNITS"))
    L.append(r(f"{'Unit':<10} {'Status':<12} {'Setpoint':>8} {'Supply':>8} {'Fan%':>5} {'CFM':>7} {'kW':>6}"))
    L.append(r("-"*(w-2)))
    for c in cracs: L.append(r(c))
    L.append(h()); L.append(r("ZONE TEMPERATURES"))
    L.append(r(f"{'Zone':<8} {'Cold Aisle':>10} {'Hot Aisle':>10} {'Max Inlet':>10} {'IT Load':>8} {'Class':>6}"))
    L.append(r("-"*(w-2)))
    for z in zones: L.append(r(z))
    L.append(h()); L.append(r("RACK TEMPERATURES (top 5 hottest)"))
    L.append(r(f"{'Rack':<8} {'Inlet':>8} {'Outlet':>8} {'Load kW':>8} {'CFM':>7}"))
    L.append(r("-"*(w-2)))
    for rk in racks: L.append(r(rk))
    L.append(h()); L.append(r("POWER"))
    L.append(r(f"IT Load: {it_kw:.1f} kW | Cooling: {cool_kw:.1f} kW | PUE: {pue:.2f}"))
    L.append(r(pline)); L.append(r(uline))
    L.append(h()); L.append(r("ENVIRONMENT"))
    L.append(r(f"Outside: {outside:.1f}°C | Humidity: 40% RH"))
    L.append(f"╚{'═'*w}╝")
    return "\n".join(L)


def usr(action_result, dashboard, steps_remaining):
    return f"**Action Result:** {action_result}\n\n**Steps Remaining:** {steps_remaining}\n\n{dashboard}"


# ============================================================================
# FACILITY CONFIGURATIONS
# ============================================================================

FACILITIES = {
    "default": {
        "zones": ["zone_a", "zone_b"],
        "cracs": ["CRAC-1", "CRAC-2", "CRAC-3", "CRAC-4"],
        "zone_cracs": {"zone_a": ["CRAC-1", "CRAC-2"], "zone_b": ["CRAC-3", "CRAC-4"]},
        "racks_per_zone": {"zone_a": [f"A-{i:02d}" for i in range(1,11)],
                           "zone_b": [f"B-{i:02d}" for i in range(1,11)]},
        "it_load_kw": 160.0, "ashrae": "A2",
    },
    "small": {
        "zones": ["zone_a"],
        "cracs": ["CRAC-1", "CRAC-2"],
        "zone_cracs": {"zone_a": ["CRAC-1", "CRAC-2"]},
        "racks_per_zone": {"zone_a": [f"A-{i:02d}" for i in range(1,11)]},
        "it_load_kw": 80.0, "ashrae": "A2",
    },
    "large": {
        "zones": ["zone_a", "zone_b", "zone_c", "zone_d"],
        "cracs": [f"CRAC-{i}" for i in range(1,9)],
        "zone_cracs": {"zone_a": ["CRAC-1","CRAC-2"], "zone_b": ["CRAC-3","CRAC-4"],
                       "zone_c": ["CRAC-5","CRAC-6"], "zone_d": ["CRAC-7","CRAC-8"]},
        "racks_per_zone": {"zone_a": [f"A-{i:02d}" for i in range(1,11)],
                           "zone_b": [f"B-{i:02d}" for i in range(1,11)],
                           "zone_c": [f"C-{i:02d}" for i in range(1,6)],
                           "zone_d": [f"D-{i:02d}" for i in range(1,6)]},
        "it_load_kw": 600.0, "ashrae": "A2",
    },
}

# ============================================================================
# REASONING TEMPLATES (to avoid repetition)
# ============================================================================

ASSESS_REASONS = [
    "I need to understand the full situation before acting. Let me check the system status first.",
    "First step in any incident response: assess. I'll request a full status report.",
    "Before making any changes, I should get a complete picture of the datacenter state.",
    "Standard operating procedure begins with a status check to establish baseline.",
    "My first priority is to understand what we're dealing with. Requesting dashboard status.",
    "Incident response protocol step 1: assess the situation. Checking overall system status.",
    "I need to see the full dashboard to understand zone temperatures, cooling status, and power state.",
    "Assessment first — I can't make good decisions without knowing the current state of all systems.",
]

DIAGNOSE_REASONS_CRAC = [
    "I need to formally diagnose {unit} to confirm the fault type and unlock the resolution path.",
    "{unit} is showing a fault. Diagnosing it will give me detailed information and is required before I can compensate.",
    "Proper procedure requires diagnosing {unit} before adjusting other CRACs. This earns the procedure bonus.",
    "The dashboard shows {unit} has an issue. Let me run diagnostics to confirm the exact fault.",
    "I must diagnose the faulty unit ({unit}) before taking corrective action. This is both good practice and required for resolution.",
    "Before compensating with other CRACs, I need to diagnose {unit} to understand what failed.",
]

DIAGNOSE_REASONS_HEALTHY = [
    "Let me verify {unit} is healthy before relying on it for compensation.",
    "I should confirm {unit} has no issues — I'll be increasing its load to compensate.",
    "Checking {unit} to make sure it's operating normally before pushing it harder.",
    "Thorough diagnosis requires checking all units, not just the faulty one. Inspecting {unit}.",
]

SETPOINT_LOWER_REASONS = [
    "Lowering {unit} setpoint from {old}°C to {new}°C to increase its cooling output and compensate for the lost CRAC.",
    "I'll reduce {unit}'s setpoint to {new}°C. This makes the compressor work harder, extracting more heat from the airstream.",
    "{unit} setpoint going from {old}°C down to {new}°C — this will increase cooling capacity on this unit.",
    "Compensating by lowering {unit} to {new}°C setpoint. The colder supply air will help offset the missing CRAC.",
]

SETPOINT_RAISE_REASONS = [
    "Raising {unit} from {old}°C to {new}°C. This reduces compressor work and will lower PUE.",
    "Setting {unit} to {new}°C — well within ASHRAE A2 recommended range (18-27°C) while improving efficiency.",
    "{unit} setpoint increase to {new}°C. Inlet temps have margin — this cuts unnecessary cooling energy.",
    "Adjusting {unit} to {new}°C for better energy efficiency. Current inlet temps are far below ASHRAE limits.",
]

FAN_REASONS = [
    "Setting {unit} fan speed to {pct}%. Fan power follows the cubic law: P ∝ (speed)³.",
    "Adjusting {unit} fan to {pct}% for optimal airflow balance.",
    "{unit} fan speed to {pct}%. This adjusts cooling capacity appropriately for the current conditions.",
]

LOAD_SHED_REASONS = [
    "Shedding load on rack {rack} from {old} kW to {new} kW to reduce thermal load and extend battery life.",
    "Reducing rack {rack} to {new} kW. Less IT load means less heat to extract and lower power demand.",
    "Migrating workload off rack {rack} — dropping to {new} kW provides thermal and power margin.",
    "Load shedding on {rack}: {old} kW → {new} kW. This buys time for the generator to come online.",
]

WAIT_REASONS = [
    "The generator is in its startup sequence. I need to wait for it to complete warmup.",
    "Allowing time for the simulation to advance. The system needs time to respond to my changes.",
    "Waiting for the generator to finish its warmup cycle before I can verify it.",
    "The cooldown process takes time. I'll wait for it to progress.",
    "No immediate action needed — letting the thermal dynamics settle after my adjustments.",
]

ACK_REASONS = [
    "All investigations complete. Acknowledging the alarm to close out this incident.",
    "Power chain audit finished — everything is healthy. Time to acknowledge and clear the alarm.",
    "Investigation thorough and complete. Acknowledging to finalize the protocol.",
    "Alarm acknowledged after proper investigation and verification of all systems.",
]

GEN_VERIFY_REASONS = [
    "Generator should be running now. I need to diagnose GEN-1 to verify it's operating properly — this is a required protocol step.",
    "Critical verification step: diagnosing GEN-1 to confirm it started successfully and is producing stable output.",
    "The generator should have completed startup. Let me verify with a diagnostic check on GEN-1.",
]


# ============================================================================
# EPISODE GENERATORS — Parametric
# ============================================================================

def gen_a1_parametric(facility_name, setpoint, fan_pct):
    """A1: Cooling Setpoint Optimization — parametric."""
    fac = FACILITIES[facility_name]
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    n_cracs = len(fac["cracs"])
    n_zones = len(fac["zones"])
    base_inlet = round(random.uniform(16.5, 17.5), 1)
    base_pue = round(random.uniform(1.82, 1.92), 2)

    # Build initial CRAC lines
    crac_lines = []
    for c in fac["cracs"]:
        crac_lines.append(f"{c:<10} RUNNING       15.0°C   15.2°C   100  12000    9.8")

    zone_lines = []
    for z in fac["zones"]:
        cls = "H1" if "zone_c" == z and facility_name == "large" else fac["ashrae"]
        zone_lines.append(f"{z:<8} {base_inlet:>9.1f}°C     32.5°C {base_inlet:>9.1f}°C     {fac['it_load_kw']/n_zones:.1f} {cls:>6}")

    all_racks = []
    for z, rlist in fac["racks_per_zone"].items():
        for rk in rlist[:2]:
            all_racks.append(f"{rk:<8} {base_inlet:>7.1f}°C   32.6°C      8.0    1280")
    rack_lines = all_racks[:5]

    dash = make_dash(sim_min=2.0, step=0, max_steps=10, stype="thermal",
                     alert="NOTICE: PUE exceeds 1.8 — cooling setpoints may be suboptimal.",
                     cracs=crac_lines, zones=zone_lines, racks=rack_lines,
                     it_kw=fac["it_load_kw"], cool_kw=fac["it_load_kw"]*0.87, pue=base_pue)

    convs.append({"from": "human", "value": usr("Environment initialized. Awaiting your command.", dash, 10)})
    convs.append({"from": "gpt", "value": fmt_agent(random.choice(ASSESS_REASONS), "check_status")})

    # Step 1: status confirmed
    convs.append({"from": "human", "value": usr("Full status displayed in dashboard.", dash, 9)})

    # Raise setpoints one by one
    step = 2
    current_pue = base_pue
    current_inlet = base_inlet
    for i, crac in enumerate(fac["cracs"]):
        current_pue = round(current_pue - random.uniform(0.03, 0.06), 2)
        current_inlet = round(current_inlet + random.uniform(0.5, 1.2), 1)
        reason = random.choice(SETPOINT_RAISE_REASONS).format(unit=crac, old=15, new=setpoint)
        convs.append({"from": "gpt", "value": fmt_agent(reason, f"adjust_setpoint {crac} {setpoint}")})

        # Next dashboard
        new_crac_lines = []
        for j, c in enumerate(fac["cracs"]):
            sp = setpoint if j <= i else 15.0
            sup = round(sp - random.uniform(0.5, 2.0), 1)
            kw = round(9.8 - (sp - 15) * 0.25, 1) if sp > 15 else 9.8
            new_crac_lines.append(f"{c:<10} RUNNING       {sp:.1f}°C   {sup:.1f}°C   100  12000  {kw:>5.1f}")

        new_zone_lines = []
        for z in fac["zones"]:
            cls = fac["ashrae"]
            new_zone_lines.append(f"{z:<8} {current_inlet:>9.1f}°C     {current_inlet+13:.1f}°C {current_inlet:>9.1f}°C     {fac['it_load_kw']/n_zones:.1f} {cls:>6}")

        ndash = make_dash(sim_min=2.0+step, step=step, max_steps=10, stype="thermal",
                          alert="", cracs=new_crac_lines, zones=new_zone_lines, racks=rack_lines,
                          it_kw=fac["it_load_kw"], cool_kw=round(fac["it_load_kw"]*0.87 - i*5, 1), pue=current_pue)
        convs.append({"from": "human", "value": usr(
            f"Setpoint for {crac} adjusted to {setpoint:.1f}°C. Supply temp will converge over ~30 seconds.",
            ndash, 10-step)})
        step += 1

    # Fan speed reductions
    for i, crac in enumerate(fac["cracs"][:min(4, n_cracs)]):
        current_pue = round(current_pue - random.uniform(0.01, 0.03), 2)
        current_inlet = round(current_inlet + random.uniform(0.3, 0.6), 1)
        reason = random.choice(FAN_REASONS).format(unit=crac, pct=fan_pct)
        convs.append({"from": "gpt", "value": fmt_agent(
            f"PUE at {current_pue:.2f}. {reason}",
            f"set_fan_speed {crac} {fan_pct}")})

        ndash = make_dash(sim_min=2.0+step, step=step, max_steps=10, stype="thermal",
                          alert="PUE optimized within target range." if current_pue <= 1.60 else "",
                          cracs=new_crac_lines, zones=new_zone_lines, racks=rack_lines,
                          it_kw=fac["it_load_kw"], cool_kw=round(fac["it_load_kw"]*0.6, 1),
                          pue=max(1.55, current_pue))
        convs.append({"from": "human", "value": usr(
            f"Fan speed for {crac} set to {fan_pct}%.",
            ndash, max(1, 10-step))})
        step += 1
        if step >= 10:
            break

    # Final response
    convs.append({"from": "gpt", "value": fmt_agent(
        f"PUE optimization complete. Raised all setpoints to {setpoint}°C and reduced fan speeds to {fan_pct}%. "
        f"All inlet temperatures remain within ASHRAE {fac['ashrae']} recommended range.",
        "wait")})

    return {"conversations": convs}


def gen_a2_parametric(facility_name, failed_crac):
    """A2: Thermal Event — parametric with different failed CRACs."""
    fac = FACILITIES[facility_name]
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    # Find which zone the failed CRAC is in
    failed_zone = None
    for z, clist in fac["zone_cracs"].items():
        if failed_crac in clist:
            failed_zone = z
            break
    if not failed_zone:
        return None

    surviving_cracs = [c for c in fac["cracs"] if c != failed_crac]

    # Initial dashboard
    crac_lines = []
    for c in fac["cracs"]:
        if c == failed_crac:
            crac_lines.append(f"{c:<10} !! COMPRESSOR 18.0°C      ---     0      0    0.0")
        else:
            crac_lines.append(f"{c:<10} RUNNING       18.0°C   18.2°C   100  12000    8.5")

    base_a = round(random.uniform(19.5, 20.2), 1)
    base_b = round(base_a + random.uniform(0.2, 0.8), 1)
    zone_lines = []
    for i, z in enumerate(fac["zones"]):
        t = base_b if z == failed_zone else base_a
        zone_lines.append(f"{z:<8} {t:>9.1f}°C     {t+13:.1f}°C {t:>9.1f}°C     {fac['it_load_kw']/len(fac['zones']):.1f} {fac['ashrae']:>6}")

    rack_lines = []
    for z, rlist in fac["racks_per_zone"].items():
        t = base_b if z == failed_zone else base_a
        rack_lines.append(f"{rlist[4]:<8} {t:>7.1f}°C   {t+13:.1f}°C      8.0    1280")
    rack_lines = rack_lines[:5]

    dash = make_dash(sim_min=2.0, step=0, max_steps=15, stype="thermal",
                     alert=f"CRITICAL: {failed_crac} compressor failure detected. Temperatures rising. Investigate and stabilize.",
                     cracs=crac_lines, zones=zone_lines, racks=rack_lines,
                     it_kw=fac["it_load_kw"], cool_kw=round(fac["it_load_kw"]*0.68,1),
                     pue=round(random.uniform(1.65, 1.72),2))

    convs.append({"from": "human", "value": usr("Environment initialized. Awaiting your command.", dash, 15)})
    convs.append({"from": "gpt", "value": fmt_agent(random.choice(ASSESS_REASONS), "check_status")})
    convs.append({"from": "human", "value": usr("Full status displayed in dashboard.", dash, 14)})

    # Diagnose faulty unit
    convs.append({"from": "gpt", "value": fmt_agent(
        random.choice(DIAGNOSE_REASONS_CRAC).format(unit=failed_crac),
        f"diagnose {failed_crac}")})

    diag_result = (f"=== Diagnostic Report: {failed_crac} ===\nStatus: fault\nFault: compressor\n"
                   f"Setpoint: 18.0°C\nSupply Temp: 18.0°C\nFan Speed: 0%\nAirflow: 0.000 m³/s\n"
                   f">> FAULT DETECTED: compressor\n>> Recommended: repair or replace component")
    convs.append({"from": "human", "value": usr(diag_result, dash, 13)})

    # Diagnose a healthy unit
    healthy = random.choice(surviving_cracs[:2])
    convs.append({"from": "gpt", "value": fmt_agent(
        random.choice(DIAGNOSE_REASONS_HEALTHY).format(unit=healthy),
        f"diagnose {healthy}")})

    healthy_result = (f"=== Diagnostic Report: {healthy} ===\nStatus: running\nFault: none\n"
                      f"Setpoint: 18.0°C\nSupply Temp: 18.2°C\nFan Speed: 100%\nAirflow: 5.663 m³/s\n"
                      f">> No faults detected. Unit operating normally.")
    convs.append({"from": "human", "value": usr(healthy_result, dash, 12)})

    # Compensate — lower setpoints on 2 survivors
    for i, crac in enumerate(surviving_cracs[:2]):
        reason = random.choice(SETPOINT_LOWER_REASONS).format(unit=crac, old=18, new=16)
        convs.append({"from": "gpt", "value": fmt_agent(reason, f"adjust_setpoint {crac} 16")})
        convs.append({"from": "human", "value": usr(
            f"Setpoint for {crac} adjusted to 16.0°C. Supply temp will converge over ~30 seconds.",
            dash, 11-i)})

    # Max fan speeds
    for i, crac in enumerate(surviving_cracs[:2]):
        reason = random.choice(FAN_REASONS).format(unit=crac, pct=100)
        convs.append({"from": "gpt", "value": fmt_agent(reason, f"set_fan_speed {crac} 100")})

        alert_msg = "Thermal event stabilized. All zones within recommended range." if i == 1 else ""
        resolve_dash = make_dash(sim_min=8.0+i, step=7+i, max_steps=15, stype="thermal",
                                 alert=alert_msg, cracs=crac_lines, zones=zone_lines, racks=rack_lines,
                                 it_kw=fac["it_load_kw"], cool_kw=round(fac["it_load_kw"]*0.75,1),
                                 pue=round(random.uniform(1.72,1.80),2))
        convs.append({"from": "human", "value": usr(
            f"Fan speed for {crac} set to 100%.",
            resolve_dash, 8-i)})

    convs.append({"from": "gpt", "value": fmt_agent(
        f"Thermal event resolved. {failed_crac} compressor fault was diagnosed. The surviving CRACs are compensating. "
        f"All zones within ASHRAE {fac['ashrae']} recommended range.",
        "wait")})

    return {"conversations": convs}


def gen_a4_parametric(facility_name):
    """A4: CRAC Cascade — parametric."""
    fac = FACILITIES[facility_name]
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    failed1, failed2 = "CRAC-1", "CRAC-3"
    survivors = [c for c in fac["cracs"] if c not in (failed1, failed2)]

    crac_lines = []
    for c in fac["cracs"]:
        if c == failed1:
            crac_lines.append(f"{c:<10} !! COMPRESSOR 18.0°C      ---     0      0    0.0")
        elif c == failed2:
            crac_lines.append(f"{c:<10} !! FAN        18.0°C      ---     0      0    0.0")
        else:
            crac_lines.append(f"{c:<10} RUNNING       18.0°C   18.2°C   100  12000    8.5")

    base_t = round(random.uniform(19.8, 20.2), 1)
    zone_lines = [f"{z:<8} {base_t:>9.1f}°C     {base_t+13:.1f}°C {base_t:>9.1f}°C     {fac['it_load_kw']/len(fac['zones']):.1f} {fac['ashrae']:>6}"
                  for z in fac["zones"]]
    rack_lines = [f"{'B-05':<8} {base_t:>7.1f}°C   {base_t+13:.1f}°C      8.0    1280"] * 5

    dash = make_dash(sim_min=1.0, step=0, max_steps=20, stype="thermal",
                     alert="CRITICAL: Multiple CRAC failures detected. CRAC-1 compressor fault, CRAC-3 fan fault. Temperatures rising rapidly.",
                     cracs=crac_lines, zones=zone_lines, racks=rack_lines,
                     it_kw=fac["it_load_kw"], cool_kw=round(fac["it_load_kw"]*0.5,1),
                     pue=round(random.uniform(1.68,1.75),2))

    convs.append({"from": "human", "value": usr("Environment initialized. Awaiting your command.", dash, 20)})
    convs.append({"from": "gpt", "value": fmt_agent(
        "CRITICAL — TWO CRACs down. 50% cooling capacity lost. I must act quickly. Assess first.", "check_status")})
    convs.append({"from": "human", "value": usr("Full status displayed in dashboard.", dash, 19)})

    # Diagnose both
    convs.append({"from": "gpt", "value": fmt_agent(
        random.choice(DIAGNOSE_REASONS_CRAC).format(unit=failed1), f"diagnose {failed1}")})
    convs.append({"from": "human", "value": usr(
        f"=== Diagnostic Report: {failed1} ===\nStatus: fault\nFault: compressor\n>> FAULT DETECTED: compressor", dash, 18)})

    convs.append({"from": "gpt", "value": fmt_agent(
        f"First fault confirmed. Now diagnosing {failed2} — both must be diagnosed for resolution.", f"diagnose {failed2}")})
    convs.append({"from": "human", "value": usr(
        f"=== Diagnostic Report: {failed2} ===\nStatus: fault\nFault: fan\n>> FAULT DETECTED: fan", dash, 17)})

    # Compensate survivors
    for i, crac in enumerate(survivors[:2]):
        convs.append({"from": "gpt", "value": fmt_agent(
            random.choice(SETPOINT_LOWER_REASONS).format(unit=crac, old=18, new=16),
            f"adjust_setpoint {crac} 16")})
        convs.append({"from": "human", "value": usr(f"Setpoint for {crac} adjusted to 16.0°C.", dash, 16-i)})

    for i, crac in enumerate(survivors[:2]):
        convs.append({"from": "gpt", "value": fmt_agent(
            random.choice(FAN_REASONS).format(unit=crac, pct=100),
            f"set_fan_speed {crac} 100")})
        convs.append({"from": "human", "value": usr(f"Fan speed for {crac} set to 100%.", dash, 14-i)})

    # Load shed
    shed_rack = random.choice(["B-05", "A-05", "B-03"])
    convs.append({"from": "gpt", "value": fmt_agent(
        random.choice(LOAD_SHED_REASONS).format(rack=shed_rack, old=8, new=4),
        f"set_rack_load {shed_rack} 4")})
    resolve_dash = make_dash(sim_min=8.0, step=8, max_steps=20, stype="thermal",
                             alert="CRAC cascade stabilized. Temps within allowable range.",
                             cracs=crac_lines, zones=zone_lines, racks=rack_lines,
                             it_kw=fac["it_load_kw"]-4, cool_kw=round(fac["it_load_kw"]*0.55,1),
                             pue=round(random.uniform(1.70,1.80),2))
    convs.append({"from": "human", "value": usr(f"IT load for rack {shed_rack} set to 4.0 kW.", resolve_dash, 12)})
    convs.append({"from": "gpt", "value": fmt_agent(
        "Cascade stabilized. Both faults diagnosed, survivors compensating at max, load shed performed. All zones within allowable limits.",
        "wait")})

    return {"conversations": convs}


def gen_b4_parametric(facility_name):
    """B4: Power Failure Cascade — parametric."""
    fac = FACILITIES[facility_name]
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    soc = random.randint(94, 98)
    dash = make_dash(sim_min=0.0, step=0, max_steps=20, stype="power",
                     alert="CRITICAL: Utility power lost. UPS on battery. Generator startup in progress.",
                     cracs=[f"{c:<10} RUNNING       18.0°C   18.2°C   100  12000    8.5" for c in fac["cracs"]],
                     zones=[f"{z:<8}    19.9°C     33.6°C     19.9°C     {fac['it_load_kw']/len(fac['zones']):.1f} {fac['ashrae']:>6}" for z in fac["zones"]],
                     racks=[f"{'A-05':<8}    19.9°C   33.7°C      8.0    1280"]*5,
                     it_kw=fac["it_load_kw"], cool_kw=round(fac["it_load_kw"]*0.74,1),
                     pue=1.74,
                     pline="Utility: DOWN | Gen: STARTING (cranking) | ATS: TRANSFERRING",
                     uline=f"UPS: UPS-1: BATTERY {soc}% 48min | UPS-2: BATTERY {soc}% 48min")

    convs.append({"from": "human", "value": usr("Environment initialized. Awaiting your command.", dash, 20)})
    convs.append({"from": "gpt", "value": fmt_agent(
        f"CRITICAL — utility power DOWN. UPS on battery at {soc}% SOC. Generator auto-starting. I need to assess, diagnose UPS, then consider load shedding.",
        "check_status")})
    convs.append({"from": "human", "value": usr("Full status displayed in dashboard.", dash, 19)})

    convs.append({"from": "gpt", "value": fmt_agent(
        "I must diagnose UPS-1 to unlock the resolution gate and assess battery drain rate.",
        "diagnose UPS-1")})
    soc -= 3
    convs.append({"from": "human", "value": usr(
        f"=== Diagnostic Report: UPS-1 ===\nMode: on_battery\nLoad: 16.0%\nEfficiency: 92.0%\nBattery SOC: {soc}%\nOutput: 85.0 kW\nLosses: 7.4 kW",
        dash, 18)})

    convs.append({"from": "gpt", "value": fmt_agent(
        f"UPS-1 on battery at {soc}% SOC. Draining ~2-3% per step. Let me also check UPS-2.",
        "diagnose UPS-2")})
    soc -= 3
    convs.append({"from": "human", "value": usr(
        f"=== Diagnostic Report: UPS-2 ===\nMode: on_battery\nLoad: 17.0%\nBattery SOC: {soc}%",
        dash, 17)})

    # Load shed
    shed_racks = random.sample(["A-05", "B-05", "A-04", "B-04", "A-03"], k=random.randint(2,3))
    for rk in shed_racks:
        new_kw = random.choice([2, 3, 4])
        convs.append({"from": "gpt", "value": fmt_agent(
            random.choice(LOAD_SHED_REASONS).format(rack=rk, old=8, new=new_kw),
            f"set_rack_load {rk} {new_kw}")})
        convs.append({"from": "human", "value": usr(f"IT load for rack {rk} set to {new_kw:.1f} kW.", dash, 16)})

    # Wait for generator
    convs.append({"from": "gpt", "value": fmt_agent(random.choice(WAIT_REASONS), "wait")})
    gen_dash = make_dash(sim_min=1.5, step=7, max_steps=20, stype="power", alert="",
                         cracs=[f"{c:<10} RUNNING       18.0°C   18.2°C   100  12000    8.4" for c in fac["cracs"]],
                         zones=[f"{z:<8}    19.7°C     33.3°C     19.7°C     {fac['it_load_kw']/len(fac['zones']):.1f} {fac['ashrae']:>6}" for z in fac["zones"]],
                         racks=[f"{'A-03':<8}    19.7°C   33.4°C      8.0    1280"]*5,
                         it_kw=fac["it_load_kw"]-len(shed_racks)*4, cool_kw=round(fac["it_load_kw"]*0.70,1),
                         pue=1.75,
                         pline="Utility: DOWN | Gen: LOADED 20% (fuel: >100h) | ATS: GENERATOR",
                         uline="UPS: UPS-1: DOUBLE CONVERSION 15% η94% | UPS-2: DOUBLE CONVERSION 15% η94%")
    convs.append({"from": "human", "value": usr("Waiting. No action taken.", gen_dash, 13)})

    # Verify generator
    convs.append({"from": "gpt", "value": fmt_agent(random.choice(GEN_VERIFY_REASONS), "diagnose GEN-1")})
    resolve_dash = make_dash(sim_min=1.75, step=8, max_steps=20, stype="power",
                             alert="Power failure resolved. Generator online, temps stable.",
                             cracs=[f"{c:<10} RUNNING       18.0°C   18.2°C   100  12000    8.4" for c in fac["cracs"]],
                             zones=[f"{z:<8}    19.7°C     33.3°C     19.7°C     {fac['it_load_kw']/len(fac['zones']):.1f} {fac['ashrae']:>6}" for z in fac["zones"]],
                             racks=[f"{'A-03':<8}    19.7°C   33.4°C      8.0    1280"]*5,
                             it_kw=fac["it_load_kw"]-len(shed_racks)*4, cool_kw=round(fac["it_load_kw"]*0.70,1),
                             pue=1.75,
                             pline="Utility: DOWN | Gen: LOADED 20% (fuel: 24.9h) | ATS: GENERATOR",
                             uline="UPS: UPS-1: DOUBLE CONVERSION 15% η94% | UPS-2: DOUBLE CONVERSION 15% η94%")
    convs.append({"from": "human", "value": usr(
        "=== Diagnostic Report: GEN-1 ===\nState: loaded\nOutput: 152.0 kW\nLoad: 20.3%\n>> Generator LOADED at 20%. Operating normally.",
        resolve_dash, 12)})
    convs.append({"from": "gpt", "value": fmt_agent(
        f"Power failure resolved. Generator LOADED and operating normally. UPS back on double-conversion. "
        f"Shed {len(shed_racks)} racks during transfer. Facility stable on backup power.",
        "wait")})

    return {"conversations": convs}


def gen_b1_parametric():
    """B1: UPS Alarm — with varying investigation depth."""
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    soc = random.randint(82, 90)
    dash = make_dash(sim_min=2.0, step=0, max_steps=10, stype="power",
                     alert=f"WARNING: UPS-1 transferred to battery. Utility restored. Battery SOC: {soc}%.",
                     cracs=[f"CRAC-{i:<6} RUNNING       18.0°C   18.2°C   100  12000    8.5" for i in range(1,5)],
                     zones=["zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                            "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2"],
                     racks=["A-05      19.5°C   33.3°C      8.0    1280"]*5,
                     it_kw=160.0, cool_kw=117.8, pue=1.74)

    convs.append({"from": "human", "value": usr("Environment initialized. Awaiting your command.", dash, 10)})
    convs.append({"from": "gpt", "value": fmt_agent(random.choice(ASSESS_REASONS), "check_status")})
    convs.append({"from": "human", "value": usr("Full status displayed in dashboard.", dash, 9)})

    convs.append({"from": "gpt", "value": fmt_agent(
        "UPS alarm requires diagnosis. I must inspect UPS-1 to verify it's back on normal mode and check SOC.",
        "diagnose UPS-1")})
    convs.append({"from": "human", "value": usr(
        f"=== Diagnostic Report: UPS-1 ===\nMode: double_conversion\nLoad: 16.0%\nBattery SOC: {soc+1}%",
        dash, 8)})

    # Random investigation depth
    extra_checks = random.sample(["diagnose UPS-2", "diagnose GEN-1", "diagnose PDU-A-01", "diagnose PDU-B-01"], k=random.randint(1,3))
    for chk in extra_checks:
        unit = chk.split()[-1]
        convs.append({"from": "gpt", "value": fmt_agent(
            f"Verifying {unit} as part of power chain audit.", chk)})
        convs.append({"from": "human", "value": usr(
            f"=== Diagnostic Report: {unit} ===\n>> No faults detected. Operating normally.", dash, 7)})

    convs.append({"from": "gpt", "value": fmt_agent(random.choice(ACK_REASONS), "acknowledge_alarm")})
    convs.append({"from": "human", "value": usr("Alarm acknowledged.", dash, 5)})
    convs.append({"from": "gpt", "value": fmt_agent(
        "UPS alarm properly investigated and acknowledged. All power chain components verified healthy.",
        "wait")})

    return {"conversations": convs}


def gen_b3_parametric():
    """B3: Generator Test — with timing variations."""
    convs = [{"from": "system", "value": SYSTEM_PROMPT}]

    dash = make_dash(sim_min=0.2, step=0, max_steps=15, stype="power",
                     alert="SCHEDULED: Monthly generator test due. Follow standard test protocol.",
                     cracs=[f"CRAC-{i:<6} RUNNING       18.0°C   18.2°C   100  12000    8.5" for i in range(1,5)],
                     zones=["zone_a    19.5°C     33.2°C     19.5°C     80.0     A2",
                            "zone_b    19.5°C     33.2°C     19.5°C     80.0     A2"],
                     racks=["A-05      19.5°C   33.3°C      8.0    1280"]*5,
                     it_kw=160.0, cool_kw=117.8, pue=1.74)

    convs.append({"from": "human", "value": usr("Environment initialized. Awaiting your command.", dash, 15)})
    convs.append({"from": "gpt", "value": fmt_agent(
        "Monthly generator test. Protocol: check_status → start_generator → wait → diagnose GEN-1 → stop_generator → acknowledge_alarm.",
        "check_status")})
    convs.append({"from": "human", "value": usr("Full status displayed in dashboard.", dash, 14)})

    # Optional pre-inspect
    if random.random() > 0.5:
        convs.append({"from": "gpt", "value": fmt_agent(
            "Pre-test inspection of generator before starting.",
            "diagnose GEN-1")})
        convs.append({"from": "human", "value": usr(
            "=== Diagnostic Report: GEN-1 ===\nState: off\nFuel Level: 1000 L / 1000 L\n>> Generator is OFF.", dash, 13)})

    convs.append({"from": "gpt", "value": fmt_agent(
        "Starting the generator for the monthly test.", "start_generator")})
    convs.append({"from": "human", "value": usr("Generator start sequence initiated.", dash, 12)})

    convs.append({"from": "gpt", "value": fmt_agent(random.choice(WAIT_REASONS), "wait")})
    convs.append({"from": "human", "value": usr("Waiting. No action taken.", dash, 11)})

    convs.append({"from": "gpt", "value": fmt_agent(random.choice(GEN_VERIFY_REASONS), "diagnose GEN-1")})
    convs.append({"from": "human", "value": usr(
        "=== Diagnostic Report: GEN-1 ===\nState: loaded\nOutput: 157.5 kW\nLoad: 21.0%\n>> Generator LOADED at 21%. Operating normally.",
        dash, 10)})

    # Optional status check while running
    if random.random() > 0.5:
        convs.append({"from": "gpt", "value": fmt_agent(
            "Verified. Let me check full system while generator is running.", "check_status")})
        convs.append({"from": "human", "value": usr("Full status displayed in dashboard.", dash, 9)})

    convs.append({"from": "gpt", "value": fmt_agent(
        "Test successful. Initiating generator cooldown.", "stop_generator")})
    convs.append({"from": "human", "value": usr("Generator cooldown initiated.", dash, 8)})

    convs.append({"from": "gpt", "value": fmt_agent(
        "Cooldown in progress. Waiting for turbocharger to cool.", "wait")})
    convs.append({"from": "human", "value": usr("Waiting. No action taken.", dash, 7)})

    convs.append({"from": "gpt", "value": fmt_agent(
        "Protocol complete: started ✓, verified ✓, stopped ✓. Acknowledging to finalize.", "acknowledge_alarm")})
    convs.append({"from": "human", "value": usr("Alarm acknowledged.", dash, 6)})
    convs.append({"from": "gpt", "value": fmt_agent(
        "Generator test protocol completed. All steps followed in correct order.", "wait")})

    return {"conversations": convs}


# ============================================================================
# MAIN
# ============================================================================

def main():
    all_episodes = []

    # Load seed data
    seed_path = Path("./dc_ops_sft_data.jsonl")
    if seed_path.exists():
        with open(seed_path) as f:
            seed_data = [json.loads(line) for line in f]
        all_episodes.extend(seed_data)
        print(f"Loaded {len(seed_data)} seed episodes")

    # Generate A1 variations
    print("Generating A1 (Setpoint Optimization) variations...")
    for fac in ["default", "small", "large"]:
        for sp in [20, 21, 22, 23, 24, 25, 26]:
            for fan in [60, 65, 70, 75, 80]:
                all_episodes.append(gen_a1_parametric(fac, sp, fan))
    print(f"  A1 episodes: {3*7*5} = 105")

    # Generate A2 variations
    print("Generating A2 (Thermal Event) variations...")
    for fac in ["default", "small", "large"]:
        cracs = FACILITIES[fac]["cracs"]
        for failed in cracs:
            for _ in range(3):  # 3 random variations per CRAC
                ep = gen_a2_parametric(fac, failed)
                if ep:
                    all_episodes.append(ep)
    a2_count = sum(1 for f in ["default","small","large"] for c in FACILITIES[f]["cracs"]) * 3
    print(f"  A2 episodes: ~{a2_count}")

    # Generate A4 variations
    print("Generating A4 (CRAC Cascade) variations...")
    for fac in ["default", "large"]:
        for _ in range(10):
            all_episodes.append(gen_a4_parametric(fac))
    print(f"  A4 episodes: 20")

    # Generate B1 variations
    print("Generating B1 (UPS Alarm) variations...")
    for _ in range(30):
        all_episodes.append(gen_b1_parametric())
    print(f"  B1 episodes: 30")

    # Generate B3 variations
    print("Generating B3 (Generator Test) variations...")
    for _ in range(25):
        all_episodes.append(gen_b3_parametric())
    print(f"  B3 episodes: 25")

    # Generate B4 variations
    print("Generating B4 (Power Failure) variations...")
    for fac in ["default", "small", "large"]:
        for _ in range(15):
            all_episodes.append(gen_b4_parametric(fac))
    print(f"  B4 episodes: 45")

    # Shuffle
    random.shuffle(all_episodes)

    # Write output
    output_path = Path("./dc_ops_sft_augmented.jsonl")
    with open(output_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"TOTAL EPISODES: {len(all_episodes)}")
    print(f"Output: {output_path}")

    # Stats
    total_turns = sum(len(ep["conversations"]) for ep in all_episodes)
    agent_turns = sum(1 for ep in all_episodes for t in ep["conversations"] if t["from"] == "gpt")
    print(f"Total conversation turns: {total_turns}")
    print(f"Agent turns (training targets): {agent_turns}")

    # Command coverage
    cmd_counts = {}
    for ep in all_episodes:
        for t in ep["conversations"]:
            if t["from"] == "gpt" and "<command>" in t["value"]:
                try:
                    cmd = t["value"].split("<command>\n")[1].split("\n</command>")[0].strip()
                    cmd_name = cmd.split()[0]
                    cmd_counts[cmd_name] = cmd_counts.get(cmd_name, 0) + 1
                except:
                    pass
    print(f"\nCommand distribution:")
    for cmd, count in sorted(cmd_counts.items(), key=lambda x: -x[1]):
        print(f"  {cmd:<25} {count:>5}")

    # File size
    import os
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"\nFile size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
