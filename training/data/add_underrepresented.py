#!/usr/bin/env python3
"""
DC-Ops Additional Data: Underrepresented commands + edge cases.
Appends to dc_ops_sft_augmented.jsonl
"""

import json
import random
from pathlib import Path

random.seed(456)

SYSTEM_PROMPT = open("./dc_ops_sft_augmented.jsonl").readline()
# Actually just redefine it
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


def fmt(reasoning, command):
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<command>\n{command}\n</command>"


def gen_ups_mode_episodes():
    """Episodes specifically teaching set_ups_mode command."""
    episodes = []

    modes = [
        ("eco", "ECO mode uses a bypass path when utility is clean, reducing conversion losses. Efficiency improves from ~94% to ~98%."),
        ("double_conversion", "Double conversion provides the highest protection — power is always conditioned through the inverter. Standard for critical loads."),
        ("bypass", "Bypass mode sends utility power directly to the load, bypassing the UPS for maintenance. Only safe when utility is stable."),
    ]

    for ups_id in ["UPS-1", "UPS-2"]:
        for mode, explanation in modes:
            for context in [
                f"Utility is stable and clean. Switch {ups_id} to {mode} mode.",
                f"After verifying power quality, change {ups_id} to {mode}.",
                f"Operations requests {ups_id} be set to {mode} mode for the current conditions.",
            ]:
                episodes.append({"conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": f"**Action Result:** {context}\n\n**Steps Remaining:** 8\n\n[Dashboard showing normal operations]"},
                    {"from": "gpt", "value": fmt(
                        f"{explanation} Setting {ups_id} to {mode} mode as requested.",
                        f"set_ups_mode {ups_id} {mode}"
                    )}
                ]})

    return episodes


def gen_refuel_episodes():
    """Episodes specifically teaching refuel_generator command."""
    episodes = []

    scenarios = [
        ("Generator fuel level is at 200L out of 1000L after extended outage. Refuel to full.",
         "The generator has used 800L during the outage. I need to refuel it to ensure it's ready for the next emergency.",
         "refuel_generator"),
        ("Generator has 500L remaining. Add 300 liters of diesel.",
         "Adding 300L to bring the tank up. This is enough for approximately 7 more hours of operation.",
         "refuel_generator 300"),
        ("Post-test: generator used 3L during monthly test. Top it off.",
         "Minimal fuel used during the test. Refueling to full tank to maintain readiness.",
         "refuel_generator"),
        ("Emergency reserves low at 150L. Add 500 liters immediately.",
         "Critical to refuel — at 150L we only have about 3.5 hours of runtime. Adding 500L.",
         "refuel_generator 500"),
        ("Routine maintenance: ensure generator has full fuel tank.",
         "Part of routine maintenance — verifying and topping off the generator fuel supply.",
         "refuel_generator"),
        ("After extended generator run, fuel is at 400L. Refuel to maximum capacity.",
         "Extended power outage depleted fuel significantly. Refueling to full 1000L tank.",
         "refuel_generator"),
    ]

    for context, reasoning, cmd in scenarios:
        episodes.append({"conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"**Action Result:** {context}\n\n**Steps Remaining:** 5\n\n[Dashboard showing generator state]"},
            {"from": "gpt", "value": fmt(reasoning, cmd)}
        ]})

    return episodes


def gen_start_stop_crac_episodes():
    """Episodes teaching start_crac and stop_crac commands."""
    episodes = []

    # Start CRAC scenarios
    start_scenarios = [
        ("CRAC-2 is in standby. Zone A temperatures are climbing. Start it up.",
         "Zone A needs additional cooling capacity. CRAC-2 is in standby and ready — starting it will provide more airflow and cooling.",
         "start_crac CRAC-2"),
        ("Maintenance on CRAC-4 is complete. Return it to service.",
         "CRAC-4 maintenance is done. Starting it to restore full cooling redundancy.",
         "start_crac CRAC-4"),
        ("Emergency: zone B has no active CRACs. Start CRAC-3 immediately.",
         "Critical situation — zone B has lost all cooling. CRAC-3 must be started immediately to prevent thermal runaway.",
         "start_crac CRAC-3"),
        ("Standby CRAC-1 should be activated to prepare for a planned IT load increase.",
         "Pre-emptively starting CRAC-1 ahead of the scheduled load increase to ensure adequate cooling capacity is available.",
         "start_crac CRAC-1"),
    ]

    for context, reasoning, cmd in start_scenarios:
        for _ in range(3):
            episodes.append({"conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": f"**Action Result:** {context}\n\n**Steps Remaining:** 10\n\n[Dashboard showing CRAC in standby]"},
                {"from": "gpt", "value": fmt(reasoning, cmd)}
            ]})

    # Stop CRAC scenarios
    stop_scenarios = [
        ("CRAC-1 needs maintenance. All other CRACs are running. Put CRAC-1 in standby.",
         "With 3 other CRACs running, we have sufficient N+1 cooling. Placing CRAC-1 in standby for maintenance.",
         "stop_crac CRAC-1"),
        ("Energy optimization: zone A is overcooled with two CRACs. Put CRAC-2 in standby.",
         "Zone A has excess cooling capacity. Stopping CRAC-2 will reduce energy consumption while CRAC-1 alone can handle the load.",
         "stop_crac CRAC-2"),
        ("Scheduled maintenance window for CRAC-3. Zone B has redundant cooling.",
         "CRAC-4 provides sufficient cooling for zone B alone. Placing CRAC-3 in standby for the maintenance window.",
         "stop_crac CRAC-3"),
        ("CRAC-4 is making unusual noise. Put it in standby for inspection.",
         "Abnormal noise from CRAC-4 could indicate an impending failure. Better to stop it proactively and inspect than risk a fault.",
         "stop_crac CRAC-4"),
    ]

    for context, reasoning, cmd in stop_scenarios:
        for _ in range(3):
            episodes.append({"conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": f"**Action Result:** {context}\n\n**Steps Remaining:** 10\n\n[Dashboard showing CRAC running]"},
                {"from": "gpt", "value": fmt(reasoning, cmd)}
            ]})

    return episodes


def gen_escalation_episodes():
    """Episodes teaching when to escalate (and when NOT to)."""
    episodes = []

    # When to escalate
    escalate_cases = [
        "All CRACs have failed. Temperatures are at 34°C and rising. No cooling capacity available. This is beyond my ability to resolve with available commands.",
        "Generator failed to start after 3 attempts. UPS battery at 8% and dropping. Utility still down. This needs a senior engineer with physical access.",
        "Multiple cascading failures across thermal and power systems simultaneously. The situation is too complex for the available action space.",
    ]

    for case in escalate_cases:
        episodes.append({"conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"**Action Result:** {case}\n\n**Steps Remaining:** 3\n\n[Dashboard showing critical failures]"},
            {"from": "gpt", "value": fmt(
                "This situation has exceeded what I can handle with the available commands. Multiple critical systems have failed and I have no remaining actions that can stabilize the datacenter. I need to escalate to a senior engineer who can take physical action.",
                "escalate"
            )}
        ]})

    # When NOT to escalate (teach the model to try before escalating)
    no_escalate_cases = [
        ("CRAC-3 has failed but 3 others are running. Should I escalate?",
         "No — I should NOT escalate yet. With 3 remaining CRACs and N+1 redundancy, I can compensate by lowering setpoints and raising fan speeds on the survivors. Escalation is a last resort. Let me diagnose and compensate first.",
         "diagnose CRAC-3"),
        ("Generator hasn't started yet but UPS SOC is at 75%. Should I escalate?",
         "No escalation needed. UPS SOC at 75% gives us plenty of time. The generator may still start. I should try starting it manually and shed some load to extend battery life.",
         "start_generator"),
        ("Zone B temperature reached 25°C. Should I escalate?",
         "25°C is still within ASHRAE A2 recommended range (max 27°C). I have room to act. Let me lower setpoints on zone B CRACs to bring the temperature down.",
         "adjust_setpoint CRAC-4 16"),
    ]

    for context, reasoning, cmd in no_escalate_cases:
        for _ in range(3):
            episodes.append({"conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": f"**Action Result:** {context}\n\n**Steps Remaining:** 12\n\n[Dashboard]"},
                {"from": "gpt", "value": fmt(reasoning, cmd)}
            ]})

    return episodes


def gen_load_shedding_episodes():
    """More load shedding episodes with various rack IDs."""
    episodes = []

    rack_ids = ["A-01", "A-02", "A-03", "A-04", "A-05", "A-06", "A-07", "A-08", "A-09", "A-10",
                "B-01", "B-02", "B-03", "B-04", "B-05", "B-06", "B-07", "B-08", "B-09", "B-10"]
    target_kws = [2, 3, 4, 5, 6]

    contexts = [
        "Battery draining fast. Need to reduce IT load to extend UPS runtime.",
        "Two CRACs down. Must reduce heat generation to prevent thermal runaway.",
        "Cooling capacity insufficient. Shedding non-critical workloads.",
        "Emergency load reduction required during power transition.",
        "Proactive load shedding ahead of maintenance window.",
    ]

    reasonings = [
        "Reducing rack {rack} from 8 kW to {kw} kW. This immediately cuts heat output and power draw, giving us more thermal and power margin.",
        "Load shedding on {rack} to {kw} kW. The workload migration will reduce total IT load, helping stabilize the situation.",
        "Rack {rack} being shed to {kw} kW as a precautionary measure. Non-critical workloads can be deferred.",
        "Migrating workload from {rack}, dropping to {kw} kW. This extends battery life and reduces cooling demand.",
    ]

    for _ in range(50):
        rack = random.choice(rack_ids)
        kw = random.choice(target_kws)
        episodes.append({"conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"**Action Result:** {random.choice(contexts)}\n\n**Steps Remaining:** {random.randint(5,15)}\n\n[Dashboard showing emergency]"},
            {"from": "gpt", "value": fmt(
                random.choice(reasonings).format(rack=rack, kw=kw),
                f"set_rack_load {rack} {kw}"
            )}
        ]})

    return episodes


def gen_diagnose_all_targets():
    """Diagnose episodes covering ALL valid target IDs."""
    episodes = []

    targets = {
        "CRAC-1": "cooling unit in zone A",
        "CRAC-2": "cooling unit in zone A",
        "CRAC-3": "cooling unit in zone B",
        "CRAC-4": "cooling unit in zone B",
        "CRAC-5": "cooling unit in zone C (large facility)",
        "CRAC-6": "cooling unit in zone C (large facility)",
        "CRAC-7": "cooling unit in zone D (large facility)",
        "CRAC-8": "cooling unit in zone D (large facility)",
        "UPS-1": "primary uninterruptible power supply",
        "UPS-2": "redundant uninterruptible power supply",
        "GEN-1": "diesel backup generator",
        "PDU-A-01": "power distribution unit serving rack A-01",
        "PDU-A-05": "power distribution unit serving rack A-05",
        "PDU-A-10": "power distribution unit serving rack A-10",
        "PDU-B-01": "power distribution unit serving rack B-01",
        "PDU-B-05": "power distribution unit serving rack B-05",
        "PDU-B-10": "power distribution unit serving rack B-10",
    }

    for target, desc in targets.items():
        for context in [
            f"I need to inspect {target} ({desc}) for potential issues.",
            f"Checking {target} as part of routine diagnostics.",
            f"Verifying {target} is operating normally before relying on it.",
        ]:
            episodes.append({"conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": f"**Action Result:** {context}\n\n**Steps Remaining:** 10\n\n[Dashboard]"},
                {"from": "gpt", "value": fmt(
                    f"I need to run diagnostics on {target} ({desc}). This will give me detailed status information including any faults, operating parameters, and health indicators.",
                    f"diagnose {target}"
                )}
            ]})

    return episodes


def gen_multi_step_setpoint_fan():
    """Multi-step episodes combining setpoint and fan adjustments on various CRACs."""
    episodes = []

    crac_ids = ["CRAC-1", "CRAC-2", "CRAC-3", "CRAC-4"]
    setpoints = [16, 18, 20, 22, 24, 26]
    fans = [50, 60, 70, 80, 90, 100]

    for _ in range(40):
        crac = random.choice(crac_ids)
        sp = random.choice(setpoints)
        fan = random.choice(fans)
        convs = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"**Action Result:** Adjust {crac} for current conditions.\n\n**Steps Remaining:** 8\n\n[Dashboard]"},
            {"from": "gpt", "value": fmt(
                f"First adjusting {crac} setpoint to {sp}°C to set the target supply temperature.",
                f"adjust_setpoint {crac} {sp}"
            )},
            {"from": "human", "value": f"**Action Result:** Setpoint for {crac} adjusted to {sp:.1f}°C.\n\n**Steps Remaining:** 7\n\n[Dashboard]"},
            {"from": "gpt", "value": fmt(
                f"Setpoint adjusted. Now setting {crac} fan speed to {fan}% to control airflow volume.",
                f"set_fan_speed {crac} {fan}"
            )},
            {"from": "human", "value": f"**Action Result:** Fan speed for {crac} set to {fan}%.\n\n**Steps Remaining:** 6\n\n[Dashboard]"},
            {"from": "gpt", "value": fmt(
                f"{crac} is now configured with setpoint {sp}°C and fan speed {fan}%. Monitoring for temperature response.",
                "check_status"
            )},
        ]
        episodes.append({"conversations": convs})

    return episodes


def main():
    # Load existing data
    aug_path = Path("./dc_ops_sft_augmented.jsonl")
    with open(aug_path) as f:
        existing = [json.loads(line) for line in f]
    print(f"Existing episodes: {len(existing)}")

    new_episodes = []

    print("Generating UPS mode episodes...")
    ups_eps = gen_ups_mode_episodes()
    new_episodes.extend(ups_eps)
    print(f"  UPS mode: {len(ups_eps)}")

    print("Generating refuel episodes...")
    refuel_eps = gen_refuel_episodes()
    new_episodes.extend(refuel_eps)
    print(f"  Refuel: {len(refuel_eps)}")

    print("Generating start/stop CRAC episodes...")
    crac_eps = gen_start_stop_crac_episodes()
    new_episodes.extend(crac_eps)
    print(f"  Start/Stop CRAC: {len(crac_eps)}")

    print("Generating escalation episodes...")
    esc_eps = gen_escalation_episodes()
    new_episodes.extend(esc_eps)
    print(f"  Escalation: {len(esc_eps)}")

    print("Generating load shedding episodes...")
    load_eps = gen_load_shedding_episodes()
    new_episodes.extend(load_eps)
    print(f"  Load shedding: {len(load_eps)}")

    print("Generating diagnose-all-targets episodes...")
    diag_eps = gen_diagnose_all_targets()
    new_episodes.extend(diag_eps)
    print(f"  Diagnose targets: {len(diag_eps)}")

    print("Generating multi-step setpoint+fan episodes...")
    multi_eps = gen_multi_step_setpoint_fan()
    new_episodes.extend(multi_eps)
    print(f"  Multi-step: {len(multi_eps)}")

    # Combine
    all_episodes = existing + new_episodes
    random.shuffle(all_episodes)

    # Write final output
    final_path = Path("./dc_ops_sft_final.jsonl")
    with open(final_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"TOTAL EPISODES: {len(all_episodes)}")
    print(f"Output: {final_path}")

    total_turns = sum(len(ep["conversations"]) for ep in all_episodes)
    agent_turns = sum(1 for ep in all_episodes for t in ep["conversations"] if t["from"] == "gpt")
    print(f"Total conversation turns: {total_turns}")
    print(f"Agent turns (training targets): {agent_turns}")

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
        pct = count / sum(cmd_counts.values()) * 100
        print(f"  {cmd:<25} {count:>5}  ({pct:.1f}%)")

    import os
    size_mb = os.path.getsize(final_path) / (1024*1024)
    print(f"\nFile size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
