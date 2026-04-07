# DC-Ops: Scenario Demo Plans

> **All demos verified with 256/256 tests passing. Every plan resolves successfully in 8–10 steps.**

---

## Overview

| # | Scenario | Facility | Steps | Reward | Key Skill Tested |
|---|----------|----------|-------|--------|-----------------|
| 1 | A1 — Setpoint Optimization | Default | **9** | +0.324 | Energy efficiency tuning (PUE optimization) |
| 2 | A2 — Thermal Event | Default | **8** | +0.873 | Single-failure incident response |
| 3 | A2 — Thermal Event | Large | **8** | +0.831 | Multi-zone awareness, H1 isolation |
| 4 | A4 — CRAC Cascade | Default | **8** | +1.230 | Multi-failure triage + load shedding |
| 5 | A4 — CRAC Cascade | Large | **8** | +1.150 | Multi-zone cascade with H1 envelope |
| 6 | B1 — UPS Alarm | Default | **8** | +0.512 | Power chain audit, alarm management |
| 7 | B3 — Generator Test | Default | **10** | +0.567 | 5-step protocol compliance |
| 8 | B4 — Power Failure | Default | **8** | +0.934 | Battery management, gen startup |
| 9 | B4 — Power Failure | Small | **8** | +0.948 | Aggressive load shedding at scale |

All demos resolve in **8–10 steps** and demonstrate proper operational procedure: **assess → diagnose → compensate → verify → resolve**.

---

## Demo 1: A1 — Cooling Setpoint Optimization

**Facility:** Default (2 zones, 160 kW, 4 CRACs)  
**Difficulty:** Easy | **Budget:** 10 steps | **Result:** Resolved in 9 steps, cumulative reward **+0.324**

### Context

All four CRACs are set to 15°C — far below what the servers need. This wastes energy: the compressors run hard, fans blow at 100%, and PUE sits at 1.87. ASHRAE A2 class allows inlet temps up to 27°C. The agent must raise setpoints and reduce fan speeds to approach the PUE target of 1.6 without overheating.

### Step-by-Step

| Step | Command | Reward | Cumul | PUE | Inlets (A/B) | Reasoning |
|------|---------|--------|-------|-----|-------------|-----------|
| 1 | `check_status` | +0.131 | +0.131 | 1.87 | 17.1 / 17.1 | **Procedure rule:** must check status before adjusting setpoints (+0.2 bonus). Establishes baseline — all CRACs at 15°C, PUE 1.87, inlet temps far below recommended range. |
| 2 | `adjust_setpoint CRAC-1 24` | +0.047 | +0.178 | 1.80 | 17.7 / 17.1 | Raise from 15→24°C. CRAC-1 compressor works less → immediate PUE drop. Supply temp lags ~30s behind the setpoint due to first-order thermal lag. |
| 3 | `adjust_setpoint CRAC-2 24` | +0.039 | +0.217 | 1.76 | 19.2 / 17.0 | Same for CRAC-2. PUE continues falling. Zone A inlets rising as CRACs supply warmer air — still safely below 27°C limit. |
| 4 | `adjust_setpoint CRAC-3 24` | +0.038 | +0.255 | 1.71 | 20.7 / 17.6 | Zone B beginning to respond. The thermal mass (11.1 kJ/K per server) causes gradual warming — the RC network models this physical inertia. |
| 5 | `adjust_setpoint CRAC-4 24` | +0.025 | +0.280 | 1.69 | 21.9 / 19.1 | All CRACs at 24°C. PUE at 1.69 — still above 1.6 target. Diminishing returns from setpoint-only optimization; fan power reduction is the next lever. |
| 6 | `set_fan_speed CRAC-1 70` | −0.019 | +0.261 | 1.68 | 22.8 / 20.6 | Reduce fan from 100→70%. Fan power follows **cubic affinity law**: P = P_rated × (speed%)³. At 70%, fan power = 34% of rated — a 66% power reduction per CRAC fan. |
| 7 | `set_fan_speed CRAC-2 70` | −0.017 | +0.244 | 1.66 | 23.4 / 21.9 | PUE 1.66 — getting close. Inlet temps at 23.4°C — still 3.6°C below ASHRAE A2 recommended max (27°C). |
| 8 | `set_fan_speed CRAC-3 70` | −0.012 | +0.232 | 1.63 | 23.9 / 22.7 | Almost there. The diminishing negative rewards show the system approaching equilibrium. |
| 9 | `set_fan_speed CRAC-4 70` | **+0.092** | **+0.324** | **1.60** | 24.3 / 23.3 | **✓ RESOLVED.** PUE hits target (≤1.6). All inlets within ASHRAE recommended range (18–27°C). Speed bonus: (10−9)/10 = +0.1. |

### Why This Works

The agent follows a two-phase optimization strategy:

1. **Phase 1 (steps 2–5): Setpoint adjustment.** Raising setpoints from 15→24°C reduces compressor load. PUE drops 1.87→1.69 (10% improvement). But setpoint changes alone can't reach 1.6 because fan power is a significant fraction of total cooling power.

2. **Phase 2 (steps 6–9): Fan speed reduction.** The cubic fan law means reducing speed from 100→70% cuts fan power by 66%. This pushes PUE from 1.69→1.60, crossing the target.

---

## Demo 2: A2 — Thermal Event Response (Default Facility)

**Facility:** Default (2 zones, 160 kW, 4 CRACs)  
**Difficulty:** Medium | **Budget:** 15 steps | **Result:** Resolved in 8 steps, cumulative reward **+0.873**

### Context

CRAC-3 compressor has failed. With only 3 of 4 CRACs operational, cooling capacity is reduced. The default facility has N+1 redundancy, so temperatures won't spike catastrophically, but the agent must diagnose the fault and compensate to ensure long-term stability.

### Step-by-Step

| Step | Command | Reward | Cumul | Inlets (A/B) | Reasoning |
|------|---------|--------|-------|-------------|-----------|
| 1 | `check_status` | +0.204 | +0.204 | 19.8 / 20.0 | Assess the situation. Dashboard shows CRAC-3 as "!! COMPRESSOR" with no supply temp, no airflow, 0 kW. Zone B slightly warmer — the lost CRAC served zone B. |
| 2 | `diagnose CRAC-3` | +0.054 | +0.258 | 19.8 / 20.2 | **Unlocks the resolution gate.** Diagnostic report confirms "FAULT DETECTED: compressor." Procedure rule rewards diagnosing before adjusting (+0.3 bonus on subsequent setpoint changes). |
| 3 | `diagnose CRAC-1` | −0.021 | +0.237 | 19.8 / 20.4 | Verify remaining CRACs are healthy. Report: "No faults detected. Unit operating normally." |
| 4 | `adjust_setpoint CRAC-1 16` | +0.034 | +0.270 | 19.6 / 20.5 | Lower setpoint 18→16°C on a surviving CRAC. Increases cooling output. Earns procedure bonus (diagnosis was done first). |
| 5 | `adjust_setpoint CRAC-2 16` | +0.034 | +0.304 | 19.3 / 20.6 | Both zone A CRACs now overcooling to compensate. Zone A cools to 19.3°C while zone B stabilizes around 20.6°C. |
| 6 | `set_fan_speed CRAC-1 100` | +0.034 | +0.338 | 19.0 / 20.8 | Ensure max airflow on surviving CRACs. |
| 7 | `set_fan_speed CRAC-2 100` | +0.034 | +0.372 | 18.7 / 20.8 | Zone B stabilizing at ~20.8°C — well within ASHRAE A2 recommended range (27°C max). |
| 8 | `set_fan_speed CRAC-4 100` | **+0.501** | **+0.873** | 18.5 / 20.9 | **✓ RESOLVED.** All zones within recommended range for 2+ consecutive stable steps. Speed bonus: (15−8)/15 = +0.467. |

---

## Demo 3: A2 — Thermal Event Response (Large Facility)

**Facility:** Large (4 zones, 600 kW, 8 CRACs including H1 zone)  
**Difficulty:** Medium | **Budget:** 15 steps | **Result:** Resolved in 8 steps, cumulative reward **+0.831**

### Context

Same CRAC-3 failure, but in a larger facility with 4 zones including an H1 (high-density GPU) zone. The H1 zone has a tighter thermal envelope (recommended max 22°C vs 27°C for A2).

### Step-by-Step

| Step | Command | Reward | Cumul | Inlets (A/B/C/D) | Reasoning |
|------|---------|--------|-------|-------------------|-----------|
| 1 | `check_status` | +0.207 | +0.207 | 20.0 / 20.8 / 19.2 / 19.7 | 4-zone dashboard. Zone B already 0.8°C warmer. Zone C (H1) at 19.2°C — only 2.8°C below its 22°C recommended max. |
| 2 | `diagnose CRAC-3` | +0.057 | +0.263 | 20.0 / 21.6 / 19.2 / 19.7 | FAULT: compressor. Zone B rising +0.8°C/step. Zone C stable — dedicated CRACs unaffected. |
| 3 | `diagnose CRAC-1` | −0.019 | +0.245 | 20.0 / 22.3 / 19.2 / 19.6 | Confirm CRAC-1 healthy. |
| 4 | `diagnose CRAC-2` | −0.019 | +0.225 | 20.0 / 23.0 / 19.2 / 19.6 | Confirm CRAC-2 healthy. Thorough diagnosis of all CRACs in the affected zone pair. |
| 5 | `adjust_setpoint CRAC-2 16` | +0.035 | +0.261 | 19.9 / 23.6 / 19.2 / 19.6 | Lower setpoint on surviving zone B CRAC. Procedure bonus earned (diagnosed first). |
| 6 | `adjust_setpoint CRAC-4 16` | +0.035 | +0.296 | 19.7 / 24.0 / 19.2 / 19.6 | Zone B rate of rise slowing — compensation taking effect. |
| 7 | `set_fan_speed CRAC-2 100` | +0.035 | +0.330 | 19.6 / 24.3 / 19.2 / 19.6 | Max airflow. Zone B at 24.3°C — 2.7°C below ASHRAE A2 recommended max. |
| 8 | `set_fan_speed CRAC-4 100` | **+0.501** | **+0.831** | 19.5 / 24.6 / 19.2 / 19.6 | **✓ RESOLVED.** Zone B stabilized at 24.6°C. H1 zone completely unaffected (19.2°C). Speed bonus: +0.467. |

### Default vs Large Comparison

| Metric | Default | Large |
|--------|---------|-------|
| Max inlet temp | 20.9°C | 24.6°C |
| H1 zone impact | N/A | None (19.2°C) |
| Cumulative reward | +0.873 | +0.831 |

---

## Demo 4: A4 — CRAC Failure Cascade (Default Facility)

**Facility:** Default (2 zones, 160 kW, 4 CRACs)  
**Difficulty:** Hard | **Budget:** 20 steps | **Result:** Resolved in 8 steps, cumulative reward **+1.230**

### Context

Two CRACs fail simultaneously: CRAC-1 (compressor) and CRAC-3 (fan). Only CRAC-2 and CRAC-4 remain operational — 50% cooling capacity lost. The agent must diagnose both failures, aggressively compensate, and consider load shedding.

### Step-by-Step

| Step | Command | Reward | Cumul | Inlets (A/B) | Reasoning |
|------|---------|--------|-------|-------------|-----------|
| 1 | `check_status` | +0.212 | +0.212 | 19.9 / 19.9 | Dashboard shows two red-flagged CRACs. 50% cooling capacity lost. |
| 2 | `diagnose CRAC-1` | +0.062 | +0.274 | 20.0 / 20.0 | "FAULT DETECTED: compressor." First of two required diagnoses. |
| 3 | `diagnose CRAC-3` | +0.027 | +0.301 | 20.1 / 20.1 | "FAULT DETECTED: fan." Both failures identified → resolution gate unlocked. Extra +0.2 procedure bonus for diagnosing both units. |
| 4 | `adjust_setpoint CRAC-2 16` | +0.062 | +0.363 | 20.2 / 20.2 | Lower surviving CRAC setpoint. Procedure bonus (+0.2) earned because diagnosis preceded this. |
| 5 | `adjust_setpoint CRAC-4 16` | +0.062 | +0.425 | 20.2 / 20.3 | Both survivors at aggressive 16°C setpoints. |
| 6 | `set_fan_speed CRAC-2 100` | +0.062 | +0.487 | 20.1 / 20.2 | Confirm max airflow. Temps stabilizing. |
| 7 | `set_fan_speed CRAC-4 100` | +0.062 | +0.549 | 20.1 / 20.2 | Temps flat at ~20.1°C. Stable for consecutive steps. |
| 8 | `set_rack_load B-05 4` | **+0.681** | **+1.230** | 20.0 / 20.1 | **✓ RESOLVED.** Load shed on hottest rack as precaution. Speed bonus: (20−8)/20 = +0.600. |

### Why Load Shedding Matters

Reducing rack B-05 from 8 kW to 4 kW:
- Provides additional thermal margin (less heat to extract)
- Demonstrates workload migration capability
- Earns action quality bonus (+0.2 for interventions)

---

## Demo 5: A4 — CRAC Failure Cascade (Large Facility)

**Facility:** Large (4 zones, 600 kW, 8 CRACs)  
**Difficulty:** Hard | **Budget:** 20 steps | **Result:** Resolved in 8 steps, cumulative reward **+1.150**

### Step-by-Step

| Step | Command | Reward | Cumul | Inlets (A/B/C/D) | Reasoning |
|------|---------|--------|-------|-------------------|-----------|
| 1 | `check_status` | +0.210 | +0.210 | 20.4 / 20.4 / 19.2 / 19.7 | CRAC-1 and CRAC-3 down out of 8. Zones C/D (including H1) have their own CRACs. |
| 2 | `diagnose CRAC-1` | +0.059 | +0.269 | 20.8 / 20.8 / 19.2 / 19.7 | FAULT: compressor. Zone A/B rising ~0.4°C/step. |
| 3 | `diagnose CRAC-3` | +0.024 | +0.293 | 21.2 / 21.2 / 19.2 / 19.7 | FAULT: fan. Both diagnosed — resolution gate unlocked. |
| 4 | `diagnose CRAC-2` | +0.024 | +0.317 | 21.6 / 21.6 / 19.2 / 19.7 | Verify CRAC-2 healthy. With 2 failures, confirming survivor health is critical. |
| 5 | `adjust_setpoint CRAC-2 16` | +0.059 | +0.376 | 21.9 / 22.0 / 19.2 / 19.6 | Compensate. Zone B at 22.0°C — 13°C below ASHRAE A2 allowable max (35°C). |
| 6 | `adjust_setpoint CRAC-4 16` | +0.058 | +0.434 | 22.2 / 22.3 / 19.2 / 19.6 | Rate of rise slowing. H1 zone completely unaffected. |
| 7 | `set_fan_speed CRAC-2 100` | +0.058 | +0.492 | 22.5 / 22.5 / 19.2 / 19.6 | Max airflow on survivor. Zone B stabilizing. |
| 8 | `set_fan_speed CRAC-4 100` | **+0.658** | **+1.150** | 22.7 / 22.8 / 19.2 / 19.6 | **✓ RESOLVED.** All zones within allowable. Speed bonus: +0.600. |

### Default vs Large Comparison

| Metric | Default | Large |
|--------|---------|-------|
| Final zone B inlet | 20.1°C | 22.8°C |
| H1 zone impact | N/A | None (19.2°C) |
| Cumulative reward | +1.230 | +1.150 |

---

## Demo 6: B1 — UPS Alarm Response

**Facility:** Default (2 zones, 160 kW)  
**Difficulty:** Medium | **Budget:** 10 steps | **Result:** Resolved in 8 steps, cumulative reward **+0.512**

### Context

A brief utility dip caused UPS-1 to transfer to battery. Utility has been restored and UPS switched back to double-conversion mode, but the alarm persists. The agent must investigate the entire power chain and acknowledge the alarm.

### Step-by-Step

| Step | Command | Reward | Cumul | Reasoning |
|------|---------|--------|-------|-----------|
| 1 | `check_status` | −0.007 | −0.007 | Baseline assessment. Dashboard shows utility NORMAL, generator OFF, ATS on UTILITY. |
| 2 | `diagnose UPS-1` | +0.143 | +0.137 | **Key step.** Report: mode=double_conversion, SOC=86%. Resolution gate requires this diagnosis. |
| 3 | `diagnose UPS-2` | −0.007 | +0.130 | Verify redundant UPS. mode=double_conversion, SOC=87%. |
| 4 | `diagnose GEN-1` | −0.007 | +0.123 | Generator in standby and ready — confirming readiness is proper procedure. |
| 5 | `diagnose PDU-A-01` | −0.007 | +0.117 | Verify power distribution is intact. |
| 6 | `check_status` | −0.007 | +0.110 | Re-verify overall state before closing incident. |
| 7 | `diagnose PDU-B-01` | −0.007 | +0.103 | Complete the power chain audit. |
| 8 | `acknowledge_alarm` | **+0.408** | **+0.512** | **✓ RESOLVED.** Alarm acknowledged after thorough investigation. Speed bonus: (10−8)/10 = +0.200. |

### Reward Structure Note

Steps 3–7 return −0.007 each because the delta-based progress metric doesn't change during investigation — only the final acknowledgment triggers progress completion. The cumulative reward is still positive (+0.512) thanks to the large resolution step reward.

---

## Demo 7: B3 — Generator Test Protocol

**Facility:** Default (2 zones, 160 kW)  
**Difficulty:** Easy | **Budget:** 15 steps | **Result:** Resolved in 10 steps, cumulative reward **+0.567**

### Context

Routine monthly generator test. No fault, no emergency — the agent must follow the 5-step protocol: check → start → verify → stop → acknowledge.

### Step-by-Step

| Step | Command | Reward | Cumul | Generator State | Reasoning |
|------|---------|--------|-------|----------------|-----------|
| 1 | `check_status` | −0.007 | −0.007 | OFF | Baseline. All systems normal. |
| 2 | `diagnose GEN-1` | −0.007 | −0.013 | off | Pre-test inspection — verify before starting. |
| 3 | `start_generator` | +0.113 | +0.100 | START_DELAY→CRANKING | Generator start sequence initiated. |
| 4 | `wait` | −0.022 | +0.078 | READY/LOADED | Let the generator complete warmup. |
| 5 | `diagnose GEN-1` | +0.043 | +0.122 | ready | **Critical verification.** Confirms generator running properly. |
| 6 | `check_status` | −0.007 | +0.115 | LOADED | Full dashboard confirms generator loaded. |
| 7 | `stop_generator` | +0.113 | +0.228 | COOLDOWN | Initiate cooldown (300s for turbocharger). |
| 8 | `wait` | −0.022 | +0.207 | COOLDOWN | Allow cooldown to proceed. |
| 9 | `diagnose GEN-1` | −0.032 | +0.175 | cooldown | Post-shutdown inspection. |
| 10 | `acknowledge_alarm` | **+0.392** | **+0.567** | COOLDOWN | **✓ RESOLVED.** Protocol complete: started ✓, verified ✓, stopped ✓, acknowledged ✓. Speed bonus: (15−10)/15 = +0.333. |

### Protocol Enforcement

B3 tracks four internal flags that must be set in order:

1. `_started` — `start_generator` issued
2. `_verified` — `diagnose GEN-1` while generator is running
3. `_stopped` — `stop_generator` (only if started + verified)
4. `_completed` — `acknowledge_alarm` (only if stopped)

The agent **cannot skip steps** — issuing `stop_generator` before `diagnose GEN-1` won't set `_stopped`.

---

## Demo 8: B4 — Power Failure Cascade (Default Facility)

**Facility:** Default (2 zones, 160 kW)  
**Difficulty:** Hard | **Budget:** 20 steps | **Result:** Resolved in 8 steps, cumulative reward **+0.934**

### Context

Total utility power loss. UPS batteries bridging while generator starts. Generator warmup extended (15s vs default 8s). Agent must manage battery life, consider load shedding, and verify generator operation.

### Step-by-Step

| Step | Command | Reward | Cumul | Key Metrics | Reasoning |
|------|---------|--------|-------|-------------|-----------|
| 1 | `check_status` | +0.108 | +0.108 | UPS on battery, SOC ~97% | Utility LOST. ATS initiating transfer. Generator auto-starting. |
| 2 | `diagnose UPS-1` | +0.131 | +0.239 | mode=on_battery, SOC=95% | **Resolution gate unlocked.** Battery draining ~2%/step at 160 kW load. |
| 3 | `diagnose UPS-2` | +0.078 | +0.317 | mode=on_battery, SOC=90% | Redundant UPS also on battery. |
| 4 | `start_generator` | −0.007 | +0.310 | Gen: CRANKING→READY | Generator already auto-starting — hence slight negative reward for redundant command. |
| 5 | `set_rack_load A-05 4` | +0.062 | +0.371 | IT load: 156 kW | Shed 4 kW to extend battery life. |
| 6 | `set_rack_load B-05 4` | +0.054 | +0.425 | IT load: 152 kW | Total shed: 8 kW (5% of IT load). Conservative but effective. |
| 7 | `wait` | −0.052 | +0.373 | Gen: LOADED, ATS: GEN | Generator online. UPS batteries begin recharging. |
| 8 | `diagnose GEN-1` | **+0.561** | **+0.934** | state=loaded | **✓ RESOLVED.** All conditions met: gen loaded, temps OK, SOC >10%. Speed bonus: (20−8)/20 = +0.600. |

### Battery SOC Timeline

| Step | Sim Time | SOC (UPS-1) | Event |
|------|----------|-------------|-------|
| 0 | 0 min | 97% | Utility lost |
| 2 | 2 min | 95% | Diagnosed |
| 4 | 4 min | ~90% | Gen starting |
| 6 | 6 min | ~87% | Load shed |
| 7 | 7 min | ~88% | Gen loaded, recharging begins |
| 8 | 8 min | ~89% | Resolved |

---

## Demo 9: B4 — Power Failure Cascade (Small Facility)

**Facility:** Small (1 zone, 80 kW, 2 CRACs)  
**Difficulty:** Hard | **Budget:** 20 steps | **Result:** Resolved in 8 steps, cumulative reward **+0.948**

### Step-by-Step

| Step | Command | Reward | Cumul | Key Metrics | Reasoning |
|------|---------|--------|-------|-------------|-----------|
| 1 | `check_status` | +0.111 | +0.111 | 1 zone, UPS on battery | Smaller facility: 80 kW IT load, 1 zone, 2 CRACs. Less redundancy. |
| 2 | `diagnose UPS-1` | +0.133 | +0.244 | SOC=91% | Battery draining faster relative to capacity. Gate unlocked. |
| 3 | `start_generator` | +0.073 | +0.317 | Gen starting | Explicit start command. |
| 4 | `set_rack_load A-05 4` | +0.066 | +0.383 | IT: 76 kW | 5% load reduction. |
| 5 | `set_rack_load A-04 4` | +0.056 | +0.438 | IT: 72 kW | 10% load reduction. |
| 6 | `set_rack_load A-03 4` | +0.042 | +0.481 | IT: 68 kW | 15% total load shed — more aggressive for smaller facility. |
| 7 | `wait` | −0.069 | +0.412 | Gen LOADED | Generator online. Battery recharging. |
| 8 | `diagnose GEN-1` | **+0.537** | **+0.948** | state=loaded | **✓ RESOLVED.** Speed bonus: +0.600. |

### Small vs Default Comparison

| Metric | Default (160 kW) | Small (80 kW) |
|--------|-----------------|---------------|
| Racks shed | 2 (8 kW, 5%) | 3 (12 kW, 15%) |
| Cumulative reward | +0.934 | +0.948 |

The small facility earns slightly higher reward due to more aggressive proportional load shedding, producing a stronger positive signal from the power safety component.

---

## Resolution Gate Design

Each affected scenario requires the agent to actually **do something** before resolution:

| Scenario | Diagnosis Gate | Min Steps |
|----------|---------------|-----------|
| A2 | Must `diagnose CRAC-3` | ≥ 8 steps |
| A4 | Must `diagnose CRAC-1` AND `diagnose CRAC-3` | ≥ 8 steps |
| B4 | Must `diagnose UPS-*` | ≥ 8 steps |

Without these gates, scenarios A2, A4, and B4 would auto-resolve within 2–3 steps of passive `wait` commands due to:

- N+1 cooling redundancy handling CRAC failures naturally
- Generator auto-start/load completing within one 60-second agent step
- `resolved` only checking `_stable_count >= 2` with no gate on agent behavior

**Reward ordering validates the design:** fast diagnosis > late diagnosis > no diagnosis (never resolves).
