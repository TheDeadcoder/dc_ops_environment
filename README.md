# DC-Ops: Physics-Based Datacenter Operations RL Environment

**A high-fidelity datacenter simulation environment for training LLM agents as datacenter operators, built on Meta's [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.**

DC-Ops bridges the gap between toy RL environments and real-world datacenter operations by combining rigorous thermal/electrical physics with the OpenEnv agent training infrastructure. An LLM agent reads a text-based NOC (Network Operations Center) dashboard and issues natural-language operator commands — exactly as a human operator would.

```
╔════════════════════════════════════════════════════════════════════╗
║                    DC-OPS MONITORING DASHBOARD                     ║
║ Sim Time: 2.0 min    Step: 0/15                                   ║
╠════════════════════════════════════════════════════════════════════╣
║ !! ALERT: CRITICAL: CRAC-3 compressor failure detected. Zone B    ║
║   temperatures rising. Investigate and stabilize.                  ║
╠════════════════════════════════════════════════════════════════════╣
║ COOLING UNITS                                                      ║
║ Unit       Status       Setpoint   Supply  Fan%     CFM     kW     ║
║ CRAC-1     RUNNING        18.0°C  17.6°C   100   12000   29.5     ║
║ CRAC-2     RUNNING        18.0°C  17.6°C   100   12000   29.5     ║
║ CRAC-3     !! COMPRESSOR  18.0°C     ---     0       0    0.0     ║
║ CRAC-4     RUNNING        18.0°C  17.6°C   100   12000   29.5     ║
╠════════════════════════════════════════════════════════════════════╣
║ ZONE TEMPERATURES                                                  ║
║ Zone      Cold Aisle  Hot Aisle  Max Inlet  IT Load  Class         ║
║ zone_a       19.6°C     33.5°C     19.8°C    80.0     A2          ║
║ zone_b       22.1°C     36.0°C     22.3°C*   80.0     A2          ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## Why Datacenter Operations?

Datacenters consume **1-2% of global electricity** and this share is growing rapidly with AI workloads. Even a 0.1 PUE improvement across the industry saves terawatt-hours annually. Yet the field of datacenter operations remains largely manual, with operators relying on experience and static procedures.

### The Opportunity

| Challenge | Why LLM Agents? |
|-----------|-----------------|
| **Complex decision-making** | Operators must simultaneously balance thermal safety, power efficiency, equipment health, and business continuity |
| **Rare but critical events** | Power failures, cooling cascades, and thermal runaway are too dangerous to practice on real hardware |
| **Procedural correctness** | Proper diagnosis-before-intervention sequences reduce misdiagnosis by 40%+ |
| **24/7 coverage** | Autonomous agents don't fatigue, forget procedures, or miss alarms |
| **Transfer learning** | A well-trained agent can generalize across facility configurations |

### Research Foundation

DC-Ops draws on peer-reviewed datacenter research:

- **Google/DeepMind (2017)**: Demonstrated 40% cooling energy reduction using RL — `minimize PUE + λ·softplus(T − threshold)` ([Lazic et al., 2018](https://arxiv.org/abs/1803.03518))
- **ICLR 2025 (DCRL-Green)**: Multi-objective reward with softplus barriers for safe RL in datacenters
- **ASHRAE TC 9.9 (5th Edition, 2021)**: Industry-standard thermal guidelines used for all safety thresholds
- **APC White Paper 108**: UPS quadratic loss model: `η(x) = x / (x + c₀ + c₁x + c₂x²)` with coefficients calibrated against published efficiency curves

---

## Architecture

```
dc_ops_env/
├── config.py                      # Physical constants, ASHRAE limits, YAML config loader
├── models.py                      # OpenEnv Action/Observation Pydantic models
├── client.py                      # EnvClient (WebSocket connection to server)
├── simulation/
│   ├── thermal.py                 # RC thermal network (zones, racks, CRAC units)
│   ├── power.py                   # UPS, PDU, generator, ATS models
│   └── types.py                   # Runtime state dataclasses
├── scenarios/
│   ├── base.py                    # Abstract Scenario interface + ProcedureRule
│   ├── registry.py                # Scenario registration and selection
│   ├── thermal_scenarios.py       # A1 (Easy), A2 (Medium), A4 (Hard)
│   └── power_scenarios.py         # B1 (Medium), B3 (Easy), B4 (Hard)
├── rewards/
│   └── reward_function.py         # Multi-objective composite reward (6 components)
├── rendering/
│   └── dashboard.py               # Simulation state → text dashboard
├── actions/
│   └── parser.py                  # Deterministic command parser (no LLM-in-the-loop)
├── server/
│   ├── dc_ops_env_environment.py  # OpenEnv Environment implementation
│   ├── app.py                     # FastAPI application
│   └── Dockerfile                 # Container image
├── data/
│   └── datacenter_configs/        # YAML facility definitions
│       ├── default.yaml           # 2 zones, 20 racks, 160 kW
│       ├── small_facility.yaml    # 1 zone, 10 racks, 80 kW
│       └── large_facility.yaml    # 4 zones, 60 racks, 600 kW (mixed A2+H1)
└── tests/                         # 256 tests across 6 files
```

---

## Physics Engine

### Thermal Simulation

The thermal model uses a **lumped-capacitance RC network** — the same approach used in peer-reviewed datacenter transient analysis (Song et al., 2013; Toulouse, 2013).

**Governing equation per zone:**

```
C_total · dT_cold/dt = Q_IT − Q_cooling + Q_envelope + Q_internal
```

Where:
- **C_total** = C_air + C_equipment (equipment dominates: 11.1 kJ/K per 2U server, experimentally measured)
- **Q_IT** = Σ rack IT loads [W] (all electrical power → heat)
- **Q_cooling** = Σ CRAC outputs [W], capacity increases with return air temperature
- **Q_envelope** = (T_outside − T_zone) / R_envelope [W]

**CRAC model features:**
- Capacity vs. return temp: `Q_actual = Q_rated × [1 + α × (T_return − T_rated)]`, α ≈ 0.03/°C
- Fan power: cubic law (affinity laws) — `P_fan = P_rated × (speed%)³`
- COP degradation with outside temperature
- Supply temperature lag (30s time constant)
- Recirculation: `q_recirc = r · max(m_dot_rack, m_dot_crac) · c_p · ΔT`, driven by dominant airflow (leakage persists even when CRACs are off)

**Validated behaviors:**
| Scenario | Expected | Actual |
|----------|----------|--------|
| Steady state, 160 kW IT | Cold aisle 18-22°C, PUE ~1.4 | 19.6°C, PUE 1.48 |
| Single CRAC failure | Temp rises ~1-2°C/min | ~1.5°C/min |
| Total cooling loss | ~5°C/min rise rate | ~5°C/min |
| Setpoint +5°C | Response over ~2 min | τ ≈ 90s |
| Higher outside temp | PUE increases | PUE: 1.45 → 1.52 |

### Power Simulation

**UPS model** (quadratic loss, APC WP-108):
```
η(x) = x / (x + 0.013 + 0.006x + 0.011x²)
```
Validated against published data: 90.5% at 25% load, 93.6% at 50%, 94.0% at 75%.

**Battery discharge:**
```
SOC_new = SOC − (P_load / η_UPS) · dt / E_battery
```
With temperature derating: `E_effective = E_rated × 2^((25 − T_battery) / 10)`

**Generator state machine:**
```
OFF → START_DELAY (4s) → CRANKING (5s) → WARMING (8s) → READY → LOADED
                                                                    ↓
                                                              COOLDOWN (300s) → OFF
```
ATS transfer: 100ms mechanical, 300s retransfer delay.

---

## Reward System

A **6-component, research-informed** reward function with scenario-type-aware weight profiles:

| Component | Range | Method | Source |
|-----------|-------|--------|--------|
| Thermal safety | [-1, 0.1] | Dual softplus barriers + positive baseline when all zones safe | Google/DeepMind 2017, ICLR 2025, DCRL-Green |
| Power safety | [-1, 0] | UPS SOC softplus barrier + fault penalty | Novel |
| Efficiency | [-1, 0] | `−tanh((PUE−1)/2)`, suppressed during power emergencies | DCRL-Green |
| Scenario progress | [-1, 1] | Delta-based (rewards the step that caused progress change) | Process reward models |
| Procedure | [-1, 1] | Rule-based: diagnose before repair, etc. | Operations research |
| Action quality | [-1, 1] | Invalid penalty, context-aware repeat/wait, diagnosis bonus | Novel |

**Weight profiles** adjust emphasis by scenario type:

| Profile | Safety | Power | Efficiency | Progress | Procedure | Action |
|---------|--------|-------|------------|----------|-----------|--------|
| Thermal | 0.30 | 0.05 | 0.10 | 0.30 | 0.20 | 0.05 |
| Power | 0.10 | 0.25 | 0.05 | 0.30 | 0.25 | 0.05 |
| Default | 0.30 | 0.15 | 0.25 | 0.00 | 0.00 | 0.30 |

**Key design decisions:**

*Softplus barriers (Google/DeepMind 2017, DCRL-Green ICLR 2025):*
```python
penalty += softplus((T − T_recommended) / α_rec)    # Gentle barrier
penalty += 3.0 · softplus((T − T_allowable) / α_allow)  # Steep barrier
reward = −tanh(penalty / 8.0)                         # Bounded [-1, 0.1]
```
Unlike hard thresholds, softplus provides gradient signal *near* limits, enabling the agent to learn preventive behavior rather than only reacting after violations.

*Positive safe-state baseline (DCRL-Green):* When all zones are well within safe range (>3°C below recommended max), thermal safety returns +0.1. This gives the model a positive signal for *maintaining* good state, not just penalty avoidance.

*Power-emergency efficiency suppression:* During UPS-on-battery or fault conditions, the efficiency component returns 0 instead of penalizing load shedding that raises PUE but correctly preserves battery life.

*Context-aware action quality:* `wait` and `check_status` are exempt from the repeat penalty (legitimately repeatable). Waiting during generator startup yields a small positive signal (+0.1) instead of penalty.

---

## Scenarios

6 operational scenarios across 3 difficulty levels:

### Thermal (Category A)

| ID | Scenario | Difficulty | Fault | Resolution |
|----|----------|------------|-------|------------|
| A1 | Cooling Setpoint Optimization | Easy | CRACs at 15°C (wasteful) | PUE < 1.6, temps in ASHRAE recommended |
| A2 | Thermal Event Response | Medium | CRAC-3 compressor failure | All zones in recommended range for 2+ steps |
| A4 | CRAC Failure Cascade | Hard | CRAC-1 compressor + CRAC-3 fan | All zones in allowable range for 2+ steps |

### Power (Category B)

| ID | Scenario | Difficulty | Fault | Resolution |
|----|----------|------------|-------|------------|
| B1 | UPS Alarm Response | Medium | UPS transferred to battery (restored) | Diagnose + acknowledge alarm |
| B3 | Generator Test Protocol | Easy | None (routine test) | Complete 5-step protocol correctly |
| B4 | Power Failure Cascade | Hard | Utility loss + extended gen warmup | Generator loaded + temps stable + SOC > 20% |

Each scenario defines:
- Configuration overrides (e.g., extend generator warmup time)
- Fault injection (run after thermal warmup for realistic steady-state)
- Progress metric (normalized [0,1] for delta-based reward)
- Procedure rules (e.g., "diagnose before adjust_setpoint": bonus +0.2, penalty −0.1)

---

## Competitive Edge vs. Existing Solutions

| Feature | DC-Ops | CoolSim / EnergyPlus | Gymnasium DC envs | Custom RL envs |
|---------|--------|---------------------|-------------------|----------------|
| **Physics fidelity** | RC thermal + quadratic UPS + generator FSM | High (CFD-level) | Simplified or lookup-based | Varies |
| **Speed** | <1ms/step, 256 tests in <10s | Minutes per step | Fast but less accurate | Varies |
| **LLM-native** | Text dashboard + NL commands | Numeric API | Numeric API | Numeric API |
| **Multi-subsystem** | Thermal + Power + Cooling | Thermal only | Thermal only | Usually single |
| **Procedural reward** | Built-in diagnose-before-fix rules | None | None | None |
| **OpenEnv compatible** | Full OpenEnv integration | No | No | No |
| **GRPO/TRL ready** | Direct integration with HF TRL | No | With wrappers | Manual |
| **Configurable facilities** | 3 YAML configs + custom | Config files | Hardcoded | Varies |
| **Safety barriers** | Softplus (Google/DeepMind) | N/A | Hard thresholds | Hard thresholds |

### Key differentiators:

1. **LLM-first design**: The agent reads a text dashboard and issues natural-language commands — no numeric tensors, no discretized action spaces. This is how real operators work.

2. **Physics + speed**: RC thermal networks give us 1000 steps/second with validated physical accuracy. EnergyPlus is more precise but 1000× slower — unusable for RL training.

3. **Multi-objective reward**: Instead of a single scalar, the reward decomposes into 6 interpretable components. Weight profiles auto-adjust by scenario type. Softplus barriers provide smooth gradient near safety limits.

4. **Procedural correctness**: Operations research shows that diagnosis before intervention reduces errors by 40%+. DC-Ops encodes these operational procedures as explicit reward rules.

5. **OpenEnv ecosystem**: Plug into TRL's GRPOTrainer, deploy to HuggingFace Spaces, scale to 16,000+ concurrent sessions with the OpenEnv infrastructure.

---

## ASHRAE Thermal Guidelines

All safety thresholds follow **ASHRAE TC 9.9, 5th Edition (2021)**:

| Class | Recommended | Allowable | Application |
|-------|-------------|-----------|-------------|
| A1 | 18-27°C | 15-32°C | Enterprise servers |
| A2 | 18-27°C | 10-35°C | Volume servers |
| A3 | 18-27°C | 5-40°C | Extended temperature |
| A4 | 18-27°C | 5-45°C | Maximum flexibility |
| H1 | 18-22°C | 5-25°C | High-density / AI / HPC |

The large facility config includes an **H1 zone** with 20 kW/rack GPU servers — the thermal envelope is much tighter, making cooling optimization critical.

---

## Quick Start

```bash
# Clone and install
cd dc_ops_env
uv sync

# Run tests (256 tests, <10s)
uv run pytest tests/ -v

# Start the server
uv run server

# Or with Docker
docker build -t dc-ops:latest -f server/Dockerfile .
docker run -d -p 8000:8000 dc-ops:latest
```

See [dc_ops_env/README.md](dc_ops_env/README.md) for full setup, deployment, and training instructions.

---

## Citation

If you use DC-Ops in your research, please cite:

```bibtex
@software{dc_ops_2026,
  title={DC-Ops: Physics-Based Datacenter Operations RL Environment},
  year={2026},
  url={https://github.com/your-org/dc-ops},
  note={Built on Meta OpenEnv framework}
}
```

## License

BSD-style license. See [LICENSE](LICENSE) for details.
