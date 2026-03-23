# DC-Ops Reward System Design

## Research Foundation

This reward system draws from three key bodies of work:

1. **Google/DeepMind (2017)** — _Transforming Cooling Optimization for Green Data Center via DRL_
   - Reward: `minimize ε_pue + λ·softplus(T − ϕ)`
   - Key insight: **softplus barriers** for temperature constraints (smooth gradient, no cliff)
   - λ=0.01, ϕ=29°C. Penalty coefficient requires per-problem tuning.

2. **ICLR 2025** — _Data Center Cooling System Optimization Using Offline RL_
   - Reward: `r = r₀ − β₁Σf³ − β₂Σsoftplus(T_cold − ρ_T) − β₃Σo − β₄Σsoftplus(T_lat − ρ_L)`
   - Key insight: **normalize coefficients as reciprocal of mean** of each term from baseline data.
   - Ablation shows system is robust to coefficient range.

3. **Process Reward Models for LLM Agents (2025)**
   - Key insight: **delta-based progress rewards** give better credit assignment than cumulative flags.
   - Per-step process rewards outperform sparse outcome-only rewards for multi-step agent tasks.

---

## Problems with Current System

### 1. Scale Mismatch

| Component | Range | Weight | Effective Range |
|-----------|-------|--------|-----------------|
| R_safety | [-10, 0] | 0.4 | [-4.0, 0] |
| R_energy | [-1.5, 0] | 0.2 | [-0.3, 0] |
| R_action | [-0.1, +0.1] | 0.4 | [-0.04, +0.04] |

Safety dominates by 100x over action quality. The signal is almost entirely "how bad are temperatures" — everything else is noise.

### 2. Double-Counting

Safety violations penalized in BOTH `_compute_reward` (R_safety) AND scenario rewards:
- A2: `scenario_reward = -max_over × 0.5`
- A4: `scenario_reward = -max_over × 2.0`
- B4: `scenario_reward -= max_over × 1.5`

Same physical temperature overshoot counted twice with different scales.

### 3. Always-On PUE Penalty

PUE penalty fires every step even in B1 (UPS alarm) or B3 (generator test) where PUE is irrelevant. Adds noise.

### 4. Stateful Scenario Rewards

B1 gives `+0.3` for `diagnosed_ups` on **every step** after diagnosis. B3 accumulates `+0.1+0.2+0.2+0.3` and fires every step once flags are set. Longer episodes inflate rewards.

### 5. No Improvement Signal

Reward doesn't capture _change_ — an agent can't distinguish "things are getting better" from "things are still bad but stable."

---

## Design Principles

1. **Bounded components**: Every component maps to [-1, 1] via tanh normalization
2. **Softplus barriers**: Smooth gradients near thresholds, no reward cliffs (Google/DeepMind)
3. **Delta-based progress**: Reward the _change_ in scenario progress, not cumulative state
4. **No double-counting**: Safety comes from the reward function, scenarios report progress only
5. **Scenario-aware weights**: Different weight profiles for thermal vs power scenarios
6. **GRPO-compatible**: Bounded per-step rewards work well with advantage estimation

---

## Mathematical Foundation

### Softplus Barrier Function

```
softplus(x) = ln(1 + eˣ)
```

Properties:
- x ≪ 0: softplus(x) → 0 (exponentially small)
- x = 0: softplus(0) = ln(2) ≈ 0.693
- x ≫ 0: softplus(x) → x (linear)
- Derivative: sigmoid(x) — always smooth, never discontinuous

### Tanh Normalization

```
tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)
```

Properties:
- Range: (-1, 1), monotonically increasing
- tanh(0) = 0, tanh(1) ≈ 0.76, tanh(2) ≈ 0.96

Combined: `-tanh(β · softplus((T − T_limit) / α))` gives:
- 0 when T well below limit
- Smooth transition as T approaches limit
- Saturates at -1 for severe violations
- Always in [-1, 0]

---

## Component Specifications

### Total Reward

```
R_total = Σᵢ wᵢ · Rᵢ,  clamped to [-1, 1]
```

Where weights wᵢ sum to 1.0 and each Rᵢ ∈ [-1, 1].

Speed bonus (one-time at resolution) is added on top:
```
R_resolution_step = R_total + speed_bonus
speed_bonus = (step_budget − steps_used) / step_budget  ∈ [0, 1]
```

---

### Component 1: Thermal Safety — R_thermal ∈ [-1, 0]

Penalizes ASHRAE temperature violations using two softplus barriers per zone.

```
For each zone z with ASHRAE class limits (T_rec, T_allow):
    T = max_inlet_temp(z)

    barrier_rec(z) = softplus((T − T_rec) / α_rec)      # α_rec = 2.0°C
    barrier_allow(z) = softplus((T − T_allow) / α_allow) # α_allow = 1.5°C

    penalty(z) = barrier_rec(z) + 3.0 · barrier_allow(z)

penalty_avg = (1/N) · Σ_z penalty(z)    # Average across N zones

R_thermal = −tanh(penalty_avg / 8.0)
```

**Verification table** (ASHRAE A2: T_rec=27°C, T_allow=35°C, single zone):

| T_inlet (°C) | Over Rec | Over Allow | R_thermal | Interpretation |
|--------------|----------|------------|-----------|----------------|
| 22 | — | — | −0.01 | Normal operation |
| 27 | 0 | — | −0.09 | At recommended limit |
| 30 | +3 | — | −0.22 | Moderate concern |
| 35 | +8 | 0 | −0.64 | At allowable limit |
| 38 | +11 | +3 | −0.90 | Severe violation |
| 40 | +13 | +5 | −0.97 | Near-catastrophic |

**Design rationale**:
- `α_rec = 2.0`: Gentle transition at recommended boundary (±4°C to go from 10% to 90% of penalty)
- `α_allow = 1.5`: Sharper transition at allowable boundary (more urgent)
- `weight = 3.0`: Allowable violations are 3x worse per degree than recommended violations
- `/8.0`: Normalization constant calibrated so T=40°C → R ≈ −0.97

---

### Component 2: Power Safety — R_power ∈ [-1, 0]

Penalizes UPS battery depletion and fault conditions.

```
penalty = 0

For each UPS unit u:
    if mode(u) = ON_BATTERY:
        penalty += softplus((0.5 − SOC(u)) / 0.15)
    elif mode(u) = FAULT:
        penalty += 5.0

R_power = −tanh(penalty / 4.0)
```

**Verification table** (single UPS):

| Mode | SOC | Penalty | R_power | Interpretation |
|------|-----|---------|---------|----------------|
| DOUBLE_CONV | 1.0 | 0 | 0.00 | Normal |
| ON_BATTERY | 0.85 | 0.09 | −0.02 | Brief outage, minor |
| ON_BATTERY | 0.50 | 0.69 | −0.17 | Moderate concern |
| ON_BATTERY | 0.20 | 2.13 | −0.49 | Low battery |
| ON_BATTERY | 0.05 | 3.05 | −0.65 | Critical |
| FAULT | — | 5.0 | −0.85 | UPS fault |

**Design rationale**:
- SOC threshold at 0.5: Concern increases as battery drops below half
- `α = 0.15`: Sharp transition around 50% SOC
- No per-unit normalization: Multiple failing UPS units compound (physically correct)
- `/ 4.0`: Calibrated so single UPS fault ≈ −0.85

---

### Component 3: Efficiency — R_efficiency ∈ [-1, 0]

PUE-based energy efficiency penalty.

```
R_efficiency = −tanh((PUE − 1.0) / 2.0)
```

| PUE | R_efficiency | Rating |
|-----|-------------|--------|
| 1.0 | 0.00 | Ideal |
| 1.2 | −0.10 | Excellent |
| 1.5 | −0.24 | Average |
| 2.0 | −0.46 | Poor |
| 3.0 | −0.76 | Very poor |

**Design rationale**:
- Simple, direct metric used across industry
- `/ 2.0`: Moderate sensitivity — PUE in normal range (1.3–1.8) gives penalty −0.15 to −0.38
- tanh saturation prevents PUE from dominating during extreme conditions

---

### Component 4: Scenario Progress — R_progress ∈ [-1, 1]

Delta-based progress toward scenario resolution. Each scenario reports a `progress` value in [0, 1].

```
R_progress = progress(t) − progress(t−1)
```

Positive for forward progress, negative for regression (e.g., temperatures went back up).

**Progress definitions per scenario**:

| Scenario | Progress Formula | Resolution |
|----------|-----------------|------------|
| A1 | `0.7·clamp((2.0−PUE)/0.4, 0, 1) + 0.3·[temps OK]` | PUE<1.6 ∧ within rec |
| A2 | `0.5 + 0.5·stable/2` if within rec, else `0.4/(1+overshoot)` | 2 stable steps within rec |
| A4 | `0.5 + 0.5·stable/2` if within allow, else `0.4/(1+overshoot)` | 2 stable steps within allow |
| B1 | `0.5·diagnosed + 0.5·acknowledged` | Both done |
| B3 | `0.25·(started + verified + stopped + completed)` | All 4 steps |
| B4 | `0.5·conditions/3` or `0.5 + 0.5·stable/2` if all 3 met | 2 stable steps, all conds |

**Credit assignment example (B1)**:
- Step 1 (diagnose): progress 0→0.5, delta=+0.5 — clear reward for the action that helped
- Step 2 (acknowledge): progress 0.5→1.0, delta=+0.5 — clear reward for resolution
- Step 3 onward: progress stays 1.0, delta=0 — no inflation

---

### Component 5: Procedure — R_procedure ∈ [-1, 1]

Procedural correctness from scenario rules (diagnose before fix, etc.).

```
R_procedure = clamp(scenario.check_procedure(action, history), −1, 1)
```

Unchanged from current implementation. ProcedureRule already provides bonus/penalty values.

---

### Component 6: Action Quality — R_action ∈ [-1, 1]

Assesses whether the action was useful given the current state.

```
if command is invalid:
    R_action = −0.5
elif command is exact repeat of a prior command:
    R_action = −0.2
elif command is "wait" during active concern:
    R_action = −0.2
elif command is "wait" (no concern):
    R_action = 0.0
elif command is diagnostic (diagnose, check_status):
    R_action = 0.3
elif command is intervention (adjust_setpoint, set_fan_speed, etc.):
    R_action = 0.2
elif command is administrative (acknowledge_alarm):
    R_action = 0.1
else:
    R_action = 0.1
```

"Active concern" = any zone above recommended temp OR any UPS on battery.

---

## Weight Profiles

Weights are scenario-type-dependent and sum to 1.0.

| Component | Thermal | Power | Default (no scenario) |
|-----------|---------|-------|----------------------|
| thermal_safety | **0.30** | 0.10 | 0.30 |
| power_safety | 0.05 | **0.25** | 0.15 |
| efficiency | 0.10 | 0.05 | 0.25 |
| scenario_progress | **0.30** | **0.30** | 0.00 |
| procedure | 0.20 | 0.25 | 0.00 |
| action_quality | 0.05 | 0.05 | 0.30 |

**Rationale**:
- **Thermal scenarios**: Safety and progress dominate. Procedure is significant (diagnose before fix). Efficiency matters (especially A1).
- **Power scenarios**: Power safety and progress dominate. Procedure is critical (follow protocols). Thermal is secondary.
- **Default (no scenario)**: No progress/procedure signals. Weight redistributed to safety, efficiency, and action quality.

---

## Integration Design

### Scenario Changes

Add `progress` field to `ScenarioResult`:

```python
@dataclass
class ScenarioResult:
    resolved: bool = False
    resolution_message: str = ""
    scenario_reward: float = 0.0    # Kept for backward compat, not used by new reward
    procedure_reward: float = 0.0
    progress: float = 0.0           # NEW: [0, 1] normalized progress
    info: dict[str, Any] = field(default_factory=dict)
```

### Environment Changes

```python
class DcOpsEnvironment:
    def reset(self, ...):
        # ... existing logic ...
        scenario_type = self._scenario.scenario_type if self._scenario else "default"
        self._reward_fn = RewardFunction(scenario_type=scenario_type)

    def step(self, action, ...):
        # 1. Parse command
        # 2. Advance simulation
        # 3. Evaluate scenario → ScenarioResult (with progress)
        # 4. Compute reward via RewardFunction
        # 5. Check termination
        # 6. Speed bonus on resolution (additive)
```

### Reward Flow

```
step() called
    │
    ├─ parse_command() → CommandResult
    ├─ _advance_simulation() → alarms
    ├─ scenario.evaluate_step() → ScenarioResult (progress, procedure_reward)
    │
    └─ reward_fn.compute(thermal_sim, power_sim, cmd_result, scenario_result)
        ├─ R_thermal  = -tanh(softplus_barriers / 8)     [-1, 0]
        ├─ R_power    = -tanh(soc_barriers / 4)           [-1, 0]
        ├─ R_efficiency = -tanh((PUE-1)/2)                [-1, 0]
        ├─ R_progress = progress_delta                    [-1, 1]
        ├─ R_procedure = clamp(procedure_reward)          [-1, 1]
        └─ R_action   = action_quality_score              [-1, 1]

        → total = clamp(Σ wᵢRᵢ, -1, 1)

    If resolved:
        speed_bonus = (budget - steps) / budget
        total += speed_bonus
```

---

## Testing Strategy

### Unit Tests (per component)

1. **softplus**: Numerical stability (x=100, x=-100, x=0)
2. **R_thermal**: Verify table values above (±5% tolerance)
3. **R_power**: Verify table values above
4. **R_efficiency**: PUE 1.0→0, PUE 2.0→-0.46
5. **R_progress**: Delta computation, no-scenario=0, regression handled
6. **R_procedure**: Bounded [-1, 1]
7. **R_action**: Invalid=-0.5, repeated=-0.2, diagnose=+0.3

### Integration Tests

8. **Full compute**: Proper weighting, total in [-1, 1]
9. **Weight profiles**: Correct profile selected per scenario type
10. **Speed bonus**: Applied at resolution, correct value
11. **Escalation**: Penalty applied correctly
12. **Environment integration**: Reward flows through DcOpsEnvironment correctly
13. **Backward compatibility**: Tests from test_scenarios.py and test_environment.py still pass
