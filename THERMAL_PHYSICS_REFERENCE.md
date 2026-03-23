# Datacenter Thermal Physics & Operations Reference
## Engineering Data for Physics-Based Simulation

---

## 1. RC Thermal Network Models for Datacenters

### Fundamental Equations (Electrical-Thermal Analogy)

| Electrical | Thermal | Formula | Units |
|-----------|---------|---------|-------|
| Voltage (V) | Temperature (T) | - | K or °C |
| Current (I) | Heat flow (Q) | - | W |
| Resistance (R) | Thermal Resistance (R_th) | See below | K/W |
| Capacitance (C) | Thermal Capacitance (C_th) | C = m * c_p | J/K |
| Time constant | Thermal time constant | τ = R_th * C_th | seconds |

### Thermal Resistance Formulas

**Conduction:**
```
R_cond = L / (k * A)
```
- L = thickness (m)
- k = thermal conductivity (W/(m·K))
- A = cross-sectional area (m²)

**Convection:**
```
R_conv = 1 / (h * A)
```
- h = convective heat transfer coefficient (W/(m²·K))
- A = surface area (m²)

**Forced convection (airflow through equipment):**
```
R_forced = 1 / (ṁ * c_p)
```
- ṁ = mass flow rate (kg/s)
- c_p = specific heat of air (J/(kg·K))

### Thermal Capacitance

```
C_th = m * c_p    [J/K]
```

### Governing ODE for Single-Node RC Model

```
C * dT/dt = Q_in - (T - T_ambient) / R
```

Solution (step response):
```
T(t) = T_ambient + Q_in * R * (1 - e^(-t/τ))
```
where τ = R * C

### Multi-Node Network Equations

For each node i in the thermal network:
```
C_i * dT_i/dt = Q_i + Σ_j [ (T_j - T_i) / R_ij ]
```

In matrix form (state-space):
```
C * dT/dt = -G * T + Q + G_boundary * T_boundary
```
where G is the thermal conductance matrix (W/K).

Thermal couplings:
- **Parallel:** TC_total = TC_1 + TC_2 + ...
- **Series:** 1/TC_total = 1/TC_1 + 1/TC_2 + ...

### Specific Datacenter Thermal Values

**Server thermal mass (measured experimentally):**
- Typical 2U server: **11.1 kJ/K** (average of 12 test runs, ±7%)
- Server power range tested: 220-350W
- Air mass flow rate at max fan: 6.61×10⁻² kg/s
- Component temperatures: 27°C (chassis) to 75°C (CPU)
- Specific heat of server materials: 0.5-0.9 kJ/(kg·K)

**Server energy balance equation:**
```
Q̇_server = ṁ * c_p * ΔT_air + (ρVc_p)_unit * dT_unit/dt
```

**Air properties (standard conditions, ~20°C, sea level):**
- Density (ρ): 1.2 kg/m³
- Specific heat (c_p): 1.005 kJ/(kg·K) = 1005 J/(kg·K)
- ρ * c_p = 1.206 kJ/(m³·K)

**Water properties:**
- Specific heat (c_p): 4.2 kJ/(kg·K)
- Density: 997 kg/m³

### Temperature Rise During Cooling Failure

- Maximum initial rate of temperature rise (before thermal mass absorbs heat): **5°C/minute or more**
- At 200 W/ft²: reaches 60°C (140°F) in ~5 minutes
- Chilled water thermal mass can delay temperature rise by minutes to hours depending on loop volume
- Server thermal inertia provides only a few seconds of delay
- Fan rotational inertia: airflow continues briefly after power loss
- Chiller restart time after power restoration: **10-15 minutes**

---

## 2. CRAC/CRAH Unit Performance

### Sensible Heat Formula (Imperial)

```
Q_sensible = 1.08 × CFM × ΔT    [BTU/hr]
```
where:
- 1.08 = ρ_air × 60 min/hr × c_p = 0.075 lb/ft³ × 60 × 0.24 BTU/(lb·°F)
- CFM = volumetric airflow in cubic feet per minute
- ΔT = temperature difference in °F

More precise constant: **1.085** (using standard CFM at sea level, 1 atm)

### Sensible Heat Formula (SI / Metric)

```
Q = ṁ × c_p × ΔT    [W]
Q = ρ × V̇ × c_p × ΔT    [W]
```
where:
- ṁ = mass flow rate (kg/s)
- V̇ = volumetric flow rate (m³/s)
- ρ = 1.2 kg/m³
- c_p = 1005 J/(kg·K)
- ΔT in K or °C

### Conversion: Watts to BTU/hr
```
1 W = 3.412 BTU/hr
1 kW = 3,412 BTU/hr
```

### Airflow Requirements per kW of IT Load
- Rack-mount servers: **~160 CFM per kW**
- Blade servers: **~120 CFM per kW**
- At ΔT = 20°F (11.1°C): 1 kW requires ~158 CFM

### Typical CRAC Unit Specifications

| Parameter | 20-Ton CRAC | Notes |
|-----------|-------------|-------|
| Nominal cooling capacity | 70 kW (20 tons) | 1 ton ≈ 3.517 kW thermal |
| Sensible capacity at 75°F return | 84 kW | Manufacturer rated |
| Sensible capacity at 90°F return | 137 kW | Higher return = more capacity |
| Airflow (CFM) | 12,000 CFM | Typical for 20-ton unit |
| Net sensible (after fan heat) | ~68.7 kW | Fan heat deducted |
| Floor footprint | ~3 × 1 m | Per unit |

### Capacity vs. Return Air Temperature

CRAC capacity increases with higher return air temperature. A unit rated at 84 kW with 75°F return air can deliver **137 kW with 90°F return air** -- a ~63% increase.

### Conversion Factors
```
1 ton of refrigeration = 12,000 BTU/hr = 3.517 kW (thermal)
1 kW IT load ≈ 0.284 tons of cooling needed
```

### CRAC Sizing Guidelines
```
Required CRAC Capacity = Number of IT cabinets × kW per cabinet
```

With redundancy:
- Tier I: N (no redundancy)
- Tier II-IV: N+1 or N+2

Recommended: Average cabinet load should not exceed **6 kW** for efficient space utilization.

---

## 3. PUE (Power Usage Effectiveness)

### Formula

```
PUE = Total Facility Power / IT Equipment Power
```

Reciprocal metric:
```
DCiE = 1 / PUE = IT Equipment Power / Total Facility Power
```

### Components of Total Facility Power

```
Total Facility Power = IT Power + Cooling Power + Power Distribution Losses + Lighting + Other
```

Breakdown:
- **IT Equipment**: Servers, storage, networking
- **Cooling**: Chillers, CRACs/CRAHs, cooling towers, pumps, fans
- **Power Distribution**: UPS losses, PDU losses, transformer losses, switchgear
- **Lighting & Other**: Lighting, security, fire suppression

### Typical PUE Values

| Datacenter Type | PUE Range | Notes |
|----------------|-----------|-------|
| Best-in-class hyperscale | 1.1 - 1.2 | Google, Facebook, Microsoft |
| Efficient enterprise | 1.2 - 1.4 | Modern, well-managed |
| Average datacenter | 1.5 - 1.6 | Industry average (2024 Uptime survey: 1.5) |
| Legacy/inefficient | 1.8 - 3.0 | Older facilities |
| Ideal (theoretical) | 1.0 | All power goes to IT |

### PUE Rating Scale
- < 1.2: Excellent
- 1.3 - 1.5: Good
- 1.6 - 1.8: Acceptable
- > 1.8: Poor

### Power Distribution Loss Breakdown
- UPS losses: 4-10% (depends on load and topology)
- PDU/Transformer losses: 1-3%
- Total electrical distribution: 10-12% of total facility energy (average)

---

## 4. Hot Aisle / Cold Aisle Containment

### Temperature Deltas

| Configuration | Typical ΔT (Supply to Return) |
|--------------|-------------------------------|
| No containment | Variable, lots of mixing |
| Cold Aisle Containment (CAC) | 10-15°C (18-27°F) |
| Hot Aisle Containment (HAC) | 15-20°C (27-36°F) |

### Typical Temperatures

| Parameter | Without Containment | With Containment |
|-----------|-------------------|-----------------|
| Cold aisle supply | 18-22°C (64-72°F) | 18-27°C (64-80°F) |
| Hot aisle return | 30-40°C (86-104°F) | 35-45°C (95-113°F) |
| CRAC supply air | 12-15°C (54-59°F) | 12-18°C (54-64°F) |

### Recirculation and Bypass

**Return Temperature Index (RTI):**
```
RTI = (T_return_rack - T_supply_CRAC) / (T_return_CRAC - T_supply_CRAC) × 100
```
or equivalently:
```
RTI = (Rack Airflow / CRAC Airflow) × 100
```

- RTI = 100%: Balanced (no recirculation, no bypass)
- RTI > 100%: Net recirculation (hot air re-entering cold aisle)
- RTI < 100%: Net bypass (cold air bypassing racks to return)

### Energy Benefits of Containment

- Containment allows raising supply temperature setpoint by **~10°F (5.5°C)**
- Hot aisle containment saves **~43% in annual cooling energy cost**
- Corresponding PUE reduction: **~15%**
- Cooling energy reduction with containment: **40-50%**

---

## 5. UPS Specifications

### Efficiency Formula

```
η_UPS = P_output / P_input = P_output / (P_output + P_losses)
```

### UPS Loss Model (Quadratic)

Total losses follow a quadratic function of load:
```
P_losses = a * P_rated + b * P_load + c * P_load² / P_rated
```

Or as fraction of rated power (y = losses/P_rated, x = load fraction):
```
y = c_0 + c_1 * x + c_2 * x²
```

Where:
- **c_0** = no-load losses (fixed, independent of load): transformers, capacitors, logic boards, comms cards
- **c_1 * x** = proportional losses (linear with load): conduction losses, heat dissipation
- **c_2 * x²** = square-law losses: I²R losses in conductors, bus bars, circuit breakers

**Example coefficients (delta conversion UPS):**
```
y = 0.01333 + 0.00640 * x + 0.01081 * x²
```
(where x = load fraction 0-1, y = loss fraction of rated capacity)

Then efficiency:
```
η = x / (x + y) = x / (x + c_0 + c_1*x + c_2*x²)
```

### No-load losses represent over **40%** of all UPS losses and are the largest opportunity for efficiency improvement.

### Typical Double-Conversion UPS Efficiency by Load

| Load % | 500 kVA UPS | 225-300 kVA UPS | General Range |
|--------|-------------|-----------------|---------------|
| 25% | 90.5% | 85.5% | 85-91% |
| 50% | 93.6% | 91.7% | 90-94% |
| 75% | 94.0% | 92.8% | 92-95% |
| 100% | 93.9% | 92.9% | 92-94% |

### UPS Operating Modes

| Mode | Efficiency | Trade-off |
|------|-----------|-----------|
| Double conversion (VFI) | 90-95% | Full protection, lowest efficiency |
| Line interactive (VI) | 95-98% | Partial protection |
| Eco mode / bypass | 98-99% | Minimal protection |

### Modern transformer-free UPS: maintains >95% efficiency from 25% to 100% load (flatter efficiency curve).

### Battery Runtime Calculation

```
Runtime (hours) = (V_battery × Ah × η_discharge × d_aging) / P_load
```

Where:
- V_battery = nominal battery voltage (V)
- Ah = battery capacity (Ah)
- η_discharge = discharge efficiency (0.8-0.95)
- d_aging = aging derating factor (typically 0.8-0.9 for end-of-life)
- P_load = load power in watts

**Minimum DC-side energy:**
```
E_required = P_load × t_runtime / η_UPS
```

**Example:** 20 kW load, 15 min runtime, 94% UPS efficiency:
```
E = 20,000 × (15/60) / 0.94 = 5,319 Wh ≈ 5.3 kWh (DC side)
```
(With N+1 redundancy and derating: ~7.4 kWh)

### Battery Sizing Factors

- Growth buffer: at least 25%
- Aging allowance: ~10%
- Temperature: every 10°C increase above 25°C **halves** VRLA battery lifespan
- Peukert effect: higher discharge rates reduce effective capacity

### Typical Runtime Requirements

| Facility Type | Battery Runtime |
|--------------|----------------|
| Hyperscale (Internet giants) | 1-2 minutes |
| Cloud/colocation | 5 minutes |
| Financial/enterprise | 10-15 minutes |
| Battery recharge time | ~10× discharge time |

---

## 6. PDU Power Distribution

### Three-Phase Power Formulas

**Three-phase power:**
```
P_3φ = √3 × V_L-L × I_L × PF
```

**Single-phase power (line-to-neutral):**
```
P_1φ = V_L-N × I × PF
```

**Voltage relationships (Wye/Star configuration):**
```
V_L-L = √3 × V_L-N
```

Common voltage systems:
- North America: 208V L-L / 120V L-N (also 480V distribution)
- Europe/International: 400V L-L / 230V L-N

### Typical PDU Specifications

| Parameter | US 3-Phase PDU | European 3-Phase PDU |
|-----------|---------------|---------------------|
| Input voltage | 208V (L-L) | 400V (L-L) |
| Phase-neutral voltage | 120V | 230V |
| Max current per phase | 24A | 32A |
| Total capacity | 8.6 kW | 22 kW |
| Typical outlets | 48 | 24-48 |
| Circuit breaker per branch | 20A | 16A or 20A |

**US Example:**
```
P = √3 × 208V × 24A = 8,646W ≈ 8.6 kW
```

### 80% Derating Rule (NEC)

Circuit breakers must be derated to 80% of rating for continuous loads:
```
P_continuous_max = 0.80 × P_breaker_rated
```

Example: 20A breaker at 208V:
```
P_max = 0.80 × 20A × 208V = 3,328W per branch
```

### Phase Balancing

- Each of 3 branches must stay under maximum phase current rating
- Ideal: equal power across all three phases
- Unbalanced loads cause higher neutral current and increased losses
- At 208V/24A: only 1.1A headroom per phase with four 1,800W servers

### PDU Losses

- High-efficiency transformer PDU: **97-99% efficient** (1-3% losses)
- Standard transformer PDU: **95-97% efficient** (3-5% losses)
- Difference between high-efficiency and standard: 2-3% improvement
- Total electrical distribution losses: 10-12% of total datacenter energy

---

## 7. Typical Datacenter Operating Parameters

### ASHRAE TC 9.9 Thermal Guidelines (2021, 5th Edition)

**Recommended Envelope (all classes):**
- Temperature: **18-27°C** (64.4-80.6°F)
- Humidity: Dew point -9°C to 15°C, max 60% RH

**Allowable Envelopes by Class:**

| Class | Temp Min | Temp Max | Max DP | Max RH | Application |
|-------|----------|----------|--------|--------|-------------|
| A1 | 15°C (59°F) | 32°C (89.6°F) | 17°C | 80% | Enterprise servers |
| A2 | 10°C (50°F) | 35°C (95°F) | 21°C | 80% | Volume servers |
| A3 | 5°C (41°F) | 40°C (104°F) | 24°C | 85% | Extended temp |
| A4 | 5°C (41°F) | 45°C (113°F) | 24°C | 90% | Max flexibility |
| H1 | 5°C (41°F) | 25°C (77°F) | - | - | High-density/AI/HPC |

**H1 Class (High-Density):**
- Recommended: **18-22°C** (64.4-71.6°F) -- narrower band
- Allowable upper limit: **25°C** (77°F) -- much tighter than standard

**Minimum humidity (all classes):** Higher of -12°C dew point OR 8% RH (intersect at ~25°C)

**Rate of Change Limits:**
- Solid-state equipment: 20°C/hour max; 5°C per 15-minute max
- Tape storage: 5°C/hour max

**Sensor Accuracy Requirements:**
- Standard: ±0.5°C
- High-density: ±0.3°C
- Humidity: ±3% RH

### IT Load Densities Per Rack

| Category | Power per Rack | Notes |
|----------|---------------|-------|
| Low density (legacy) | 2-5 kW | Traditional enterprise |
| Standard density | 5-10 kW | Current average ~6-8 kW |
| High density | 10-30 kW | Modern enterprise, HPC |
| Ultra-high density | 30-85 kW | GPU clusters, AI training |
| Extreme density | 85-100+ kW | Latest AI/HPC racks |

Industry averages:
- 2016: 6.1 kW/rack
- 2023 (Uptime Institute): ~6 kW/rack average
- 2024 (AFCOM): ~12 kW/rack average
- 37% of datacenters still operate below 10 kW/rack

Individual component power:
- Modern server processor: >200W
- Complete server: ~500W average
- GPU: >300W

### Cooling Power as Fraction of IT Load

**COP (Coefficient of Performance) of cooling system:**
```
COP = Q_cooling / P_cooling_input
```

**Chiller efficiency:**
```
kW/ton = P_input / Q_cooling_tons
kW/ton = 3.517 / COP
COP = 3.517 / (kW/ton)
```

| Cooling Component | Typical COP / Efficiency |
|------------------|------------------------|
| Water-cooled chiller | COP ~5-7 (0.5-0.7 kW/ton) |
| Air-cooled chiller | COP ~2.5-3.5 (1.0-1.4 kW/ton) |
| Typical central plant total | 0.6-0.73 kW/ton |
| CRAC fan power | ~5-15% of CRAC capacity |

**IPLV/NPLV (Part-Load Chiller Efficiency):**
```
For kW/ton: IPLV = 1 / [(0.01/A) + (0.42/B) + (0.45/C) + (0.12/D)]
For COP:    IPLV = 0.01*A + 0.42*B + 0.45*C + 0.12*D
```
Where A, B, C, D = efficiency at 100%, 75%, 50%, 25% load respectively.

Standard conditions:
- Chilled water supply: 6.7°C (44°F)
- Condenser water supply: 29.4°C (85°F)

**Cooling power fraction of IT load:**
```
P_cooling / P_IT ≈ (PUE - 1) × (cooling fraction of overhead)
```
Typically cooling is 30-50% of non-IT overhead, meaning:
- PUE 1.5 → overhead = 50% of IT → cooling ≈ 25-35% of IT load
- PUE 2.0 → overhead = 100% of IT → cooling ≈ 50-70% of IT load
- Best case (PUE 1.1) → cooling ≈ 5-8% of IT load

### Power Transfer Timing

| Event | Typical Duration |
|-------|-----------------|
| Utility power failure detection | 0.5-1 second |
| Generator start delay (programmed) | 3-5 seconds |
| Engine cranking to running | 2-10 seconds |
| Generator warm-up / load acceptance | 5-10 seconds |
| ATS transfer time (mechanical) | 50-200 milliseconds |
| **Total: utility loss to generator power** | **10-20 seconds** |
| UPS battery bridge time | Covers the 10-20s gap |
| Chiller restart after power restoration | **10-15 minutes** |

**NFPA 110 requirement:** Type 10 (life safety): generator must accept load within 10 seconds.

Utility re-closer patterns:
- Short bump: <1 second
- Final re-closer attempt: up to ~3 seconds
- If power is down >3 seconds, likely down for minutes to hours

**UPS backup durations:**
- Conventional battery: 5-15 minutes
- Flywheel UPS: 7-20 seconds only
- Battery recharge time: ~10× discharge time

### Heat Load Conversion

Nearly all IT power consumption converts to heat:
```
Q_heat ≈ P_electrical (for IT equipment)
1 W electrical = 1 W thermal = 3.412 BTU/hr
```

Additional heat sources in datacenter:
- Lighting: typically 1-2 W/ft² (10-20 W/m²)
- People: ~100-120 W sensible heat per person
- UPS losses: 4-10% of IT load
- PDU/transformer losses: 1-3% of IT load

---

## 8. Quick-Reference Formulas for Simulation

### Airflow-Temperature-Power Triangle
```
Q [kW] = ṁ [kg/s] × c_p [kJ/(kg·K)] × ΔT [K]
Q [kW] = ρ [kg/m³] × V̇ [m³/s] × c_p [kJ/(kg·K)] × ΔT [K]

# With ρ=1.2, c_p=1.005:
Q [kW] = 1.206 × V̇ [m³/s] × ΔT [K]

# Imperial:
Q [BTU/hr] = 1.085 × CFM × ΔT [°F]

# CFM to m³/s:
1 CFM = 4.719 × 10⁻⁴ m³/s
1 m³/s = 2,118.88 CFM
```

### Datacenter RC Model (Simplified Zone)
```
C_zone × dT_zone/dt = Q_IT + Q_envelope - Q_cooling

Where:
  C_zone = C_air + C_equipment + C_structure  [J/K]
  Q_IT = IT power dissipation  [W]
  Q_envelope = (T_outdoor - T_zone) / R_envelope  [W]
  Q_cooling = ṁ_cooling × c_p × (T_zone - T_supply)  [W]
```

### PUE Dynamic Calculation
```
PUE(t) = [P_IT(t) + P_cooling(t) + P_UPS_loss(t) + P_PDU_loss(t) + P_lighting(t)] / P_IT(t)
```

### UPS Efficiency as Function of Load
```
η(x) = x / (x + c_0 + c_1*x + c_2*x²)

Where x = P_load / P_rated (0 to 1)

# Example coefficients (modern double-conversion):
# c_0 = 0.013 (no-load: ~1.3% of rated)
# c_1 = 0.006 (proportional: ~0.6% of rated at full load)
# c_2 = 0.011 (square-law: ~1.1% of rated at full load)
```

### CRAC Cooling Output vs Return Temperature
```
Q_CRAC = ṁ_air × c_p × (T_return - T_supply_coil)

# Capacity increases with higher return air temperature
# Linear approximation over operating range:
Q_actual ≈ Q_rated × [1 + α × (T_return - T_return_rated)]
# where α ≈ 0.02-0.04 per °C
```

---

## Sources

### RC Thermal Models
- [Lumped Parameter Model and Thermal Network Method](https://community.ptc.com/sejnu66972/attachments/sejnu66972/PTCMathcad/176512/1/2.1_Lumped_Parameter_Model_and_the_Thermal_Network_Method.pdf)
- [Hybrid Lumped Capacitance-CFD Model for Data Center Transients](https://www.researchgate.net/publication/264711473_A_hybrid_lumped_capacitance-CFD_model_for_the_simulation_of_data_center_transients)
- [Characterization of Server Thermal Mass - Electronics Cooling](https://www.electronics-cooling.com/2013/12/characterization-server-thermal-mass/)
- [Thermal Systems - Swarthmore](https://lpsa.swarthmore.edu/Systems/Thermal/SysThermalElem.html)
- [Lumped Capacitance Modeling - DSPE](https://www.dspe.nl/knowledge/thermomechanics/chapter-4-thermo-mechanical-modeling/4-2-lumped-capacitance-modeling/)

### CRAC/CRAH Units
- [Data Center Cooling: CRAC/CRAH Redundancy and Capacity - Uptime Institute](https://journal.uptimeinstitute.com/data-center-cooling-redundancy-capacity-selection-metrics/)
- [HVAC Cooling Systems for Data Centers - CED Engineering](https://www.cedengineering.com/userfiles/M05-020%20-%20HVAC%20Cooling%20Systems%20for%20Data%20Centers%20-%20US.pdf)
- [Sensible Heat Calculation - MEP Academy](https://mepacademy.com/how-to-calculate-sensible-heat-transfer-for-air/)
- [HVAC Energy Calc Cheat Sheet](https://www.wbdg.org/files/pdfs/rcx_energy_calc_cheat_sheet.pdf)

### PUE
- [Power Usage Effectiveness - Wikipedia](https://en.wikipedia.org/wiki/Power_usage_effectiveness)
- [PUE Comprehensive Examination - LBNL](https://datacenters.lbl.gov/sites/default/files/WP49-PUE%20A%20Comprehensive%20Examination%20of%20the%20Metric_v6.pdf)
- [Key Data Center Cooling Metrics - CoolSim](https://coolsimsoftware.com/articles/key-data-center-cooling-metrics/)

### Hot/Cold Aisle Containment
- [Hot vs Cold Aisle Containment - Northern Link](https://northernlink.com/hot-vs-cold-aisle-containment-managing-delta-t-for-optimal-cooling-efficiency/)
- [Impact of Hot and Cold Aisle Containment - Schneider Electric](https://www.se.com/us/en/download/document/SPD_DBOY-7EDLE8_EN/)

### UPS
- [Making Large UPS Systems More Efficient - APC White Paper 108](https://www.newark.com/pdfs/techarticles/APC/Making_Large_UPS_Systems_More_Efficient.pdf)
- [UPS Battery Capacity Estimation Guide - Attom Tech](https://attom.tech/data-center-ups-battery-capacity-estimation-from-formula-to-practice-a-complete-guide/)
- [UPS Efficiency - Riello UPS](https://www.riello-ups.com/questions/53-what-is-ups-efficiency-and-how-is-it-calculated)
- [GE SG Series UPS Datasheet](https://www.power-solutions.com/wp-content/uploads/2014/06/GE-SG-Series-500kVA-data-sheet.pdf)

### PDU
- [Three-Phase Power and PDUs - Cloudflare](https://blog.cloudflare.com/an-introduction-to-three-phase-power-and-pdus/)
- [PDU Losses - ENERGY STAR](https://www.energystar.gov/products/data_center_equipment/16-more-ways-cut-energy-waste-data-center/reduce-energy-losses-power-distribution-units-pdus)

### ASHRAE Guidelines
- [ASHRAE TC 9.9 Thermal Guide 2026 - Envigilance](https://envigilance.com/compliance/ashrae-tc-9-9/)
- [2021 Equipment Thermal Guidelines Reference Card](https://www.ashrae.org/file%20library/technical%20resources/bookstore/supplemental%20files/therm-gdlns-5th-r-e-refcard.pdf)
- [ASHRAE Thermal Guidelines Evolution - Attom Tech](https://attom.tech/ashraes-new-thermal-guideline-update-a-new-high-density-trend/)

### Operating Parameters
- [Generator Start Time Delay - Facilities.net](https://www.facilitiesnet.com/powercommunication/article/What-Is-the-Best-Start-time-Delay-For-A-Standby-Generator-In-A-Data-Center--14154)
- [Rack Power Density - Enconnex](https://blog.enconnex.com/data-center-rack-density)
- [Chiller Efficiency Calculation - AirCondLounge](https://aircondlounge.com/chiller-efficiency-calculation-kw-ton-cop-eer-iplv-nplv/)
- [Datacenter Thermal Runaway - Active Power WP 105](https://powertechniquesinc.com/wp-content/uploads/2015/08/Active-Power-WP-105-Data-Center-Thermal-Runaway.pdf)
- [Thermal Mass Availability for Cooling Data Centers](https://ansight.com/publications/thermal-mass-availability-for-cooling-data-centers-during-power-shutdown/)
