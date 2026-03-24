---
title: DC-Ops Environment Server
emoji: 🖥️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - datacenter
  - simulation
---

# DC-Ops Environment

A physics-based datacenter operations environment for training LLM agents, built on Meta's [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

The agent reads a text-based NOC dashboard and issues natural-language operator commands — exactly as a human datacenter operator would.

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker (for containerized deployment)

### Install & Run Locally

```bash
# Clone the repository
git clone <repo-url>
cd dc_ops_env

# Install dependencies
uv sync

# Run the test suite (254 tests, <10s)
uv run pytest tests/ -v

# Start the server
uv run server
```

The server starts at `http://localhost:8000` with:
- **Web UI** → `http://localhost:8000/web`
- **API docs** → `http://localhost:8000/docs`
- **Health check** → `http://localhost:8000/health`
- **WebSocket** → `ws://localhost:8000/ws`

### Run with Docker

```bash
# Build the image
docker build -t dc-ops:latest -f server/Dockerfile .

# Run the container
docker run -d -p 8000:8000 dc-ops:latest

# Verify it's running
curl http://localhost:8000/health
```

---

## OpenEnv Integration

DC-Ops is a fully compliant [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment. OpenEnv provides:
- **MCP tool-based interactions** for LLM agents (WebSocket `/ws`)
- **HTTP orchestration layer** for training pipelines (`/reset`, `/step`, `/state`)
- **HuggingFace Spaces deployment** via `openenv push`
- **TRL/GRPO integration** for RL training with `GRPOTrainer`

### Action & Observation Models

**DcOpsAction** — the agent's command:
```python
class DcOpsAction(Action):
    command: str    # e.g., "diagnose CRAC-3", "adjust_setpoint CRAC-1 20"
    reasoning: str  # Optional chain-of-thought
```

**DcOpsObservation** — what the agent sees:
```python
class DcOpsObservation(Observation):
    dashboard: str           # Text-rendered monitoring dashboard
    available_actions: list  # Valid commands the agent can issue
    alert: str               # Current active alert message
    scenario_type: str       # "thermal", "power", etc.
    steps_remaining: int     # Steps left in episode budget
    action_result: str       # Feedback from last action
```

### Available Commands

| Command | Format | Description |
|---------|--------|-------------|
| `diagnose` | `diagnose <unit_id>` | Inspect a CRAC/UPS/PDU for faults |
| `adjust_setpoint` | `adjust_setpoint <crac_id> <temp_c>` | Change CRAC supply air setpoint |
| `set_fan_speed` | `set_fan_speed <crac_id> <pct>` | Set CRAC fan speed (0-100%) |
| `set_rack_load` | `set_rack_load <rack_id> <kw>` | Adjust rack IT load (migrate workload) |
| `start_crac` | `start_crac <crac_id>` | Start a standby CRAC unit |
| `stop_crac` | `stop_crac <crac_id>` | Put a CRAC into standby |
| `start_generator` | `start_generator` | Manually start the diesel generator |
| `stop_generator` | `stop_generator` | Initiate generator cooldown |
| `set_ups_mode` | `set_ups_mode <ups_id> <mode>` | Set UPS mode (eco/double_conversion/bypass) |
| `refuel_generator` | `refuel_generator [liters]` | Refuel (default: full tank) |
| `acknowledge_alarm` | `acknowledge_alarm` | Acknowledge current alert |
| `check_status` | `check_status` | Request full status report |
| `escalate` | `escalate` | Escalate to senior engineer |
| `wait` | `wait` | Take no action this step |

---

## Using the Client

### Programmatic Usage (Python)

```python
from dc_ops_env import DcOpsAction, DcOpsEnv

# Connect to a running server
async with DcOpsEnv(base_url="http://localhost:8000") as env:
    # Reset with a specific scenario
    result = await env.reset(scenario="A2")
    print(result.observation.dashboard)

    # Agent loop
    while not result.done:
        result = await env.step(
            DcOpsAction(
                command="diagnose CRAC-3",
                reasoning="CRAC-3 shows compressor failure, need to investigate"
            )
        )
        print(f"Reward: {result.reward}")
        print(result.observation.dashboard)
```

### From Docker Image

```python
from dc_ops_env import DcOpsAction, DcOpsEnv

# Start environment from Docker (auto-manages container lifecycle)
env = DcOpsEnv.from_docker_image("dc-ops:latest")

try:
    result = env.reset(scenario="A2")
    for _ in range(15):
        result = env.step(DcOpsAction(command="check_status"))
        if result.done:
            break
finally:
    env.close()
```

### Concurrent Sessions

The server supports multiple concurrent WebSocket sessions for parallel training:

```python
# In server/app.py — adjust max_concurrent_envs
app = create_app(
    DcOpsEnvironment,
    DcOpsAction,
    DcOpsObservation,
    max_concurrent_envs=16,  # Scale up for parallel RL
)
```

```python
from concurrent.futures import ThreadPoolExecutor
from dc_ops_env import DcOpsAction, DcOpsEnv

def run_episode(scenario_id: str):
    with DcOpsEnv(base_url="http://localhost:8000") as env:
        result = env.reset(scenario=scenario_id)
        total_reward = 0.0
        while not result.done:
            result = env.step(DcOpsAction(command="check_status"))
            total_reward += result.reward
        return scenario_id, total_reward

# Run 8 episodes concurrently
scenarios = ["A1", "A2", "A4", "B1", "B3", "B4", "A2", "B4"]
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(run_episode, scenarios))
```

---

## Scenarios

6 operational scenarios across 3 difficulty levels:

| ID | Scenario | Difficulty | Type | Fault |
|----|----------|------------|------|-------|
| A1 | Cooling Setpoint Optimization | Easy | Thermal | CRACs at 15°C (wasteful) |
| A2 | Thermal Event Response | Medium | Thermal | CRAC-3 compressor failure |
| A4 | CRAC Failure Cascade | Hard | Thermal | CRAC-1 compressor + CRAC-3 fan |
| B1 | UPS Alarm Response | Medium | Power | UPS transferred to battery |
| B3 | Generator Test Protocol | Easy | Power | None (routine test) |
| B4 | Power Failure Cascade | Hard | Power | Utility loss + extended gen warmup |

Reset with a specific scenario:
```python
result = env.reset(scenario="A2")           # By ID
result = env.reset(random_scenario=True)    # Random
result = env.reset(random_scenario=True, difficulty="hard")  # Random hard
```

---

## Configuration

### Built-in Facility Configs

Three YAML configurations are included:

| Config | Zones | Racks | IT Load | CRACs | Use Case |
|--------|-------|-------|---------|-------|----------|
| `default` | 2 | 20 | 160 kW | 4 × 70 kW | Standard facility |
| `small` | 1 | 10 | 80 kW | 2 × 70 kW | Edge / branch office |
| `large` | 4 | 60 | 600 kW | 8 × 100 kW | Multi-zone + GPU (H1) |

```python
from dc_ops_env.config import load_datacenter_config

# Load a built-in config
config = load_datacenter_config("small")

# Load a custom YAML file
config = load_datacenter_config("/path/to/my_datacenter.yaml")

# Use with environment
result = env.reset(scenario="A2", config=config)
```

### Custom YAML Configuration

Create your own datacenter layout:

```yaml
name: "My Custom Facility"
outside_temp_c: 35.0
outside_humidity_rh: 0.40
simulation_dt_s: 1.0

zones:
  - zone_id: zone_a
    containment_type: cold_aisle
    recirculation_factor: 0.08
    air_volume_m3: 500.0
    envelope_r_kw: 0.02
    initial_cold_aisle_temp_c: 20.0
    ashrae_class: A2
    racks:
      - { rack_id: A-01, row: A, position: 1, it_load_kw: 8.0,
          num_servers_2u: 20, server_thermal_mass_jk: 11100.0,
          airflow_cfm_per_kw: 160.0 }
      # ... more racks
    crac_units:
      - { unit_id: CRAC-1, rated_capacity_kw: 70.0,
          rated_return_temp_c: 24.0, capacity_slope_per_c: 0.03,
          max_airflow_cfm: 12000.0, fan_rated_power_kw: 5.0,
          cop_rated: 3.5, initial_setpoint_c: 18.0,
          initial_fan_speed_pct: 100.0, supply_temp_lag_s: 30.0 }

power:
  utility_voltage_v: 480.0
  utility_available: true
  ups_units:
    - { unit_id: UPS-1, rated_capacity_kw: 500.0,
        loss_c0: 0.013, loss_c1: 0.006, loss_c2: 0.011,
        battery_capacity_kwh: 8.3, battery_discharge_efficiency: 0.90,
        battery_aging_factor: 0.85, recharge_rate_kw: 5.0,
        initial_mode: double_conversion }
  pdus:
    - { pdu_id: PDU-A-01, voltage_ll_v: 208.0,
        max_current_per_phase_a: 24.0, num_phases: 3,
        efficiency: 0.98, continuous_derating: 0.80 }
  generator:
    gen_id: GEN-1
    rated_capacity_kw: 750.0
    start_delay_s: 4.0
    crank_time_s: 5.0
    warmup_time_s: 8.0
    fuel_tank_liters: 2000.0
    consumption_lph_full: 180.0
    cooldown_time_s: 300.0
  ats:
    ats_id: ATS-1
    transfer_time_ms: 100.0
    retransfer_delay_s: 300.0
```

See [data/datacenter_configs/](data/datacenter_configs/) for complete examples.

---

## TRL / GRPO Training Integration

DC-Ops integrates directly with HuggingFace TRL's `GRPOTrainer` via the OpenEnv `environment_factory` pattern:

```python
from trl import GRPOTrainer, GRPOConfig
from dc_ops_env import DcOpsAction, DcOpsEnv

def dc_ops_environment_factory():
    """Factory that returns a DC-Ops environment instance."""
    env = DcOpsEnv(base_url="http://localhost:8000")
    return env

config = GRPOConfig(
    model_name_or_path="your-base-model",
    # ... training hyperparameters
)

trainer = GRPOTrainer(
    config=config,
    environments=dc_ops_environment_factory,
    # ... other args
)

trainer.train()
```

For multi-environment parallel training, run multiple servers or increase `max_concurrent_envs` and spawn concurrent clients.

---

## Deploy to HuggingFace Spaces

### Using OpenEnv CLI

The simplest way to deploy:

```bash
# From the dc_ops_env/ directory (where openenv.yaml is located)
cd dc_ops_env

# Login to HuggingFace (if not already)
huggingface-cli login

# Push to HuggingFace Spaces
openenv push

# Or with options
openenv push --repo-id your-username/dc-ops-env --private
openenv push --namespace your-org
```

### What Gets Deployed

The `openenv push` command:
1. Validates the `openenv.yaml` manifest
2. Builds a Docker Space on HuggingFace
3. Uploads all environment code

Your deployed Space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The Space includes:
- **Web Interface** at `/web` — Interactive scenario browser and dashboard viewer
- **API Documentation** at `/docs` — Full OpenAPI/Swagger interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent session endpoint for agent connections

### Connecting to a Deployed Space

```python
from dc_ops_env import DcOpsAction, DcOpsEnv

# Connect to your HuggingFace Space
space_url = "https://your-username-dc-ops-env.hf.space"

async with DcOpsEnv(base_url=space_url) as env:
    result = await env.reset(scenario="A2")
    print(result.observation.dashboard)
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--directory`, `-d` | Directory containing the OpenEnv environment (default: current) |
| `--repo-id`, `-r` | Repository ID `username/repo-name` (default: from openenv.yaml) |
| `--base-image`, `-b` | Override base Docker image |
| `--private` | Deploy as a private Space |
| `--namespace` | HuggingFace namespace (user or org) |

---

## Development

### Running Tests

```bash
# All tests (254 tests)
uv run pytest tests/ -v

# Specific test modules
uv run pytest tests/test_thermal.py -v      # Thermal physics
uv run pytest tests/test_power.py -v        # Power systems
uv run pytest tests/test_actions.py -v      # Command parser
uv run pytest tests/test_rewards.py -v      # Reward function
uv run pytest tests/test_scenarios.py -v    # Scenario framework
uv run pytest tests/test_integration.py -v  # End-to-end episodes

# With coverage
uv run pytest tests/ --cov=dc_ops_env --cov-report=term-missing
```

### Direct Environment Testing (No Server)

Test the environment logic without the HTTP/WebSocket layer:

```python
from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
from dc_ops_env.models import DcOpsAction

env = DcOpsEnvironment()
obs = env.reset(scenario="A2")
print(obs.dashboard)

obs = env.step(DcOpsAction(command="diagnose CRAC-3"))
print(f"Reward: {obs.reward}")
print(obs.dashboard)
```

### Running the Server Locally

```bash
# Via entry point (recommended)
uv run server

# With custom port
uv run server --port 8001

# Via uvicorn directly (with auto-reload for development)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Production (multi-worker)
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Project Structure

```
dc_ops_env/
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Dependencies and metadata
├── README.md                       # This file (HF Space README)
├── __init__.py                     # Exports: DcOpsEnv, DcOpsAction, DcOpsObservation
├── config.py                       # Physical constants, ASHRAE limits, YAML loader
├── models.py                       # Pydantic Action/Observation models
├── client.py                       # DcOpsEnv (EnvClient subclass)
├── simulation/
│   ├── thermal.py                  # RC thermal network (zones, racks, CRACs)
│   ├── power.py                    # UPS, PDU, generator, ATS models
│   └── types.py                    # Runtime state dataclasses
├── scenarios/
│   ├── base.py                     # Abstract Scenario + ProcedureRule
│   ├── registry.py                 # Scenario registration and selection
│   ├── thermal_scenarios.py        # A1, A2, A4
│   └── power_scenarios.py          # B1, B3, B4
├── rewards/
│   └── reward_function.py          # 6-component composite reward
├── rendering/
│   └── dashboard.py                # State → text dashboard
├── actions/
│   └── parser.py                   # Deterministic command parser
├── server/
│   ├── dc_ops_env_environment.py   # OpenEnv Environment implementation
│   ├── app.py                      # FastAPI application
│   └── Dockerfile                  # Container image
├── data/
│   └── datacenter_configs/         # YAML facility definitions
│       ├── default.yaml            # 2 zones, 20 racks, 160 kW
│       ├── small_facility.yaml     # 1 zone, 10 racks, 80 kW
│       └── large_facility.yaml     # 4 zones, 60 racks, 600 kW
└── tests/                          # 254 tests across 6 modules
    ├── test_thermal.py
    ├── test_power.py
    ├── test_actions.py
    ├── test_rewards.py
    ├── test_scenarios.py
    └── test_integration.py
```

## License

BSD-style license. See [LICENSE](../LICENSE) for details.
