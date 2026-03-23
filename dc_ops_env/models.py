# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic models for the DC-Ops Environment.

Action: Natural-language operator commands (e.g., "adjust_setpoint CRAC-1 20").
Observation: Text dashboard + structured metadata for the LLM agent.

These use OpenEnv's Action/Observation base classes which enforce
`extra="forbid"` — only declared fields are allowed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DcOpsAction(Action):
    """Operator command issued by the LLM agent.

    The agent reads the dashboard observation and responds with a command string.
    Commands follow the format: `command_name [target] [value]`

    Examples:
        - "diagnose CRAC-3"
        - "adjust_setpoint CRAC-1 20"
        - "increase_fan_speed CRAC-2 80"
        - "start_generator"
        - "acknowledge_alarm"
        - "escalate"
    """

    command: str = Field(
        ...,
        description="Operator command (e.g., 'diagnose CRAC-3', 'adjust_setpoint CRAC-1 20')",
    )
    reasoning: str = Field(
        default="",
        description="Optional chain-of-thought reasoning from the agent",
    )


class DcOpsObservation(Observation):
    """Text-based monitoring dashboard observation.

    The 'dashboard' field contains the full text rendering of the current
    datacenter state — formatted like a real operator's monitoring screen.
    This is the primary field the LLM agent reads.

    Structured data is available in the inherited 'metadata' dict.
    """

    dashboard: str = Field(
        default="",
        description="Text-rendered monitoring dashboard",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Valid commands the agent can issue",
    )
    alert: str = Field(
        default="",
        description="Current active alert message, if any",
    )
    scenario_type: str = Field(
        default="",
        description="Type of scenario (thermal, power, network, incident)",
    )
    steps_remaining: int = Field(
        default=0,
        description="Steps left in episode budget",
    )
    action_result: str = Field(
        default="",
        description="Feedback from the last action (success/error message)",
    )
