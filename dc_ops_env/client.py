# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DC-Ops Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DcOpsAction, DcOpsObservation


class DcOpsEnv(
    EnvClient[DcOpsAction, DcOpsObservation, State]
):
    """
    Client for the DC-Ops Environment.

    Connects to the environment server over WebSocket and provides
    reset/step/state methods for interacting with the datacenter simulation.

    Example:
        >>> async with DcOpsEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.dashboard)
        ...
        ...     result = await client.step(DcOpsAction(command="diagnose CRAC-1"))
        ...     print(result.observation.dashboard)
    """

    def _step_payload(self, action: DcOpsAction) -> Dict:
        """Convert DcOpsAction to JSON payload for step message."""
        payload = {"command": action.command}
        if action.reasoning:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DcOpsObservation]:
        """Parse server response into StepResult[DcOpsObservation]."""
        obs_data = payload.get("observation", {})
        observation = DcOpsObservation(
            dashboard=obs_data.get("dashboard", ""),
            available_actions=obs_data.get("available_actions", []),
            alert=obs_data.get("alert", ""),
            scenario_type=obs_data.get("scenario_type", ""),
            steps_remaining=obs_data.get("steps_remaining", 0),
            action_result=obs_data.get("action_result", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
