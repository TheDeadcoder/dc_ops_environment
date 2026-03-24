# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Dc Ops Env Environment.

This module creates an HTTP server that exposes the DcOpsEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DcOpsAction, DcOpsObservation
    from .dc_ops_env_environment import DcOpsEnvironment
except ModuleNotFoundError:
    from models import DcOpsAction, DcOpsObservation
    from server.dc_ops_env_environment import DcOpsEnvironment


# Create the app with web interface and README integration
app = create_app(
    DcOpsEnvironment,
    DcOpsAction,
    DcOpsObservation,
    env_name="dc_ops_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Mount custom DC-Ops dashboard UI at /web
_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/web")
async def web_ui():
    """Serve the DC-Ops operations console."""
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m dc_ops_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn dc_ops_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
