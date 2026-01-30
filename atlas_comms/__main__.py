"""
Entry point for running atlas_comms as a standalone service.

Usage:
    python -m atlas_comms

Or with uvicorn:
    uvicorn atlas_comms.api.main:app --host 0.0.0.0 --port 5003
"""

import uvicorn

from .core.config import comms_settings


def main():
    """Run the atlas_comms FastAPI server."""
    uvicorn.run(
        "atlas_comms.api.main:app",
        host=comms_settings.server.host,
        port=comms_settings.server.port,
        reload=False,
        log_level=comms_settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
