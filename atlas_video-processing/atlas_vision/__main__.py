"""
Entry point for Atlas Vision.

Usage: python -m atlas_vision
"""

import logging
import sys

import uvicorn

from .core.config import settings


def setup_logging() -> None:
    """Configure logging."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main() -> int:
    """Run the Atlas Vision server."""
    setup_logging()
    logger = logging.getLogger("atlas.vision")

    logger.info("Starting Atlas Vision server")
    logger.info("Host: %s, Port: %d", settings.server.host, settings.server.port)

    try:
        uvicorn.run(
            "atlas_vision.api.main:app",
            host=settings.server.host,
            port=settings.server.port,
            reload=settings.server.debug,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Shutting down")
    except Exception as e:
        logger.exception("Server error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
