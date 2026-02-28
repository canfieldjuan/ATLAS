"""
MCP SSE Bearer-token middleware.

Reads ATLAS_MCP_AUTH_TOKEN from the environment.
If the variable is not set, the middleware is a no-op (safe for stdio mode).
If it is set, every HTTP request to the SSE app must carry:

    Authorization: Bearer <token>

Returns HTTP 401 otherwise.  Only active in SSE transport.
"""

import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("atlas.mcp.auth")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validate Authorization: Bearer header against ATLAS_MCP_AUTH_TOKEN."""

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            logger.warning("MCP SSE rejected request: missing Bearer token from %s", request.client)
            return Response("Unauthorized", status_code=401)
        provided = auth.split(" ", 1)[1].strip()
        if not hmac.compare_digest(provided, self._token):
            logger.warning("MCP SSE rejected request: invalid token from %s", request.client)
            return Response("Unauthorized", status_code=401)
        return await call_next(request)


def apply_auth_middleware(app):
    """
    Wrap a Starlette app with bearer auth if ATLAS_MCP_AUTH_TOKEN is set.
    Returns the original app unchanged when the variable is absent.
    """
    token = os.environ.get("ATLAS_MCP_AUTH_TOKEN", "").strip()
    if not token:
        logger.info("ATLAS_MCP_AUTH_TOKEN not set -- MCP SSE running without auth")
        return app
    logger.info("MCP SSE bearer auth enabled")
    return BearerAuthMiddleware(app, token=token)


def run_sse_with_auth(mcp, host: str, port: int) -> None:
    """
    Build the SSE app, apply auth middleware, then run uvicorn.
    Replaces the mcp.run(transport='sse') call in each server's entry point.
    """
    import anyio
    import uvicorn

    starlette_app = mcp.sse_app()
    secured_app = apply_auth_middleware(starlette_app)

    async def _serve():
        config = uvicorn.Config(
            secured_app,
            host=host,
            port=port,
            log_level=mcp.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    anyio.run(_serve)
