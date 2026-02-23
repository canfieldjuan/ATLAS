"""
MCP Tool Provider -- makes Atlas consume its own MCP servers.

Spawns each MCP server (CRM, Email, Calendar, Twilio) as a stdio subprocess,
discovers tools via the MCP protocol, and registers them in tool_registry so
the Atlas agent can use them transparently through execute_with_tools().
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from ..tools.base import ToolParameter, ToolResult
from ..tools.registry import tool_registry

logger = logging.getLogger("atlas.services.mcp_client")

# Map server name -> tool category for Atlas tool registry
_SERVER_CATEGORIES = {
    "crm": "crm",
    "email": "communication",
    "calendar": "scheduling",
    "twilio": "communication",
}

# Internal tool names that MUST NOT be overwritten by MCP servers.
# These are local tools (security cameras, device control) whose names
# collide with functionally different MCP tools (e.g. Twilio call
# recording vs. security camera recording).
_PROTECTED_TOOLS = frozenset({
    "start_recording",   # security camera recording
    "stop_recording",    # security camera recording
})


def _convert_input_schema(schema: dict[str, Any]) -> list[ToolParameter]:
    """Convert MCP JSON Schema inputSchema to Atlas ToolParameter list."""
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    params = []

    for name, prop in properties.items():
        prop_type = "string"
        raw_type = prop.get("type")

        if raw_type == "integer":
            prop_type = "int"
        elif raw_type == "number":
            prop_type = "float"
        elif raw_type == "boolean":
            prop_type = "boolean"
        elif raw_type == "array":
            # Atlas ToolParameter has no array type; map to string.
            # The LLM will read the description for format guidance.
            prop_type = "string"
        elif raw_type is None:
            # Handle Optional patterns: anyOf: [{type: "string"}, {type: "null"}]
            for variant in prop.get("anyOf", []):
                t = variant.get("type")
                if t and t != "null":
                    if t == "integer":
                        prop_type = "int"
                    elif t == "number":
                        prop_type = "float"
                    elif t == "boolean":
                        prop_type = "boolean"
                    else:
                        prop_type = "string"
                    break

        params.append(ToolParameter(
            name=name,
            param_type=prop_type,
            description=prop.get("description", ""),
            required=name in required_fields,
            default=prop.get("default"),
        ))

    return params


class MCPToolWrapper:
    """Wraps an MCP tool as an Atlas Tool protocol object.

    Delegates execution to the MCP session's call_tool() method.
    """

    def __init__(
        self,
        mcp_tool,  # mcp.types.Tool
        session: ClientSession,
        server_name: str,
        name_override: str | None = None,
    ):
        self._mcp_tool = mcp_tool
        self._session = session
        self._server_name = server_name
        self._name_override = name_override
        self._parameters = _convert_input_schema(mcp_tool.inputSchema)

    @property
    def name(self) -> str:
        return self._name_override or self._mcp_tool.name

    @property
    def description(self) -> str:
        return self._mcp_tool.description or ""

    @property
    def parameters(self) -> list[ToolParameter]:
        return self._parameters

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def category(self) -> str:
        return _SERVER_CATEGORIES.get(self._server_name, "utility")

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute via MCP session.call_tool()."""
        try:
            # Always use the original MCP tool name for the server call,
            # even if registered under a prefixed name in Atlas.
            result = await self._session.call_tool(
                self._mcp_tool.name, params
            )

            if result.isError:
                error_msg = ""
                for block in result.content:
                    if isinstance(block, TextContent):
                        error_msg = block.text
                        break
                return ToolResult(
                    success=False,
                    error="MCP_TOOL_ERROR",
                    message=error_msg or "MCP tool returned an error",
                )

            # Extract text content
            text_parts = []
            for block in result.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)

            combined = "\n".join(text_parts) if text_parts else ""

            # Try to parse as JSON
            try:
                data = json.loads(combined)
                if isinstance(data, dict):
                    return ToolResult(success=True, data=data)
                return ToolResult(success=True, data={"result": data})
            except (json.JSONDecodeError, ValueError):
                return ToolResult(
                    success=True,
                    data={"result": combined},
                    message=combined,
                )

        except Exception as e:
            logger.error("MCP tool %s execution failed: %s", self.name, e)
            return ToolResult(
                success=False,
                error="MCP_EXECUTION_ERROR",
                message=str(e),
            )


class MCPToolProvider:
    """Manages MCP server subprocess connections and tool registration.

    On start(), spawns each enabled MCP server as a stdio subprocess,
    connects via the MCP protocol, discovers tools, and registers them
    in tool_registry. Connections persist for Atlas's lifetime.
    """

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._tool_count = 0

    def _get_server_configs(self) -> list[tuple[str, StdioServerParameters]]:
        """Build server configs from settings."""
        from ..config import settings

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        python = sys.executable

        servers = [
            ("crm", settings.mcp.crm_enabled, "atlas_brain.mcp.crm_server"),
            ("email", settings.mcp.email_enabled, "atlas_brain.mcp.email_server"),
            ("calendar", settings.mcp.calendar_enabled, "atlas_brain.mcp.calendar_server"),
            ("twilio", settings.mcp.twilio_enabled, "atlas_brain.mcp.twilio_server"),
        ]

        configs = []
        for name, enabled, module in servers:
            if not enabled:
                logger.info("MCP server %s disabled, skipping", name)
                continue
            configs.append((name, StdioServerParameters(
                command=python,
                args=["-m", module],
                cwd=project_root,
            )))

        return configs

    async def start(self):
        """Start MCP connections in a background task."""
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("MCP client startup timed out after 30s")

    async def _run(self):
        """Connect to all MCP servers and keep connections alive."""
        server_count = 0
        try:
            async with AsyncExitStack() as stack:
                for name, params in self._get_server_configs():
                    try:
                        read_stream, write_stream = (
                            await stack.enter_async_context(stdio_client(params))
                        )
                        session = await stack.enter_async_context(
                            ClientSession(read_stream, write_stream)
                        )
                        await session.initialize()
                        self._sessions[name] = session

                        # Discover and register tools
                        tools_result = await session.list_tools()
                        for mcp_tool in tools_result.tools:
                            reg_name = None
                            if mcp_tool.name in _PROTECTED_TOOLS:
                                existing = tool_registry.get(mcp_tool.name)
                                if existing:
                                    reg_name = f"{name}_{mcp_tool.name}"
                                    logger.info(
                                        "MCP tool %s conflicts with protected "
                                        "internal tool, registering as %s",
                                        mcp_tool.name, reg_name,
                                    )
                            wrapper = MCPToolWrapper(
                                mcp_tool, session, name,
                                name_override=reg_name,
                            )
                            tool_registry.register(wrapper)
                            self._tool_count += 1

                        server_count += 1
                        logger.info(
                            "MCP server %s connected: %d tools",
                            name, len(tools_result.tools),
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to connect MCP server %s: %s", name, e
                        )

                logger.info(
                    "MCP client initialized: %d servers, %d tools",
                    server_count, self._tool_count,
                )
                self._ready.set()

                # Keep connections alive until shutdown
                await self._shutdown_event.wait()
        except Exception as e:
            logger.error("MCP client fatal error: %s", e)
            self._ready.set()

    async def shutdown(self):
        """Signal shutdown and wait for cleanup."""
        self._shutdown_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("MCP client shutdown timed out, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        logger.info("MCP client shutdown complete")


def _resolve_tools(specs: list[str]) -> list[str]:
    """Resolve tool name specs, preferring MCP names with internal fallback.

    Each spec is "preferred|fallback" or just "name".
    Tries each |-separated name left-to-right, returns first one found
    in tool_registry. If none found, uses the first name (may be
    registered later by MCP).

    Example:
        _resolve_tools(["send_email", "send_estimate|send_estimate_email"])
        # => ["send_email", "send_estimate"]  (if MCP registered "send_estimate")
        # => ["send_email", "send_estimate_email"]  (if MCP not available)
    """
    resolved = []
    for spec in specs:
        names = [n.strip() for n in spec.split("|")]
        found = False
        for name in names:
            if tool_registry.get(name) is not None:
                resolved.append(name)
                found = True
                break
        if not found:
            resolved.append(names[0])
    return resolved
