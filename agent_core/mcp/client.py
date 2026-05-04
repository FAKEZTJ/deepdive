from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from agent_core.tools.base import ToolPermission, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for starting an MCP server subprocess."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    permission: ToolPermission = "read_only"


class MCPClient:
    """Connection manager for one MCP server."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._exit_stack: AsyncExitStack | None = None
        self._session: Any | None = None

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._session is not None:
            return

        ClientSession, StdioServerParameters, stdio_client = _import_mcp_runtime()

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )

        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise

        self._exit_stack = stack
        self._session = session
        logger.info("MCP server '%s' connected", self.config.name)

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            logger.info("MCP server '%s' closed", self.config.name)

    async def list_tools(self) -> list["MCPTool"]:
        from agent_core.mcp.tool import MCPTool

        if self._session is None:
            raise RuntimeError("MCPClient not connected")

        result = await self._session.list_tools()
        tools: list[MCPTool] = []
        for mcp_tool in result.tools:
            tools.append(
                MCPTool(
                    client=self,
                    mcp_name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    input_schema=mcp_tool.inputSchema,
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._session is None:
            raise RuntimeError("MCPClient not connected")

        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as exc:
            return ToolResult(content=f"MCP call error: {exc}", is_error=True)

        text_parts: list[str] = []
        for content in result.content:
            if hasattr(content, "text"):
                text_parts.append(content.text)
            else:
                text_parts.append(f"[non-text content: {type(content).__name__}]")

        is_error = getattr(result, "isError", getattr(result, "is_error", False))
        return ToolResult(
            content="\n".join(text_parts),
            is_error=is_error,
            metadata={"mcp_server": self.config.name},
        )


def _import_mcp_runtime():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise RuntimeError(
            "The 'mcp' package is required for MCPClient.connect(). "
            "Add it to dependencies and run the environment sync step."
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client
