from __future__ import annotations

from contextlib import AsyncExitStack

from agent_core.mcp.client import MCPClient, MCPServerConfig
from agent_core.tools.registry import ToolRegistry


class MCPManager:
    """Manage multiple MCP server connections and register their tools."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        servers: list[MCPServerConfig],
    ):
        self._registry = registry
        self._configs = servers
        self._exit_stack: AsyncExitStack | None = None
        self._clients: list[MCPClient] = []

    async def __aenter__(self) -> MCPManager:
        self._exit_stack = AsyncExitStack()
        for config in self._configs:
            client = MCPClient(config)
            await self._exit_stack.enter_async_context(client)
            self._clients.append(client)
            for tool in await client.list_tools():
                self._registry.register(tool)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._clients.clear()
