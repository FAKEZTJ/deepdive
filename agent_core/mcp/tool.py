from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agent_core.tools.base import Tool, ToolResult
from agent_core.types import ToolSchema

if TYPE_CHECKING:
    from agent_core.mcp.client import MCPClient


class _PassthroughParams(BaseModel):
    """Allow arbitrary MCP arguments and forward them to the server unchanged."""

    model_config = {"extra": "allow"}


class MCPTool(Tool[dict[str, Any]]):
    """Adapt an MCP tool into the local Tool interface."""

    params_model = _PassthroughParams

    def __init__(
        self,
        *,
        client: MCPClient,
        mcp_name: str,
        description: str,
        input_schema: dict[str, Any],
    ):
        self._client = client
        self.name = f"{client.config.name}__{mcp_name}"
        self._mcp_name = mcp_name
        self.description = description
        self._input_schema = input_schema
        self.permission = client.config.permission

    def to_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self._input_schema,
        )

    def parse_input(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        return raw_input

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        return await self._client.call_tool(self._mcp_name, params)
