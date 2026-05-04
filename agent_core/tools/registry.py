from __future__ import annotations

from typing import Iterable

from agent_core.tools.base import Tool, ToolPermission
from agent_core.types import ToolSchema


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas(
        self,
        allowed_permissions: set[ToolPermission] | None = None,
    ) -> list[ToolSchema]:
        tools = self._tools.values()
        if allowed_permissions is not None:
            tools = [tool for tool in tools if tool.permission in allowed_permissions]
        return [tool.to_schema() for tool in tools]

    def __len__(self) -> int:
        return len(self._tools)
