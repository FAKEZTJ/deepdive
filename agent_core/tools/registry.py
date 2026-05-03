# agent_core/tools/registry.py
from __future__ import annotations

from typing import Iterable

from agent_core.tools.base import Tool
from agent_core.types import ToolSchema


class ToolRegistry:
    """工具注册表。
    
    Day 2 只做最简的注册和查找。Day 3 加权限过滤、命名空间。
    """

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

    def schemas(self) -> list[ToolSchema]:
        """所有工具的 schema，用于传给 LLM。"""
        return [tool.to_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)