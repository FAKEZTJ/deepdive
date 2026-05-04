from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_core.mcp.client import MCPClient, MCPServerConfig
from agent_core.mcp.manager import MCPManager
from agent_core.mcp.tool import MCPTool
from agent_core.tools.registry import ToolRegistry


class _FakeToolResult:
    def __init__(self, content, is_error: bool = False):
        self.content = content
        self.isError = is_error


class _FakeSession:
    def __init__(self):
        self.called_with: list[tuple[str, dict]] = []

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="list_files",
                    description="List files",
                    inputSchema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )
            ]
        )

    async def call_tool(self, name: str, arguments: dict):
        self.called_with.append((name, arguments))
        return _FakeToolResult(
            content=[
                SimpleNamespace(text="first line"),
                SimpleNamespace(kind="image"),
            ]
        )


@pytest.mark.anyio
async def test_mcp_tool_uses_prefixed_name_and_passthrough_schema():
    client = MCPClient(
        MCPServerConfig(
            name="fs",
            command="npx",
            args=["server-filesystem"],
            permission="read_only",
        )
    )
    tool = MCPTool(
        client=client,
        mcp_name="list_files",
        description="List files",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    assert tool.name == "fs__list_files"
    assert tool.permission == "read_only"
    assert tool.to_schema().parameters["required"] == ["path"]
    assert tool.parse_input({"path": "/tmp"}) == {"path": "/tmp"}


@pytest.mark.anyio
async def test_mcp_client_lists_tools_and_calls_tool_without_pydantic_translation():
    client = MCPClient(
        MCPServerConfig(
            name="fs",
            command="npx",
            args=["server-filesystem"],
            permission="read_only",
        )
    )
    client._session = _FakeSession()

    tools = await client.list_tools()
    result = await client.call_tool("list_files", {"path": "/tmp"})

    assert [tool.name for tool in tools] == ["fs__list_files"]
    assert result.is_error is False
    assert result.content == "first line\n[non-text content: SimpleNamespace]"
    assert result.metadata["mcp_server"] == "fs"
    assert client._session.called_with == [("list_files", {"path": "/tmp"})]


@pytest.mark.anyio
async def test_mcp_client_call_tool_converts_exceptions_to_tool_result():
    class _CrashSession:
        async def call_tool(self, name: str, arguments: dict):
            raise RuntimeError("boom")

    client = MCPClient(
        MCPServerConfig(name="fs", command="npx", args=["server-filesystem"])
    )
    client._session = _CrashSession()

    result = await client.call_tool("list_files", {"path": "/tmp"})

    assert result.is_error is True
    assert result.content == "MCP call error: boom"


@pytest.mark.anyio
async def test_mcp_manager_registers_tools_from_multiple_clients(monkeypatch):
    class _FakeClient:
        def __init__(self, config: MCPServerConfig):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        async def list_tools(self):
            return [
                MCPTool(
                    client=self,
                    mcp_name="list_files",
                    description="List files",
                    input_schema={
                        "type": "object",
                        "properties": {},
                    },
                )
            ]

    monkeypatch.setattr("agent_core.mcp.manager.MCPClient", _FakeClient)

    registry = ToolRegistry()
    manager = MCPManager(
        registry=registry,
        servers=[
            MCPServerConfig(name="fs", command="npx", args=["fs"]),
            MCPServerConfig(name="github", command="npx", args=["gh"], permission="write"),
        ],
    )

    async with manager:
        assert registry.has("fs__list_files")
        assert registry.has("github__list_files")
        exported = registry.schemas({"read_only", "write"})
        assert [tool.name for tool in exported] == ["fs__list_files", "github__list_files"]
