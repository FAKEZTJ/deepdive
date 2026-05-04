from __future__ import annotations

import sys

import pytest

from agent_core.tools.builtins.http_get import HttpGetParams, HttpGetTool
from agent_core.tools.builtins.read_file import ReadFileParams, ReadFileTool
from agent_core.tools.builtins.shell_exec import ShellExecParams, ShellExecTool
from agent_core.tools.builtins.web_search import WebSearchParams, WebSearchTool
from agent_core.tools.builtins.write_file import WriteFileParams, WriteFileTool


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_data=None, text: str = "", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}
        self.url = "https://example.com/resource"

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "bad status",
                request=httpx.Request("GET", self.url),
                response=httpx.Response(self.status_code, request=httpx.Request("GET", self.url)),
            )

    def json(self):
        return self._json_data


class _FakeAsyncClient:
    def __init__(self, *, post_response=None, get_response=None):
        self._post_response = post_response
        self._get_response = get_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def post(self, url, json):
        return self._post_response

    async def get(self, url):
        return self._get_response


@pytest.mark.anyio
async def test_read_file_reports_missing_path(tmp_path):
    tool = ReadFileTool()

    result = await tool.execute(ReadFileParams(path=str(tmp_path / "missing.txt")))

    assert result.is_error is True
    assert "File not found:" in result.content


@pytest.mark.anyio
async def test_read_file_truncates_large_content(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    tool = ReadFileTool()

    result = await tool.execute(ReadFileParams(path=str(path), max_bytes=4))

    assert result.is_error is False
    assert result.content == "abcd\n\n[truncated to 4 bytes]"
    assert result.metadata["size"] == 10


@pytest.mark.anyio
async def test_read_file_rejects_non_utf8_content(tmp_path):
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\xff\xfe\x00")
    tool = ReadFileTool()

    result = await tool.execute(ReadFileParams(path=str(path)))

    assert result.is_error is True
    assert "File is not UTF-8 text:" in result.content


@pytest.mark.anyio
async def test_shell_exec_marks_nonzero_exit_as_error():
    tool = ShellExecTool()
    command = f'"{sys.executable}" -c "import sys; sys.stderr.write(\'bad\\\\n\'); sys.exit(2)"'

    result = await tool.execute(ShellExecParams(command=command, timeout_seconds=5.0))

    assert result.is_error is True
    assert "STDERR:\nbad" in result.content
    assert "Exit code: 2" in result.content
    assert result.metadata["exit_code"] == 2


@pytest.mark.anyio
async def test_shell_exec_times_out():
    tool = ShellExecTool()
    command = f'"{sys.executable}" -c "import time; time.sleep(1)"'

    result = await tool.execute(ShellExecParams(command=command, timeout_seconds=0.05))

    assert result.is_error is True
    assert result.content == "Command timed out after 0.05s"


@pytest.mark.anyio
async def test_shell_exec_uses_isolated_temp_dir_and_cleans_it_up():
    tool = ShellExecTool()
    command = (
        f'"{sys.executable}" -c '
        '"import os; print(os.environ.get(\'HOME\', \'\')); print(os.getcwd())"'
    )

    result = await tool.execute(ShellExecParams(command=command, timeout_seconds=5.0))

    assert result.is_error is False
    lines = result.content.splitlines()
    home_value = next(line for line in lines if line and "STDOUT:" not in line)
    working_dir = result.metadata["cwd"]
    assert home_value == working_dir
    assert f"Working dir: {working_dir}" in result.content

    from pathlib import Path

    assert Path(working_dir).exists() is False


@pytest.mark.anyio
async def test_shell_exec_rejects_working_dir_outside_allowed_root(tmp_path):
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    tool = ShellExecTool(allowed_working_root=str(allowed_root))

    result = await tool.execute(
        ShellExecParams(
            command=f'"{sys.executable}" -c "print(123)"',
            timeout_seconds=5.0,
            working_dir=str(outside),
        )
    )

    assert result.is_error is True
    assert "working_dir outside allowed_root:" in result.content


@pytest.mark.anyio
async def test_shell_exec_creates_temp_dir_inside_allowed_root_when_working_dir_missing(tmp_path):
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir()
    tool = ShellExecTool(allowed_working_root=str(allowed_root))

    result = await tool.execute(
        ShellExecParams(command=f'"{sys.executable}" -c "print(123)"', timeout_seconds=5.0)
    )

    cwd = result.metadata["cwd"]
    assert result.is_error is False
    assert str(allowed_root.resolve()) in cwd


def test_shell_exec_timeout_has_hard_upper_bound():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ShellExecParams(command="echo hi", timeout_seconds=301.0)


@pytest.mark.anyio
async def test_write_file_overwrites_and_appends(tmp_path):
    target = tmp_path / "notes" / "log.txt"
    tool = WriteFileTool(allowed_root=str(tmp_path))

    overwrite = await tool.execute(
        WriteFileParams(path=str(target), content="hello", mode="overwrite")
    )
    append = await tool.execute(
        WriteFileParams(path=str(target), content=" world", mode="append")
    )

    assert overwrite.is_error is False
    assert append.is_error is False
    assert target.read_text(encoding="utf-8") == "hello world"


@pytest.mark.anyio
async def test_write_file_rejects_path_outside_allowed_root(tmp_path):
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir()
    outside = tmp_path / "outside.txt"
    tool = WriteFileTool(allowed_root=str(allowed_root))

    result = await tool.execute(
        WriteFileParams(path=str(outside), content="nope", mode="overwrite")
    )

    assert result.is_error is True
    assert "is outside allowed root" in result.content


@pytest.mark.anyio
async def test_http_get_rejects_non_http_urls():
    tool = HttpGetTool()

    result = await tool.execute(HttpGetParams(url="file:///tmp/test.txt"))

    assert result.is_error is True
    assert result.content == "Only http/https URLs allowed"


@pytest.mark.anyio
async def test_http_get_returns_body_and_marks_http_errors(monkeypatch):
    import agent_core.tools.builtins.http_get as http_get_module

    def make_client(*args, **kwargs):
        return _FakeAsyncClient(
            get_response=_FakeResponse(
                status_code=404,
                text="missing page",
                headers={"content-type": "text/plain"},
            )
        )

    monkeypatch.setattr(http_get_module.httpx, "AsyncClient", make_client)
    tool = HttpGetTool()

    result = await tool.execute(HttpGetParams(url="https://example.com/missing"))

    assert result.is_error is True
    assert "Status: 404" in result.content
    assert "missing page" in result.content
    assert result.metadata["status_code"] == 404


@pytest.mark.anyio
async def test_web_search_formats_results(monkeypatch):
    import agent_core.tools.builtins.web_search as web_search_module

    def make_client(*args, **kwargs):
        return _FakeAsyncClient(
            post_response=_FakeResponse(
                json_data={
                    "results": [
                        {
                            "title": "Result One",
                            "url": "https://example.com/1",
                            "content": "Snippet one",
                        },
                        {
                            "title": "Result Two",
                            "url": "https://example.com/2",
                            "content": "Snippet two",
                        },
                    ]
                }
            )
        )

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", make_client)
    tool = WebSearchTool(api_key="test-key")

    result = await tool.execute(WebSearchParams(query="agent loop", max_results=2))

    assert result.is_error is False
    assert "### Result One" in result.content
    assert "URL: https://example.com/1" in result.content
    assert "Snippet two" in result.content


def test_web_search_loads_api_key_from_dotenv(monkeypatch):
    import importlib

    import dotenv
    import agent_core.tools.builtins.web_search as web_search_module

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def fake_load_dotenv():
        monkeypatch.setenv("TAVILY_API_KEY", "from-dotenv")
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    importlib.reload(web_search_module)

    tool = web_search_module.WebSearchTool()

    assert tool._api_key == "from-dotenv"
