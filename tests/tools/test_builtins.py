from __future__ import annotations

import sys

import pytest

from agent_core.tools.builtins.read_file import ReadFileParams, ReadFileTool
from agent_core.tools.builtins.shell_exec import ShellExecParams, ShellExecTool


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
