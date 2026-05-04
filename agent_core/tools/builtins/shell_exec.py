from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from agent_core.tools.base import Tool, ToolResult


class ShellExecParams(BaseModel):
    command: str = Field(..., description="Shell command to execute")
    timeout_seconds: float = Field(default=30.0, le=300)
    working_dir: str | None = Field(
        default=None,
        description="Working directory. If not specified, uses an isolated temp dir.",
    )


class ShellExecTool(Tool[ShellExecParams]):
    name = "shell_exec"
    description = "Execute a shell command and return output. Sandboxed to a temp directory by default."
    params_model = ShellExecParams
    permission = "dangerous"

    MAX_OUTPUT_BYTES = 50_000
    SAFE_ENV_VARS = {
        "PATH",
        "LANG",
        "LC_ALL",
        "TERM",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
    }

    def __init__(self, *, allowed_working_root: str | None = None):
        self._allowed_root = (
            Path(allowed_working_root).expanduser().resolve()
            if allowed_working_root
            else None
        )

    async def execute(self, params: ShellExecParams) -> ToolResult:
        cleanup_temp = False

        if params.working_dir:
            cwd = Path(params.working_dir).expanduser().resolve()
            if self._allowed_root is not None:
                try:
                    cwd.relative_to(self._allowed_root)
                except ValueError:
                    return ToolResult(
                        content=f"working_dir outside allowed_root: {cwd}",
                        is_error=True,
                    )
        else:
            temp_root = None
            if self._allowed_root is not None:
                self._allowed_root.mkdir(parents=True, exist_ok=True)
                temp_root = str(self._allowed_root)
            cwd = Path(tempfile.mkdtemp(prefix="agent_shell_", dir=temp_root))
            cleanup_temp = True

        safe_env = {key: value for key, value in os.environ.items() if key in self.SAFE_ENV_VARS}
        safe_env["HOME"] = str(cwd)
        safe_env["USERPROFILE"] = str(cwd)

        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=safe_env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=params.timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    content=f"Command timed out after {params.timeout_seconds}s",
                    is_error=True,
                )

            stdout_text = stdout[:self.MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            stderr_text = stderr[:self.MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")

            parts: list[str] = []
            if stdout_text:
                parts.append(f"STDOUT:\n{stdout_text}")
            if stderr_text:
                parts.append(f"STDERR:\n{stderr_text}")
            parts.append(f"Exit code: {proc.returncode}")
            parts.append(f"Working dir: {cwd}")

            return ToolResult(
                content="\n\n".join(parts),
                is_error=proc.returncode != 0,
                metadata={"exit_code": proc.returncode, "cwd": str(cwd)},
            )
        except Exception as exc:
            return ToolResult(content=f"Execution error: {exc}", is_error=True)
        finally:
            if cleanup_temp:
                shutil.rmtree(cwd, ignore_errors=True)
