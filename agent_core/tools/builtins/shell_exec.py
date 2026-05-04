# agent_core/tools/builtins/shell_exec.py
import asyncio
from pydantic import BaseModel, Field

from agent_core.tools.base import Tool, ToolResult


class ShellExecParams(BaseModel):
    command: str = Field(..., description="Shell command to execute, e.g. 'ls -la'")
    timeout_seconds: float = Field(
        default=30.0,
        description="Kill the process after this many seconds",
    )


class ShellExecTool(Tool[ShellExecParams]):
    name = "shell_exec"
    description = (
        "Execute a shell command and return stdout/stderr. "
        "Use this for filesystem queries, running scripts, etc."
    )
    params_model = ShellExecParams
    permission = "dangerous"

    MAX_OUTPUT_BYTES = 50_000

    async def execute(self, params: ShellExecParams) -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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

            output_parts = []
            if stdout_text:
                output_parts.append(f"STDOUT:\n{stdout_text}")
            if stderr_text:
                output_parts.append(f"STDERR:\n{stderr_text}")
            output_parts.append(f"Exit code: {proc.returncode}")

            return ToolResult(
                content="\n\n".join(output_parts),
                is_error=proc.returncode != 0,
                metadata={"exit_code": proc.returncode},
            )
        except Exception as exc:
            return ToolResult(content=f"Execution error: {exc}", is_error=True)
