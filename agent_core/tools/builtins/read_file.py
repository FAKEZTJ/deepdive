# agent_core/tools/builtins/read_file.py
from pathlib import Path
from pydantic import BaseModel, Field

from agent_core.tools.base import Tool, ToolResult


class ReadFileParams(BaseModel):
    path: str = Field(..., description="Absolute or relative file path")
    max_bytes: int = Field(
        default=100_000,
        description="Maximum bytes to read. Larger files are truncated.",
    )


class ReadFileTool(Tool[ReadFileParams]):
    name = "read_file"
    description = "Read the contents of a text file from disk. Returns up to max_bytes bytes."
    params_model = ReadFileParams
    permission = "read_only"

    async def execute(self, params: ReadFileParams) -> ToolResult:
        try:
            p = Path(params.path).expanduser()
            if not p.exists():
                return ToolResult(content=f"File not found: {params.path}", is_error=True)
            if not p.is_file():
                return ToolResult(content=f"Not a regular file: {params.path}", is_error=True)
            
            data = p.read_bytes()[:params.max_bytes]
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return ToolResult(
                    content=f"File is not UTF-8 text: {params.path}",
                    is_error=True,
                )
            
            truncated_note = ""
            if p.stat().st_size > params.max_bytes:
                truncated_note = f"\n\n[truncated to {params.max_bytes} bytes]"
            
            return ToolResult(
                content=text + truncated_note,
                metadata={"size": p.stat().st_size},
            )
        except Exception as exc:
            return ToolResult(content=f"Read error: {exc}", is_error=True)