from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent_core.tools.base import Tool, ToolResult


class WriteFileParams(BaseModel):
    path: str = Field(..., description="File path. Can be absolute or relative.")
    content: str = Field(..., description="Content to write")
    mode: str = Field(default="overwrite", description="'overwrite' or 'append'")


class WriteFileTool(Tool[WriteFileParams]):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    params_model = WriteFileParams
    permission = "write"

    def __init__(self, *, allowed_root: str | None = None):
        self._allowed_root = (
            Path(allowed_root).expanduser().resolve() if allowed_root else None
        )

    async def execute(self, params: WriteFileParams) -> ToolResult:
        try:
            target = Path(params.path).expanduser().resolve()

            if self._allowed_root is not None:
                try:
                    target.relative_to(self._allowed_root)
                except ValueError:
                    return ToolResult(
                        content=f"Path '{target}' is outside allowed root '{self._allowed_root}'",
                        is_error=True,
                    )

            target.parent.mkdir(parents=True, exist_ok=True)

            if params.mode == "overwrite":
                target.write_text(params.content, encoding="utf-8")
            elif params.mode == "append":
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(params.content)
            else:
                return ToolResult(
                    content=f"Invalid mode '{params.mode}'. Use 'overwrite' or 'append'.",
                    is_error=True,
                )

            return ToolResult(
                content=f"Wrote {len(params.content)} chars to {target}",
                metadata={"path": str(target), "size": len(params.content)},
            )
        except Exception as exc:
            return ToolResult(content=f"Write error: {exc}", is_error=True)
