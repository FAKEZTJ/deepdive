from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from agent_core.tools.base import Tool, ToolResult


class HttpGetParams(BaseModel):
    url: str = Field(..., description="URL to fetch (http/https only)")
    max_bytes: int = Field(
        default=200_000,
        description="Truncate response after this many bytes",
    )
    timeout_seconds: float = Field(default=15.0)


class HttpGetTool(Tool[HttpGetParams]):
    name = "http_get"
    description = (
        "Fetch a web page or API endpoint via HTTP GET. Returns the response body "
        "(truncated to max_bytes). Use for fetching documentation, API responses, etc."
    )
    params_model = HttpGetParams
    permission = "read_only"

    async def execute(self, params: HttpGetParams) -> ToolResult:
        if not (params.url.startswith("http://") or params.url.startswith("https://")):
            return ToolResult(content="Only http/https URLs allowed", is_error=True)

        try:
            async with httpx.AsyncClient(
                timeout=params.timeout_seconds,
                follow_redirects=True,
            ) as client:
                resp = await client.get(params.url)
        except httpx.HTTPError as exc:
            return ToolResult(content=f"HTTP error: {exc}", is_error=True)

        body = resp.text[:params.max_bytes]
        truncated = "\n\n[truncated]" if len(resp.text) > params.max_bytes else ""

        return ToolResult(
            content=(
                f"Status: {resp.status_code}\n"
                f"Content-Type: {resp.headers.get('content-type', 'unknown')}\n"
                f"\n{body}{truncated}"
            ),
            is_error=resp.status_code >= 400,
            metadata={"status_code": resp.status_code, "url": str(resp.url)},
        )
