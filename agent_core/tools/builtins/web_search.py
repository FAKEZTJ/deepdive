from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from agent_core.tools.base import Tool, ToolResult


class WebSearchParams(BaseModel):
    query: str = Field(..., description="The search query")
    max_results: int = Field(default=5, ge=1, le=20)


class WebSearchTool(Tool[WebSearchParams]):
    name = "web_search"
    description = (
        "Search the web for information. Returns a list of results with "
        "title, URL, and snippet."
    )
    params_model = WebSearchParams
    permission = "read_only"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self._api_key:
            raise ValueError("TAVILY_API_KEY not set")

    async def execute(self, params: WebSearchParams) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._api_key,
                        "query": params.query,
                        "max_results": params.max_results,
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            return ToolResult(content=f"Search failed: {exc}", is_error=True)

        results = data.get("results", [])
        if not results:
            return ToolResult(content="No results found.")

        formatted: list[str] = []
        metadata_results: list[dict[str, str]] = []
        for result in results:
            snippet = result.get("content", "")[:500]
            formatted.append(
                f"### {result['title']}\n"
                f"URL: {result['url']}\n"
                f"{snippet}"
            )
            metadata_results.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": snippet,
                }
            )
        return ToolResult(
            content="\n\n".join(formatted),
            metadata={"query": params.query, "results": metadata_results},
        )
