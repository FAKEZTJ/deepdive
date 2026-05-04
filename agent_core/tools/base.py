from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from agent_core.types import ToolSchema

TParams = TypeVar("TParams", bound=BaseModel)
ToolPermission = Literal["read_only", "write", "dangerous"]


class ToolResult(BaseModel):
    """Structured tool execution result."""

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC, Generic[TParams]):
    """Base contract for all tools."""

    name: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]
    permission: ClassVar[ToolPermission] = "read_only"

    @abstractmethod
    async def execute(self, params: TParams) -> ToolResult:
        ...

    def to_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.params_model.model_json_schema(),
        )

    def parse_input(self, raw_input: dict[str, Any]) -> TParams:
        return self.params_model.model_validate(raw_input)
