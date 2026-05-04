# agent_core/tools/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar
from pydantic import BaseModel

from agent_core.types import ToolSchema


# 工具参数类型，由具体 Tool 子类指定
TParams = TypeVar("TParams", bound=BaseModel)
ToolPermission = Literal["read_only", "write", "dangerous"]


class ToolResult(BaseModel):
    """工具执行结果。
    
    永远是结构化的，方便 trace 和持久化。
    content 字段是给 LLM 看的纯字符串。
    """
    content: str                    # 喂给 LLM 的文本
    is_error: bool = False          # 是否执行失败
    metadata: dict[str, Any] = {}   # trace 用，比如执行耗时、文件路径等


class Tool(ABC, Generic[TParams]):
    """工具接口。
    
    子类约定：
    - 类属性 ``name`` / ``description`` / ``params_model``
    - 实现 ``async def execute(params: TParams) -> ToolResult``
    
    示例:
        class ReadFileTool(Tool[ReadFileParams]):
            name = "read_file"
            description = "Read a text file from disk."
            params_model = ReadFileParams
            permission = "read_only"
            
            async def execute(self, params):
                with open(params.path) as f:
                    return ToolResult(content=f.read())
    """

    # 子类必须设置
    name: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]
    
    # 权限级别，Day 3 会在注册表和执行层真正使用
    permission: ClassVar[ToolPermission] = "read_only"

    @abstractmethod
    async def execute(self, params: TParams) -> ToolResult:
        """执行工具。失败时返回 ``ToolResult(is_error=True)``，不要抛异常。"""
        ...

    def to_schema(self) -> ToolSchema:
        """从 params_model 自动生成 ToolSchema。"""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.params_model.model_json_schema(),
        )

    def parse_input(self, raw_input: dict[str, Any]) -> TParams:
        """把 LLM 给的 dict 参数解析成 params_model 实例。

        参数校验失败会抛 ValidationError，Agent Loop 应该 catch 并作为
        is_error=True 的结果喂回 LLM，让它修正。
        """
        return self.params_model.model_validate(raw_input)
