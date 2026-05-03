"""agent_core.types

agent-core 项目的核心数据模型层。

设计原则：
1. Provider 中立：所有类型不绑定任何一家 LLM SDK 的格式，OpenAI / Anthropic /
   Gemini / DeepSeek / Qwen 等 provider 在各自的实现里负责双向转换。
2. 块为中心的流式模型：流式事件按 content block 组织（*Start / *Delta / *End），
   每个事件带 ``index`` 字段，consumer 可按事件顺序重建 ``Message.content``。
3. 严格类型：使用 Pydantic v2 + ``Literal`` discriminator，让 IDE / mypy
   能精确推断类型。
4. 可序列化：所有类型都可以 ``model_dump_json()`` 直接落盘，也能从 JSON
   反序列化（用于 session 恢复 / checkpoint）。
5. 可扩展：未来加 thinking / image / citation 等新 content block 时，
   优先扩展 union 类型，而不是改写整体事件流结构。

参见 ADR-001（Provider 抽象层）。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 1. Role 和 Content Blocks
# ============================================================

Role = Literal["system", "user", "assistant", "tool"]
"""消息角色。

- ``system``: 系统指令
- ``user``: 用户消息
- ``assistant``: 模型回复，可包含 text 和 tool_use 混合块
- ``tool``: 工具执行结果
"""

FinishReason = Literal["stop", "tool_use", "max_tokens", "error"]
"""LLM 生成结束原因，已 provider 中立化。

- ``stop``: 模型自然结束
- ``tool_use``: 模型决定调用工具
- ``max_tokens``: 达到输出上限被截断
- ``error``: 兜底错误状态，通常更推荐直接抛 ``ProviderError``
"""


class TextContent(BaseModel):
    """纯文本内容块。"""

    type: Literal["text"] = "text"
    text: str


class ToolUseContent(BaseModel):
    """assistant 发起的工具调用。

    Attributes:
        id: tool_call_id，后续 ``ToolResultContent`` 通过它引用本次调用。
        name: 工具名称，必须匹配某个已注册的 ``ToolSchema.name``。
        input: 已解析的 JSON 参数对象，不是字符串。
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    """工具执行结果，作为 ``role=tool`` 消息的 content。"""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


# 一条消息的 content 可以是多个块的混合。
ContentBlock = Annotated[
    Union[TextContent, ToolUseContent, ToolResultContent],
    Field(discriminator="type"),
]


# ============================================================
# 2. Message
# ============================================================


class Message(BaseModel):
    """统一的 Message 模型，provider 收发都使用它。

    一条消息的 ``content`` 永远是块列表，即使是纯文本也包装成
    ``[TextContent(text=...)]``。这样 assistant 才能在一条消息里同时输出
    解释文字和 ``tool_use``。
    """

    role: Role
    content: list[ContentBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content_blocks(self) -> "Message":
        """约束 role 和 content block 的合法组合。"""
        allowed_by_role = {
            "system": (TextContent,),
            "user": (TextContent,),
            "assistant": (TextContent, ToolUseContent),
            "tool": (ToolResultContent,),
        }
        allowed_types = allowed_by_role[self.role]

        for block in self.content:
            if not isinstance(block, allowed_types):
                allowed_names = ", ".join(cls.__name__ for cls in allowed_types)
                raise ValueError(
                    f"role={self.role!r} only allows content blocks of type: {allowed_names}"
                )

        return self

    # ---- 便捷构造器 ----

    @classmethod
    def system(cls, text: str) -> "Message":
        """构造系统消息。"""
        return cls(role="system", content=[TextContent(text=text)])

    @classmethod
    def user(cls, text: str) -> "Message":
        """构造用户消息。"""
        return cls(role="user", content=[TextContent(text=text)])

    @classmethod
    def assistant_text(cls, text: str) -> "Message":
        """构造纯文本 assistant 消息。"""
        return cls(role="assistant", content=[TextContent(text=text)])

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        result: str,
        *,
        is_error: bool = False,
    ) -> "Message":
        """构造单个 tool_result 消息。"""
        return cls(
            role="tool",
            content=[
                ToolResultContent(
                    tool_use_id=tool_use_id,
                    content=result,
                    is_error=is_error,
                )
            ],
        )


# ============================================================
# 3. Tool Schema（传给 LLM 的工具描述）
# ============================================================


class ToolSchema(BaseModel):
    """传给 LLM 的工具描述。

    注意这只是给 LLM 看的元信息，不是工具实现本身。
    ``parameters`` 必须是一个 ``type=object`` 的 JSON Schema。
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def validate_parameters_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("ToolSchema.parameters must be a JSON Schema object with type='object'")
        return value


# ============================================================
# 4. 非流式响应
# ============================================================


class Usage(BaseModel):
    """token 使用统计。"""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """输入与输出 token 总数。"""
        return self.input_tokens + self.output_tokens


class CompletionResponse(BaseModel):
    """``provider.chat()`` 的返回值，已 provider 中立化。"""

    message: Message
    finish_reason: FinishReason
    usage: Usage
    raw: dict[str, Any] | None = None


# ============================================================
# 5. 流式事件 —— 块为中心的对称结构
# ============================================================
#
# 不同 provider 的流式行为差异很大：
# - Anthropic: 原生 content_block_start / delta / stop 三段式事件，自带 index
# - OpenAI: 扁平 delta，content 和 tool_calls 混合，无统一块边界
#
# 我们统一抽象成下面 7 种事件。consumer 只需处理这 7 种，不需要理解 provider
# 原始流协议。
# ============================================================


class TextStart(BaseModel):
    """文本块开始。"""

    type: Literal["text_start"] = "text_start"
    index: int


class TextDelta(BaseModel):
    """文本块增量。多个 ``TextDelta`` 顺序拼接得到完整文本。"""

    type: Literal["text_delta"] = "text_delta"
    index: int
    text: str


class TextEnd(BaseModel):
    """文本块结束。后续不会再有此 ``index`` 的 ``TextDelta``。"""

    type: Literal["text_end"] = "text_end"
    index: int


class ToolUseStart(BaseModel):
    """工具调用块开始，携带工具 id 和 name。"""

    type: Literal["tool_use_start"] = "tool_use_start"
    index: int
    id: str
    name: str


class ToolUseDelta(BaseModel):
    """工具参数增量。

    consumer 只需累积 ``partial_json``，在 ``ToolUseEnd`` 时整体解析。
    """

    type: Literal["tool_use_delta"] = "tool_use_delta"
    index: int
    partial_json: str


class ToolUseEnd(BaseModel):
    """工具调用块结束。此时累积后的 JSON 应该已经完整。"""

    type: Literal["tool_use_end"] = "tool_use_end"
    index: int


class StreamEnd(BaseModel):
    """流式响应结束事件。通常应是事件流的最后一个事件。"""

    type: Literal["stream_end"] = "stream_end"
    finish_reason: FinishReason
    usage: Usage
    provider_metadata: dict[str, Any] | None = None


StreamEvent = Annotated[
    Union[
        TextStart,
        TextDelta,
        TextEnd,
        ToolUseStart,
        ToolUseDelta,
        ToolUseEnd,
        StreamEnd,
    ],
    Field(discriminator="type"),
]


# ============================================================
# 6. Provider 异常 —— 统一异常体系
# ============================================================


class ProviderError(Exception):
    """所有 provider 异常的基类。"""

    def __init__(self, message: str, *, provider: str, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"retryable={self.retryable}, message={super().__str__()!r})"
        )


class RateLimitError(ProviderError):
    """触发 provider 限流。建议指数退避后重试。"""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, retryable=True)


class ContextLengthError(ProviderError):
    """超出模型上下文窗口。重试无意义，必须先压缩或截断 context。"""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, retryable=False)


class AuthError(ProviderError):
    """API key 无效或权限不足。重试无意义。"""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, retryable=False)


class ProviderTimeoutError(ProviderError):
    """请求超时。可重试。"""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, retryable=True)


# ============================================================
# 7. 公开 API（__all__）
# ============================================================

__all__ = [
    "Role",
    "TextContent",
    "ToolUseContent",
    "ToolResultContent",
    "ContentBlock",
    "Message",
    "ToolSchema",
    "FinishReason",
    "Usage",
    "CompletionResponse",
    "TextStart",
    "TextDelta",
    "TextEnd",
    "ToolUseStart",
    "ToolUseDelta",
    "ToolUseEnd",
    "StreamEnd",
    "StreamEvent",
    "ProviderError",
    "RateLimitError",
    "ContextLengthError",
    "AuthError",
    "ProviderTimeoutError",
]
