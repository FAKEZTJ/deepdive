# agent-core

一个面向 Agent / LLM Runtime 的最小核心库，目标是提供统一的数据模型、Provider 适配层，以及可扩展的工具调用与流式事件抽象。

当前实现重点放在两件事上：

- 统一 OpenAI 与 Anthropic 的请求/响应模型
- 将不同厂商的流式输出归一化为一套稳定的事件协议

## 项目目标

本项目希望解决多 Provider 场景下常见的几个问题：

- 上层业务代码直接依赖厂商 SDK，切换模型供应商成本高
- 不同厂商的消息格式、工具调用格式、流式协议差异明显
- 工具调用、错误处理、事件消费逻辑容易散落在业务层

因此，`agent-core` 的设计目标是：

- 对上提供统一接口，不暴露厂商 SDK 细节
- 对下通过 Provider Adapter 接入不同模型厂商
- 在运行时层面统一消息、工具调用、流式事件、异常语义

## 当前能力

当前版本已经具备以下基础能力：

- 统一消息模型：`Message`、`TextContent`、`ToolUseContent`、`ToolResultContent`
- 统一 Provider 抽象：`LLMProvider`
- OpenAI Provider 适配
- Anthropic Provider 适配
- 统一流式事件模型
- 统一异常映射

当前版本仍然属于早期实现，重点在抽象边界和协议稳定性，而不是功能完备性。

## 快速开始

### 1. 安装依赖

```powershell
uv sync
```

### 2. 运行测试

```powershell
uv run pytest
```

## 核心概念

### 1. 统一消息模型

项目内部不直接使用 OpenAI 或 Anthropic 原生消息格式，而是统一使用 `agent_core.types` 中定义的消息模型。

主要类型包括：

- `Message`
- `TextContent`
- `ToolUseContent`
- `ToolResultContent`
- `CompletionResponse`

这意味着：

- 上层代码只处理统一的消息结构
- Provider 负责把统一结构转换成厂商 API 所需格式
- 厂商响应也会被转换回统一结构

### 2. Provider 抽象

`LLMProvider` 是所有模型供应商的统一接口。当前约定的核心能力包括：

- `chat(...)`：非流式调用
- `chat_stream(...)`：流式调用

这样做的目的是让运行时、工具执行层、上下文管理层不依赖某个特定厂商 SDK。

### 3. Tool Call 转换

工具调用在内部使用统一结构表达：

- `ToolUseContent`：模型请求调用工具
- `ToolResultContent`：工具执行结果回传模型

不同厂商的差异由 Provider 适配层处理：

- OpenAI 使用 `tool_calls` / `role="tool"` 消息
- Anthropic 使用 `tool_use` / `tool_result` content blocks

上层逻辑不需要关心这些差异。

### 4. 流式事件模型

不同 Provider 的流式协议差异很大，因此项目内部定义了一套统一事件：

- `TextStart`
- `TextDelta`
- `TextEnd`
- `ToolUseStart`
- `ToolUseDelta`
- `ToolUseEnd`
- `StreamEnd`

消费者只需要处理这套事件，而不需要直接解析厂商原始流。

## Stream Event 的 Index 语义

`StreamEvent` 在文本和工具调用事件上都带有 `index`，用于标识该事件属于哪个逻辑内容块。

这一字段在不同 Provider 下的语义并不完全相同。

### Anthropic

Anthropic 原生流协议本身就有明确的 content block index，因此适配层会直接保留该 index。

也就是说：

- `index` 直接对应厂商原始 block 顺序
- 文本块和工具块都沿用 Anthropic 的原生编号

### OpenAI

OpenAI Chat Completions 流式输出中，文本来自 `delta.content`，工具调用来自 `delta.tool_calls[index]`。它没有提供一套统一的“文本块 + 工具块”联合索引。

因此，`OpenAIProvider` 会在适配层自行分配 `index`：

- 按首次观察到的逻辑块顺序分配
- 第一个块为 `index=0`
- 后续每个新块依次递增
- 文本块和工具块共享同一套编号空间

例如：

```text
如果先输出文本，再输出工具：
  text      -> index=0
  tool #1   -> index=1

如果直接输出工具，没有文本：
  tool #1   -> index=0
```

因此对 OpenAI 来说，`index` 是适配层定义的稳定排序键，而不是厂商原生字段。

### 消费端建议

建议消费端遵循以下原则：

- 使用 `(事件类型族, index)` 聚合同一逻辑块
- 不要假设 OpenAI 与 Anthropic 的 `index` 可以跨 Provider 比较
- 如果需要工具调用的厂商原生身份，使用 `ToolUseStart.id`

## 目录结构

```text
agent-core/
├── pyproject.toml
├── README.md
├── docs/
│   └── adr/
│       └── 001-why-custom-provider-abstraction.md
├── agent_core/
│   ├── __init__.py
│   ├── types.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   ├── tools/
│   ├── runtime/
│   ├── context/
│   ├── observability/
│   └── cli/
└── tests/
    └── providers/
        ├── test_openai.py
        └── test_anthropic.py
```

## 开发约定

当前项目采用以下工程约定：

- 内部优先使用统一类型，不直接在业务层暴露厂商 SDK 类型
- Provider 层负责协议转换与异常映射
- 新增 Provider 时，优先对齐已有 `chat` / `chat_stream` 行为契约
- 流式行为必须通过测试覆盖，尤其是工具调用与结束事件顺序

## 已完成内容

- 统一消息与工具调用数据模型
- OpenAI Provider 基础实现
- Anthropic Provider 基础实现
- OpenAI / Anthropic 的基础测试
- 流式事件协议的统一

## 后续计划

建议后续按以下顺序推进：

1. Provider 工厂或注册表
2. Runtime 层对接统一 Provider
3. Tool 执行器与工具注册机制
4. Context 管理
5. Observability
6. CLI

## ADR

关于为什么要引入自定义 Provider 抽象，请参见：

- [ADR-001：为什么需要自定义 Provider 抽象](./docs/adr/001-why-custom-provider-abstraction.md)
