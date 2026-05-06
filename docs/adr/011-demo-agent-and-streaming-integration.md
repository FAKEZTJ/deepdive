# ADR-011：Demo Agent 与流式集成
## 状态
已采纳

## 背景

Day 6 需要在不改变 Runtime 核心执行模型的前提下，引入一个面向演示的研究型 agent、CLI 实时渲染，以及第三个 provider。

当时的主要设计问题有三个：

1. Demo agent 是否应该通过自定义 Loop 扩展 Runtime？
2. LLM 流式输出是否应该成为持久化 `RunEvent` 历史的一部分？
3. 需要引用感知的应用，应该如何从工具执行中恢复 citation 所需的上下文？

## 决策

我们采用以下规则：

1. `deepdive` 作为应用层 agent，放在 `agent_core.apps.deepdive`。
2. `AgentLoop` 仍然是唯一的 Runtime Loop。
3. 流式能力集成在 `AgentLoop` 内部，通过 `stream_llm=True` 开启。
4. 流式增量事件属于临时观察者回调，不进入持久化 `RunEvent`。
5. `ToolCallCompleted` 携带结构化 `metadata`，让应用层 agent 能在不解析 provider 特定 payload 的情况下恢复 source 上下文。
6. DeepSeek 这类 OpenAI-compatible provider 通过继承 `OpenAIProvider` 接入，而不是另起一套并行 provider 栈。

## 理由

### 1. Demo 应该证明 Runtime 是通用的

如果 `deepdive` 需要一个自定义 loop，demo 传达出来的信息就会变成“这个 Runtime 只适合为特定 agent 手工定制”。把它保持为 `prompt + tools + runner` 的组合，更能证明 Runtime 是可复用的。

### 2. 持久化事件与流式增量承担的是不同职责

持久化事件用于 replay、resume 和 trace inspection。流式增量用于实时用户体验。

如果把它们混在一起，会带来以下问题：

- 扩大持久化 schema
- 增加 resume 语义复杂度
- 把 CLI 渲染层的关注点塞进 durable execution log

因此，`RunEvent` 继续保持业务执行视角，而流式增量只通过 callback 暴露。

### 3. 引用感知应用需要工具元数据

研究类应用关心 canonical URL、搜索结果列表，以及实际抓取到的资源身份。这些信息本来就存在于工具结果里，因此把它们透传为 `ToolCallCompleted.metadata`，是最小且足够有用的扩展。

这样做能让 source tracking 保持为应用层能力，同时在已有结构化数据可用的地方避免脆弱的字符串解析。

## 影响

正面影响：

- 可以在不改 Runtime 控制流的前提下构建 demo agent
- CLI 实时渲染可以直接复用 provider adapter 已有的 streaming 工作
- 需要 source 感知的应用可以更干净地观察工具执行
- 新增 OpenAI-compatible provider 的成本保持很低

负面影响：

- Runtime 现在有两个输出通道：durable events 和 transient stream callbacks
- CLI 渲染逻辑需要同时组合这两个通道
- 由于应用层逻辑依赖 `metadata`，因此对元数据质量的要求更高

## 实现说明

本次采纳的实现具有以下特征：

1. `AgentLoop(stream_llm=True)` 会消费 `provider.chat_stream()`。
2. `StreamReconstructor` 会重建出最终 assistant `Message`，供后续 loop 继续使用。
3. CLI 渲染同时监听 `RunEvent` 和 `StreamEvent`。
4. `deepdive` 通过观察工具完成事件来追踪已咨询过的 sources。
5. DeepSeek 支持通过一个很薄的 `OpenAIProvider` 子类实现，只是默认 `base_url` 不同。
