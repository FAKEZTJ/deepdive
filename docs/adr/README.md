# ADR Index

`agent-core` 当前最值得优先阅读的 ADR 有 5 篇，它们基本覆盖了这个 runtime 的主干设计。

## 推荐阅读顺序

1. [ADR-001：为什么需要自定义 Provider 抽象](./001-why-custom-provider-abstraction.md)
2. [ADR-005：工具权限模型与并发调度](./005-tool-permission-model-and-concurrent-dispatch.md)
3. [ADR-007：Context 压缩策略](./007-context-compression-strategy.md)
4. [ADR-008：Session 持久化设计](./008-session-persistence-design.md)
5. [ADR-010：可观测性架构](./010-observability-architecture.md)

## 这些 ADR 分别回答什么问题

- `ADR-001`
回答“为什么不把某一家厂商 SDK 直接当核心接口”。

- `ADR-005`
回答“工具调用如何做权限控制、并发调度和默认安全边界”。

- `ADR-007`
回答“长任务如何压缩上下文，同时不破坏 tool_use / tool_result 结构”。

- `ADR-008`
回答“为什么恢复状态和运行事件要分表，以及为什么只支持 step 边界恢复”。

- `ADR-010`
回答“SQLite 事件、结构化日志、OpenTelemetry span 和成本统计之间如何分工”。

## 其余 ADR

- [ADR-002：为什么不在当前阶段引入 LangGraph](./002-why-not-langgraph.md)
- [ADR-006：MCP 以 client-only 方式集成](./006-mcp-integration-as-client-only.md)
- [ADR-011：demo agent 与 streaming 集成](./011-demo-agent-and-streaming-integration.md)

这些文档更偏阶段性取舍或 demo 层决策，可以在读完主干 5 篇后再看。
