# 简历素材（中文）

以下内容基于当前 `agent-core` 项目整理，尽量使用可被代码和文档支撑的表述，不使用虚构指标。

## 版本一：一段式项目描述

独立设计并实现 `agent-core`，构建面向多 Provider 的 Agent Runtime 内核。项目围绕 OpenAI / Anthropic / DeepSeek 的统一适配，抽象了 Provider 中立的数据模型与流式事件协议，落地了支持工具权限控制与并发调度的 `AgentLoop`，并实现基于 SQLite + WAL 的 session 持久化与 step 边界恢复机制。同时补齐上下文压缩、OpenTelemetry tracing、成本统计、CLI trace 导出以及基于 MCP 的工具接入能力，使该项目具备从 demo 走向可恢复、可观测、可扩展 runtime 的工程基础。

## 版本二：适合简历项目经历的 3 条要点

- 设计多 Provider Agent Runtime 架构，统一 OpenAI、Anthropic、DeepSeek 的消息模型、工具调用协议与流式事件，降低上层业务对厂商 SDK 的直接耦合。
- 实现 `AgentLoop`、`ToolDispatcher` 与 `ContextManager`，支持预算控制、工具权限白名单、只读并发/写操作串行调度，以及长任务上下文压缩。
- 基于 SQLite + WAL 完成 session 持久化与 step 级恢复，结合 OpenTelemetry、结构化日志与 CLI trace 导出，建立本地优先的可观测性闭环。

## 版本三：更偏“高级工程师 / 架构设计”表述

主导一个本地优先的 Agent Runtime 核心层建设，重点解决多模型厂商协议不统一、工具调用缺乏执行边界、长上下文成本失控、任务中断后不可恢复，以及 Agent 运行过程难以调试等工程问题。通过分层设计将 Provider 适配、运行时控制流、上下文管理、会话持久化、可观测性和 MCP 工具接入拆分为独立模块，并通过 ADR 记录关键架构取舍，形成具备扩展性和可维护性的 Agent 基础设施原型。

## 可单独摘取的关键词

- Agent Runtime
- Multi-provider abstraction
- ReAct loop
- Tool permission model
- Concurrent dispatch
- Context compression
- Session persistence
- SQLite + WAL
- OpenTelemetry
- MCP integration
- Local-first observability

## 面试时可展开的 4 个亮点

- 为什么要自定义 Provider 抽象，而不是直接绑定某家 SDK。
- 为什么工具执行不能只做 `asyncio.gather(...)`，而要区分权限和调度策略。
- 为什么 session 恢复不能简单等同于“消息落盘”。
- 为什么本地 SQLite 事件日志和 OTel span 需要同时存在。
