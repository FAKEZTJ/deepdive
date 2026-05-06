# ADR-010：可观测性架构

## 状态

已接受

## 背景

到当前阶段，`agent-core` 已经具备几项可复用的基础能力：

- `AgentLoop` 会产生稳定的 `RunEvent` 事件流
- `SessionStore` 会把运行事件持久化到 SQLite
- session 恢复与 step checkpoint 语义已经成立

但如果要把它当成一个真正可调试的 Agent runtime，仅有“能跑”和“能恢复”还不够，还需要回答以下问题：

1. 如何把日志串起来  
   一个 run 内会经历 step、LLM 调用、工具分发和工具执行，开发者需要能按同一条链路查看。

2. 如何看性能与拓扑  
   仅靠业务事件难以表达嵌套调用和耗时树，runtime 仍需要 span 级的层次结构。

3. 如何在本地调试  
   如果每次看 trace 都依赖 Jaeger、Tempo 或某个 hosted backend，本地开发成本会过高。

4. token 与 cost 应该如何记录  
   两者相关，但并不完全等价。如果把它们混成一个字段，恢复、计费和可视化都会出问题。

因此，项目需要一套既支持本地优先调试、又支持标准化遥测接入的 observability 方案。

## 决策

我们决定采用“双通道”的可观测性设计：

1. SQLite `events` 是“发生了什么”的持久化事实日志。
2. OpenTelemetry span 是“这些事情如何嵌套、耗时如何”的运行时视图。
3. 每条持久化事件都记录 trace envelope：
   - `trace_id`
   - `span_id`
   - `parent_span_id`
4. CLI 的 `sessions`、`trace`、`export-trace` 直接从 SQLite 读取，而不是依赖外部遥测后端。
5. token usage 与 cost 分开管理：
   - token 统计通过 session checkpoint 语义累加
   - cost 在 `LLMCallCompleted` 时按 provider / model 定价独立估算
6. logging、tracing 和 persistence 都围绕同一套 `RunEvent` 协议协作，而不是各自定义业务事件模型。

## 理由

### 1. 事件与 trace 解决的是不同问题

SQLite `events` 更适合表达业务执行历史，例如：

- 第几步开始
- LLM 返回了什么
- 调用了哪些工具
- 工具是否失败

而 OTel span 更适合表达：

- 调用树结构
- 嵌套耗时
- 标准化属性
- 对接外部观测平台

这两种数据如果强行合并成一个抽象，最终两边都会被削弱。

### 2. 本地 SQLite 必须足够完成调试闭环

`agent-core` 是一个本地优先的 runtime。要求开发者为了看一次单机 trace 先起 Jaeger 或接入 hosted backend，会严重破坏体验。

SQLite 的优势在于：

- 零额外基础设施
- 直接可查
- 离线可用
- 测试中可控

因此，本地 trace 查看必须以 SQLite 为一等数据源。

### 3. trace envelope 让“运行时 span”和“持久化事件”对齐

给 `events` 表增加 `trace_id / span_id / parent_span_id` 后，项目就获得了一座桥：

- 运行时的 active span 可以把上下文透传到落盘事件
- CLI 可以离线重建树形 trace
- 后续如果导出 OTel JSON，也不需要依赖 exporter 的原始输出保留

这让“本地事件日志”和“标准遥测语义”之间形成了稳定映射。

### 4. 结构化日志应该复用上下文，而不是自建 trace 源

日志天然适合携带：

- `session_id`
- `step`
- `llm_call_id`
- `tool_call_id`

但 `trace_id` / `span_id` 最好来自当前 span 上下文，而不是日志系统再自己维护一套。否则很容易出现两套 trace identity 不一致的问题。

### 5. token 与 cost 必须分账

token 是 provider 返回的使用量，决定的是上下文成本和预算累积；cost 是基于 provider / model 定价表的估算值，属于派生指标。

把两者拆开后：

- session 恢复可以只关心 token checkpoint
- cost 可以在 run 尚未完成时逐步累积
- 后续换定价表时不需要改动 usage 语义

## 备选方案

### 方案 A：只依赖 OTel backend，不持久化业务事件

优点：

- 架构表面上更“标准”
- span 查询能力更强

缺点：

- 本地调试依赖外部基础设施
- 对恢复和业务重放不友好
- 持久化语义被 exporter 能力绑架

结论：

不采用。

### 方案 B：只保留 SQLite 事件，不接 OTel

优点：

- 实现更简单
- 本地开发体验好

缺点：

- 缺少标准 span 语义
- 不利于生产环境接入外部观测平台
- 难以做层次化性能分析

结论：

不采用。

### 方案 C：SQLite 事件 + OTel span 双通道

优点：

- 本地调试和生产接入都兼顾
- 业务事实与性能拓扑职责清晰
- 能把 CLI、日志、trace 和成本统计关联起来

缺点：

- 需要维护多种观测投影
- schema 与导出逻辑更复杂

结论：

采用。

## 影响

正面影响：

- 本地无需外部 backend 也能看 session 和 trace
- 结构化日志、span、事件和成本统计可以通过共享标识关联
- 生产环境仍可接入标准 OTel 生态
- 为离线导出和 trace 重建提供稳定基础

负面影响：

- 需要保持 CLI、SQLite schema 和运行时 span 语义一致
- 可观测性代码不再只是“打一行日志”的级别
- 系统维护的是多种视图，而不是单一日志输出

## 实现约束

当前实现遵守以下约束：

1. `AgentLoop` 为 `agent_run`、`step`、`llm_call`、`tool_dispatch` 创建 span。
2. `ToolDispatcher` 为每次工具执行创建 `tool_call` span，包括失败和拒绝场景。
3. 每条落盘事件都携带 trace envelope。
4. `LLMCallCompleted` 事件直接携带 provider、model、usage、cost 元数据。
5. `sessions.total_cost_usd` 记录累计估算成本。
6. `agent-core trace` 与 `agent-core export-trace` 直接读取 SQLite，而不是读取外部 exporter 的产物。

## 结论

`agent-core` 接受“SQLite 事件作为持久化事实日志，OpenTelemetry span 作为运行时层次视图”的双通道可观测性设计。

这项决策的核心价值是：既保住本地优先的开发体验，也不给后续生产级观测能力设上限。
