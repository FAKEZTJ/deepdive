# ADR-008：Session 持久化设计

## 状态

已接受

## 背景

Day 4 之前，`AgentLoop` 的运行状态完全存在内存里：

- 当前 `messages`
- 当前 step 编号
- 累计 token usage
- 运行中的事件流

这种实现足以证明 loop 能跑通，但无法满足更长期的 agent 运行需求。一旦进程退出、崩溃或被人为中断，当前任务状态会完全丢失，也无法实现：

1. 从上一次稳定位置继续运行
2. 对运行过程做 trace 重放
3. 将运行事件与当前可恢复状态分开管理

此外，当前运行时存在一个关键约束：step 内部会经历 `LLMCallCompleted`、`ToolCallStarted`、`ToolCallCompleted` 等多个中间状态。如果在这些中间状态随意落盘 `messages`，恢复时就会读到半步状态，破坏“恢复后与不中断继续执行等价”的要求。

因此，项目需要一套同时满足“可恢复”“可观测”“边界清晰”的 session 持久化方案。

## 决策

我们决定引入基于 SQLite + WAL 的 session 持久化设计，并遵守以下原则：

1. 使用三张表分离不同职责的数据：
   - `sessions`：会话元数据与累计统计
   - `messages`：当前可恢复的稳定消息工作集
   - `events`：运行过程事件流
2. `events` 实时追加写入，作为 observability 与 replay 的基础日志。
3. `messages` 不在 step 中间落盘，只在稳定 step 边界通过 checkpoint 原子写入。
4. checkpoint 的提交时机是 `StepCompleted` 之后。
5. `resume(session_id)` 只恢复最近一次已完成 step 的状态，不支持恢复到 step 中间。
6. `resume()` 不仅恢复 `messages`，还恢复：
   - `step = total_steps + 1`
   - `total_usage`
7. session 最终状态与 `RunCompleted.stop_reason` 解耦映射：
   - `finished -> completed`
   - `error -> error`
   - `max_steps / max_tokens / timeout -> paused`

## 理由

### 1. `sessions`、`messages`、`events` 的职责不同，不能混存

这三类数据服务的目标不同：

- `sessions` 回答“这个会话现在是什么状态”
- `messages` 回答“恢复运行时应该喂给模型什么”
- `events` 回答“执行过程中发生了什么”

如果混在一张表里，会让恢复逻辑和观测逻辑彼此污染，难以保证边界清晰。

### 2. `messages` 必须是稳定 checkpoint，而不是实时镜像

如果 assistant message 或 tool result 在 step 中间就写入 `messages` 表，那么进程一旦中断，`resume()` 读到的就是半步状态。例如：

- assistant 已请求工具
- tool result 尚未写入

或反过来：

- tool result 已落盘
- 对应的 `tool_use` 已被压缩或尚未写入

这会让恢复后的上下文不一致，甚至直接不合法。

因此，`messages` 必须只保存已完成 step 后的稳定工作集。

### 3. `events` 适合实时落盘，因为它们本来就是过程日志

与 `messages` 不同，`events` 的职责就是记录过程。即使某个 step 没有最终完成，已经发生过的事件仍然是有价值的：

- 可用于调试
- 可用于 trace 展示
- 可用于 replay 分析

因此，事件流应实时落盘，而不应等到 step 完成后再统一写。

### 4. step 边界 checkpoint 是当前阶段最稳的恢复语义

支持“step 中间恢复”虽然更强，但代价显著更高：

- 需要恢复未完成的工具执行状态
- 需要处理中间消息与事件之间的因果关系
- 需要更多状态机复杂度

Day 4 的目标不是构建分布式工作流引擎，而是先建立可靠的恢复边界。基于 `StepCompleted` checkpoint 的恢复语义足够清晰，也足够支撑后续 observability 和 demo。

### 5. SQLite + WAL 对当前项目是合适的默认选择

当前项目是单进程、本地优先、开发迭代密集的 agent runtime。SQLite 的优势在于：

- 零外部依赖
- 便于本地调试与直接查询
- 文件级部署简单

开启 WAL 后，可以在保持实现简单的前提下获得更好的写入行为和可靠性。对于 Day 4 的规模，这是比 PostgreSQL 等外部数据库更合适的默认起点。

## 备选方案

### 方案 A：不做持久化，全部状态留在内存

优点：

- 实现最简单
- 无存储依赖

缺点：

- 无法恢复
- 无法回放
- 无法支撑长任务与进程中断场景

结论：

不采用。

### 方案 B：只存 `messages`，不存 `events`

优点：

- 可以恢复运行
- 数据模型更简单

缺点：

- 无法完整追踪运行过程
- Day 5 observability 和 trace replay 需要返工
- 很难排查 step 内部发生过什么

结论：

不采用。

### 方案 C：只存 `events`，恢复时完全靠事件重建

优点：

- 单一真相源
- 审计与 replay 能力最强

缺点：

- 恢复逻辑复杂
- 对事件完整性要求更高
- Day 4 实现成本过高

结论：

当前阶段不采用。

### 方案 D：分离 `sessions` / `messages` / `events`

优点：

- 恢复语义简单
- 观测日志独立
- 便于后续压缩、回放和状态管理

缺点：

- 存在多表写入
- 需要在 loop 中维护更严格的持久化边界

结论：

采用。

## 实现约束

当前实现遵守以下约束：

1. 新会话创建时，立即创建 `sessions` 记录，并写入首条 user message。
2. `resume(session_id, additional_input=...)` 时，如果追加新的 user message，也会立即持久化。
3. 每个 step 内新产生的 assistant/tool message 先进入内存和 `pending_checkpoint_messages`，不立即写入 `messages` 表。
4. 当 `StepCompleted` 发生后，调用 `checkpoint_step()` 原子写入：
   - 本 step 的新消息
   - `total_steps`
   - usage 增量
5. `RunCompleted` 和其他 run-level 事件实时写入 `events` 表。
6. 若 context 压缩发生，`messages` 表会通过 `replace_messages()` 与当前有效工作集保持一致。

## 影响

正面影响：

- 进程中断后可以从最近稳定 checkpoint 恢复
- 运行事件与恢复状态边界清晰
- 为 Day 5 的 trace、observability 和 replay 提供基础
- session 生命周期状态可以被结构化管理

负面影响：

- `AgentLoop` 引入了额外的持久化分支与辅助方法
- 数据一致性更多依赖 step 边界语义
- 写入次数变多，长任务下可能出现性能瓶颈

## 后续

后续可继续演进：

1. 针对长任务评估批量提交或更细粒度的性能优化
2. 评估是否需要归档旧 `events`
3. 如果未来出现真正的 step 中间恢复需求，再讨论更细粒度的 checkpoint 模型
4. 如果部署场景从本地单机扩展到多实例，再重新评估 SQLite 是否需要替换为外部数据库

## 结论

我们接受“SQLite + WAL + 三表分离 + step 边界 checkpoint”的 session 持久化设计，作为当前 `agent-core` 的默认恢复与观测基础设施。

这项决策的核心价值在于：

- 用最小可行复杂度建立可靠恢复边界
- 将可恢复状态与运行日志显式分离
- 为后续的 context 压缩、trace 重放和 observability 提供稳定基础
