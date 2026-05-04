# ADR-005: 工具权限模型与并发调度

## 状态
已接受

## 背景
LLM 一次响应可能返回多个 tool call。最直接的实现方式是用 `asyncio.gather` 全并发执行，但这会带来三个问题：

1. 写操作并发执行时可能发生竞争，例如两个工具同时写同一个文件。
2. `dangerous` 工具比普通只读工具需要更严格的访问控制。
3. 如果没有并发上限，模型一旦失控返回大量 tool call，可能直接打爆宿主机。

## 决策
1. 定义三档权限：`read_only`、`write`、`dangerous`，作为 Tool 契约的一部分。
2. `AgentLoop` 在启动时声明 `allowed_permissions` 白名单。未进入白名单的工具既不会暴露在 schema 中，也不会在执行阶段被允许运行。
3. 采用以下调度策略：
   - `read_only` 工具并发执行
   - `write` 和 `dangerous` 工具串行执行，并按 `tool_uses` 中出现的顺序运行
4. 通过 `max_concurrent_tools` 和 semaphore 实现全局并发上限。
5. 默认白名单为 `{read_only, write}`，`dangerous` 必须显式开启。

## 理由
- 将 `write` 串行化可以避免文件或外部状态竞争。
- 将 `dangerous` 串行化并默认禁用，是更安全的默认值。
- 对只读工具保留并发，仍然可以获得大部分性能收益。
- 在 schema 层过滤工具，比只在执行时拒绝更高效：既能节省 token，也能减少模型对不该用工具的无效尝试。

## 影响
- 引入了专门的 `ToolDispatcher` 抽象，将调度逻辑从 `AgentLoop` 中拆出。
- 多个 `write` 工具的总耗时会变为串行累计时间。
- 未来如果要支持显式工具依赖，例如“先写后读”，可以在 dispatcher 层扩展成依赖图，而不需要改动 loop 的契约。
