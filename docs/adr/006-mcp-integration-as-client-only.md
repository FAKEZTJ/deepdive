# ADR-006: MCP 集成策略（仅客户端）

## 状态
已接受

## 背景
MCP（Model Context Protocol）提供了一套标准化工具协议，生态里已经存在大量可复用的 MCP server。

从集成方式上看，有三种路线：

1. 只做 client：连接外部 MCP server，并使用它们暴露的工具
2. 只做 server：把我们自己的工具暴露成 MCP server
3. 同时做 client 和 server

## 决策
v1 只实现 MCP client。

## 理由
- 从项目叙事上看，“这个 agent 可以接入任意 MCP server”比“这个项目也提供一个 MCP server”更有价值。
- 现有生态已经有 filesystem、GitHub、Slack、Puppeteer 等成熟 server，可以直接复用。
- 在一周项目周期内，再实现 MCP server 会明显扩 scope，并偏离 runtime 主线。

## 实现说明
- 使用官方 Python `mcp` SDK。
- 每个 MCP server 通过 stdio 以子进程方式启动，并用 `AsyncExitStack` 管理生命周期。
- 通过 `MCPTool` 适配器，把 MCP 工具包装成统一的本地 `Tool` 接口。
- 工具命名空间采用 `{server_name}__{tool_name}`，避免多个 server 之间的命名冲突。
- 每个 server 共享一个权限级别，由 `MCPServerConfig.permission` 统一配置。
- server 提供的 `inputSchema` 直接透传给模型，不在本地再生成一套 Pydantic schema。
- 参数校验由 MCP server 自己负责，客户端不重复做一遍校验。

## 影响
- 项目增加了 `mcp` Python SDK 依赖。
- MCP server 崩溃后的重连机制留到 v2 处理。
- MCP 工具的参数校验在服务端完成，因此本地拿到的错误信息不如内置工具那样结构化。
