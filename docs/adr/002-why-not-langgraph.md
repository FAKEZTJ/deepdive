# ADR-002：为什么 Day 2 不使用 LangGraph

## 状态

已接受

## 背景

Day 2 的目标是把 Agent Runtime 的主循环真正跑起来：调用 LLM、识别 `tool_use`、执行工具、把工具结果反馈给模型，并在预算约束下正确结束。

这一阶段对运行时有三个硬要求：

1. 内部事件契约必须小而明确。
2. 所有错误路径都必须保证发出且只发出一次 `RunCompleted`。
3. Provider 的差异必须被限制在运行时之下，不能泄漏到 Loop 层。

这个项目当前要证明的不是“能否快速拼一个 Agent 应用”，而是“是否真的构建了自己的 Runtime 核心层”。

## 决策

Day 2 继续使用手写的 `AgentLoop`，不引入 LangGraph。

当前 Loop 直接以项目代码中的状态机表达运行语义：

- `step_started`
- `llm_call_started`
- `llm_call_completed`
- `tool_call_started`
- `tool_call_completed`
- `step_completed`
- `run_completed`

预算控制 `max_steps`、`max_tokens`、`timeout` 由 Runtime 自己负责；工具执行失败会被转成 `tool_result` 消息回喂模型，而不是把框架异常直接暴露给调用方。

## 理由

### 1. Loop 语义本身就是当前阶段的产物

对这个项目来说，Loop 不是包在产品外面的脚手架，Loop 本身就是 Day 2 要交付的核心能力。过早引入图式框架，会把最关键的控制流边界藏进框架内部。

### 2. 事件正确性比便利性更重要

当前 Runtime 明确区分了 provider 流式事件和 run-level 事件，并要求即使在失败路径上也要稳定发出 `RunCompleted`。这类精确保证在小型手写状态机里更容易验证，也更容易写出回归测试。

### 3. Provider 归一化已经是一层抽象

代码库已经通过 `LLMProvider` 吸收了 OpenAI 与 Anthropic 的差异。如果在 Runtime 之上再叠一层 LangGraph，会在运行语义还未稳定前引入第二层控制抽象，调试边界会明显变差。

### 4. Day 2 规模还不足以支撑框架收益

当前 Loop 处理的事情其实仍然有限：

- 顺序推进 LLM 回合
- 并发执行工具调用
- 结构化回传工具结果
- 基于预算显式停止

在这个规模下，手写代码比集成框架更便宜、更透明。

## 备选方案

### 方案 A：现在就接入 LangGraph

优点：

- 初始脚手架更快
- 自带图式表达能力
- 未来分支流程扩展可能更方便

缺点：

- Runtime 语义容易退化成框架语义
- 事件保证会更难精确验证
- 调试边界会从项目代码转移到框架内部
- 会削弱“自己构建 Runtime”的架构叙事

结论：

Day 2 不采用。

### 方案 B：只做 Provider，不做 Runtime

优点：

- 当前实现量更小

缺点：

- 无法满足 Day 2 目标
- 工具执行、预算控制、结束语义都还处于空缺状态

结论：

不采用。

## 影响

正面影响：

- Runtime 契约保持清晰可见
- 测试可以直接断言精确事件序列
- 预算边界和错误恢复问题更容易定位
- 项目能保留清晰的“手写 Runtime 核心”叙事

负面影响：

- 需要手工维护更多控制流代码
- 如果后续进入复杂分支或人工介入流程，可能需要再抽象成更通用的状态机

## 后续

如果后续阶段真的出现复杂图分支、可恢复节点或 human-in-the-loop checkpoint，再重新评估是否值得引入图式抽象；但那应该发生在 Runtime 语义稳定之后，而不是 Day 2。
