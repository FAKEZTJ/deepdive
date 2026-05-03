# agent-core 项目档案

> 本文档用于在新会话中快速恢复项目上下文。新开对话时，把整份内容贴给 AI 助手即可秒速接续。

---

## 一、个人背景

- **职业背景**：Java 后端程序员，转型 AI 应用开发方向
- **技术栈选择**：Python（AI 应用开发主流语言）
- **项目目的**：写在简历上，向面试官展示"我懂 Agent 开发"
- **时间预算**：1 周完成 MVP（包含包装、文档、博客）
- **代码协作方式**：通过 AI 编程工具（Cursor / Claude Code 等）实现，自己负责架构设计、code review

## 二、项目定位

**项目名**: `agent-core` —— 一个生产级的 Agent Runtime（运行时）

**一句话定位**: 不做某个具体 Agent，做"跑 Agent 的基础设施"。上层套个 Demo Agent（`deepdive` 研究型 agent）来证明能用。

**为什么这个定位**：
- 后端工程师转 AI，做 Runtime 比做 Agent 更能体现架构能力
- 避免和 opencode/Claude Code 这类 coding agent 红海正面竞争
- 边界清晰，1 周内可控
- 面试时可深可浅：聊架构 / 聊 LLM / 聊系统设计都能切

## 三、核心架构

```
┌──────────────────────────────────────────────┐
│  示例应用层 (Demo)                            │
│  - deepdive: 研究 Agent                       │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│  Agent Runtime (核心，简历主角)               │
│                                              │
│  ① Agent Loop Engine                         │
│     - ReAct / Plan-Execute 双模式             │
│     - checkpoint / resume                    │
│     - 步数/token/时间预算控制                  │
│                                              │
│  ② LLM Provider 抽象层                        │
│     - OpenAI / Anthropic / DeepSeek          │
│     - 统一 tool_call 格式                     │
│     - 块为中心的流式事件模型                   │
│                                              │
│  ③ Tool System (MCP 兼容)                    │
│     - 工具注册、参数校验、权限分级             │
│     - 内置工具 + MCP client                   │
│     - 子进程沙箱                              │
│                                              │
│  ④ Context / Memory                          │
│     - 滑动窗口 + LLM 摘要                     │
│     - SQLite 持久化                           │
│                                              │
│  ⑤ Observability (Java 背景的差异化优势)      │
│     - 结构化日志 + Trace ID                   │
│     - Token / 成本统计                        │
│     - CLI Trace 查看器                        │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│  接入层                                       │
│  - HTTP/SSE API (FastAPI)                    │
│  - CLI (Textual)                             │
└──────────────────────────────────────────────┘
```

## 四、关键设计决策

### 4.1 自研 Provider 抽象层（不用 LiteLLM/LangChain）

**理由**：
- LangChain 黑盒、调试难、依赖重，简历叙事弱
- LiteLLM 在流式 tool_call 抽象不彻底，仍有 provider-specific chunk 格式泄漏
- 自研抽象层 1 天工作量，换来完全可控的统一类型 + 完全 provider-agnostic 的上层

### 4.2 流式事件采用"块为中心"的对称结构

每个 block 有 `*Start` / `*Delta` / `*End` 三段式事件，带 `index` 字段。

事件类型：
- `TextStart(index)` / `TextDelta(index, text)` / `TextEnd(index)`
- `ToolUseStart(index, id, name)` / `ToolUseDelta(index, partial_json)` / `ToolUseEnd(index)`
- `StreamEnd(finish_reason, usage)`

**理由**：
- Anthropic 原生就是 block-oriented 模型，直通成本低
- OpenAI 在 provider 内部用状态机合成块边界，把抽象成本封装在 provider 内
- DeepSeek/Qwen 走 OpenAI 兼容接口，零成本继承
- Gemini 的 partialArgs/JSONPath 在 provider 内统一转成 partial_json 字符串
- 未来扩展 thinking / image / citation block 时事件流结构不变

### 4.3 各 provider 适配复杂度

```
零成本（继承 OpenAI）:  DeepSeek, Qwen, vLLM/Ollama OpenAI 兼容模式
低成本（直通）:        Anthropic（原生 block-oriented）
中成本（状态机合成）:   OpenAI（合成 *Start/*End，维护 index 映射）
高成本（格式转换）:     Gemini（partialArgs/JSONPath → partial_json）
```

**一周内只做 OpenAI + Anthropic + DeepSeek（复用 OpenAI）。Gemini 写进 Roadmap。**

## 五、7 天任务汇总

### Day 1 — Provider 抽象 + 数据模型 ✅ 进行中
- 项目骨架（`agent_core/`, `providers/`, `tools/`, `runtime/`, `cli/`, `tests/`）
- `agent_core/types.py`：`Message` / `ContentBlock` / `ToolSchema` / `StreamEvent` / `Usage` / 异常类
- `agent_core/providers/base.py`：`LLMProvider` 抽象 + `ProviderConfig`
- `OpenAIProvider`（含状态机合成块边界）
- `AnthropicProvider`（直通 index）
- 验收：双 provider 跑通 `test_basic_chat` + `test_tool_call` + `test_streaming`
- 产出：ADR-001（自研 Provider 抽象） + ADR-002（块为中心的流式事件模型）

### Day 2 — Agent Loop 引擎
- `AgentLoop.run(task) -> Result`：循环 LLM → 解析 tool_call → 执行 tool → 反馈
- 预算控制：max_steps / max_tokens / timeout
- `Tool` 接口（参数 schema 用 Pydantic 自动转 JSON Schema）
- 内置工具：`shell_exec`、`read_file`
- 验收：能完成 "统计当前目录有多少个 Python 文件" 这类任务
- 产出：ADR-003（为什么不用 LangGraph）

### Day 3 — Tool System + MCP Client
- `ToolRegistry`：注册、查找、参数校验
- 权限分级：`read_only` / `write` / `dangerous`
- 增加工具：`web_search`（Tavily/DuckDuckGo）、`write_file`、`http_get`
- MCP client（**只做 client 不做 server**），能调用一个公开 MCP server
- shell_exec 加超时 + 输出截断

### Day 4 — Context 管理 + Session 持久化
- `ContextManager`：超阈值时自动 LLM 摘要压缩
- `SessionStore`：SQLite 存储 messages、tool_calls、metadata
- `Checkpoint`：每 step 保存状态，支持 `resume_session(session_id)`
- 产出：ADR-004（context 压缩策略取舍）

### Day 5 — Observability（杀手锏，重投入日）
- 结构化日志（structlog）：trace_id + span_id 全链路
- Token / 成本统计：按 provider 定价折算
- 导出 trace 到 JSON（参考 OpenTelemetry 格式）
- **CLI Trace 查看器**：`agent-core trace <session_id>` 树形打印
- 可选：OpenTelemetry exporter

### Day 6 — Demo Agent + CLI 打磨
- `deepdive`：研究主题 → web_search → 网页综合 → 带引用的 markdown 报告
- CLI 用 Textual 做基础流式交互
- 录 2-3 分钟 demo 视频（asciinema / OBS）
- 顺手加 `DeepSeekProvider`（继承 OpenAIProvider，30 行）

### Day 7 — 文档 + 简历素材（不写代码）
- README：架构图、Why、核心特性、Quick start
- 整理 ADRs（4-5 篇）
- 技术博客 1 篇：标题候选 "实现 Agent Runtime 时我踩过的 5 个坑"
- 简历段落（中英文）
- demo 视频上传


## 七、简历亮点叙事（备忘）

每条都对应一个面试可展开 5-10 分钟的话题：

1. 手写 Agent Loop 引擎，支持 ReAct/Plan-Execute 双模式 + 基于 SQLite 的 checkpoint，可在任意 tool 调用前后中断恢复
2. 设计 LLM Provider 抽象层，统一 OpenAI/Anthropic/DeepSeek 的 function calling 格式差异（特别是流式 tool_call 的块边界合成），抽象成本封装在 provider 内部
3. 设计块为中心的流式事件模型（`*Start`/`*Delta`/`*End` + index），上层 consumer 可零状态重建 `Message.content` 数组
4. 实现 MCP 协议兼容的 Tool System，支持工具沙箱执行 + 权限分级（read-only/write/dangerous）
5. 设计基于滑动窗口 + LLM 摘要的混合 context 管理策略
6. 集成 OpenTelemetry，实现 Agent 执行 trace 的全链路追踪和 CLI 可视化
7. 基于 FastAPI + SSE 实现 client/server 解耦

## 八、关键技术取舍（面试问答储备）

**Q：为什么不用 LangChain / LangGraph？**
A：LangGraph 黑盒、调试难、依赖重，且 LangChain 在流式 tool_call 上抽象不彻底。自研抽象层 1 天工作量，换来完全可控的统一事件流，所有 provider 差异在 provider 内部消化。

**Q：怎么处理 OpenAI 和 Anthropic 流式差异？**
A：Anthropic 原生 content_block_start/delta/stop 三段式，自带 index，结构清晰。OpenAI 是扁平 delta，content 和 tool_calls 混在一起，且 tool_calls.index 是工具序号不是 block 序号。我在 OpenAI provider 内部用状态机合成块边界——维护 next_index 计数器和 openai_idx → block_idx 映射，在 delta.content 首次出现时开 text block，在新 tool_call 出现或 finish_reason 到来时关闭前一个 block。上层 Agent Loop 完全不感知这层差异。

**Q：pending_arguments 缓冲是为了什么？**
A：OpenAI 流式响应里 tool_call 的 id/name 和 arguments 增量到达顺序不保证。如果 args delta 先于 id/name 到达，我会缓冲到 pending_arguments 里，等 id+name 集齐后发 ToolUseStart 再补发缓冲的 deltas。这样上层 consumer 看到的事件流永远是合法序列。

**Q：为什么不一次接 5 家 provider？**
A：DeepSeek/Qwen 走 OpenAI 兼容接口，继承 OpenAIProvider 30 行代码就能加。Gemini 的 partialArgs + JSONPath 增量是独立的复杂度，需要专门的转换逻辑，做进 Roadmap 而不是塞进 MVP。这体现了边际成本递减的抽象设计。

## 九、风险点

- **Day 1 流式 tool_call 是关键路径**：拖延会影响后面所有模块
- **不要在 CLI 美化上浪费时间**：基础流式输出即可
- **demo agent 不要换主题**：选定 deepdive 不再变
- **AI 生成的代码必须 review**：尤其是流式部分，AI 一次写对的概率 < 50%

## 十、当前对话已涉及的代码资产

- `types.py`：完整接口定义（已设计）
- `providers/base.py`：抽象基类（已设计）
- `providers/openai_provider.py`：核心实现（已迭代 2 轮，主要 bug 已修，剩小调整）

新会话恢复时，可以请求 AI 助手：
1. 重新生成最新版 `types.py`（按 § 4.2 的事件模型）
2. 重新生成 `OpenAIProvider`（按 § 六的"已完成"细节）
3. 直接进入 `AnthropicProvider` 或 Day 2 任务