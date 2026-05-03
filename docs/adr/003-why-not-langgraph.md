# ADR-003: Why Day 2 Does Not Use LangGraph

## Status

Accepted

## Context

Day 2 introduces the agent loop: call the LLM, inspect tool calls, execute tools, feed results back, and stop under explicit budget constraints.

At this stage the runtime has three hard requirements:

1. The internal event contract must be explicit and small.
2. Error paths must still emit `RunCompleted` exactly once.
3. Provider-specific details must stay below the runtime layer.

The project goal is not to assemble an agent application quickly. It is to build the runtime layer itself and make its control flow observable and testable.

## Decision

Day 2 keeps a handwritten `AgentLoop` instead of adopting LangGraph.

The loop remains a direct state machine in project code:

- `step_started`
- `llm_call_started`
- `llm_call_completed`
- `tool_call_started`
- `tool_call_completed`
- `step_completed`
- `run_completed`

Budgets (`max_steps`, `max_tokens`, `timeout`) are enforced by the runtime, and tool failures are converted into tool-result messages instead of leaking provider or framework exceptions into callers.

## Rationale

### 1. The loop semantics are the product

For this project, the loop is not plumbing around the product. The loop is the product. A graph framework would hide the exact control-flow edges that Day 2 is meant to demonstrate.

### 2. Event correctness matters more than convenience

The runtime distinguishes stream events from run events and guarantees that `RunCompleted` is emitted even on failure paths. That guarantee is easier to reason about in a small handwritten loop than inside a generic orchestration framework.

### 3. Provider normalization already adds one abstraction layer

The codebase already normalizes OpenAI and Anthropic behavior behind `LLMProvider`. Adding LangGraph on top would introduce another abstraction boundary before the runtime semantics have stabilized.

### 4. Day 2 scope is small enough to keep custom

The current loop handles:

- sequential LLM turns
- parallel tool execution
- structured tool-result feedback
- explicit stop conditions

That scope is still small, so custom code is cheaper than framework integration.

## Alternatives Considered

### Use LangGraph now

Pros:

- faster initial scaffolding
- built-in graph vocabulary
- easier future branching workflows

Cons:

- runtime semantics become framework semantics
- event guarantees become harder to verify precisely
- debugging boundary shifts from project code into library internals
- weakens the architectural story of "we built the runtime"

Rejected for Day 2.

### Keep only provider adapters and delay the runtime

Pros:

- smaller immediate implementation scope

Cons:

- does not satisfy Day 2 goals
- leaves tool execution and budget handling undefined

Rejected.

## Consequences

Positive:

- the runtime contract stays explicit
- tests can assert exact event sequences
- budget bugs and error-path bugs are easier to diagnose
- the project retains a strong "built the core runtime" narrative

Negative:

- more code to maintain by hand
- future branching workflows may require refactoring into a more general state machine

## Follow-up

If later phases need richer branching, resumable nodes, or human-in-the-loop checkpoints, revisit whether a graph abstraction is still warranted after the runtime semantics are stable.
