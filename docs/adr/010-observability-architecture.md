# ADR-010: Observability Architecture
## Status
Accepted

## Context

By Day 4, `agent-core` already had three useful primitives:

- a stable `RunEvent` protocol emitted by `AgentLoop`
- a SQLite-backed `events` table that persisted those business events
- resumable `sessions` / `messages` state with checkpoint semantics

Day 5 adds production-grade observability requirements:

1. structured logs that can be correlated across a whole run
2. OpenTelemetry spans for performance and topology analysis
3. token and cost accounting for LLM calls
4. an offline trace viewer that works without Jaeger, Tempo, or any hosted backend

The core design question is whether SQLite events and OpenTelemetry traces should be treated as the same thing.

## Decision

We treat SQLite `events` as the source of truth for what happened, and OpenTelemetry spans as a logical view over that execution.

This ADR adopts the following rules:

1. `events` remains the canonical persisted execution log.
2. OTel spans are generated at runtime for live observability, but are not required for local inspection.
3. Every persisted event captures a trace envelope:
   - `trace_id`
   - `span_id`
   - `parent_span_id`
4. CLI trace inspection and JSON export read from SQLite, not from an external OTel backend.
5. `AgentLoop` stays minimally invasive:
   - logging and tracing are introduced through scoped context managers
   - core loop control flow remains event-driven
6. token usage and cost are related but stored separately:
   - token usage is checkpoint-aligned session state
   - cost is estimated independently from provider/model pricing

## Rationale

### 1. Events and traces solve different problems

The `events` table is optimized for business execution history:

- step lifecycle
- tool start/completion
- LLM response payloads
- resume/replay compatibility

OTel spans are optimized for hierarchical execution analysis:

- nested timing
- span attributes
- backend integrations
- flame-graph style exploration

Merging these concerns into one abstraction would weaken both.

### 2. SQLite must remain sufficient for local workflows

Requiring Jaeger or Tempo just to inspect a single local session would be an unacceptable developer experience regression.

SQLite gives us:

- zero extra infrastructure
- offline trace inspection
- cross-process visibility
- deterministic testability

Therefore the local viewer and JSON exporter read directly from SQLite.

### 3. Trace envelope columns bridge runtime and persistence

Adding `trace_id` / `span_id` / `parent_span_id` to `events` creates a durable bridge between:

- runtime OTel spans
- persisted business events
- offline trace reconstruction

This lets us reconstruct tree structure later without depending on exporter output retention.

### 4. Structured logs should not own trace identity

Structured logging binds business context such as:

- `session_id`
- `step`
- `llm_call_id`
- `tool_call_id`

But `trace_id` / `span_id` are derived from the current OTel span at log-render time. This avoids multiple competing trace-context sources.

### 5. Cost accounting must not duplicate token accounting

Session token totals are updated through checkpoint semantics. Cost accumulation is separate and occurs at LLM completion time.

This separation avoids double-counting usage while still letting incomplete runs surface incurred spend.

## Consequences

Positive:

- local trace inspection works without any external telemetry backend
- live OTel integration still supports production observability
- event replay and trace reconstruction stay compatible
- structured logs, spans, and persisted events can be correlated by shared identifiers
- pricing can evolve independently of token accounting logic

Negative:

- the system now maintains multiple observability projections over the same run
- persistence schema is wider and migration logic is more involved
- exporter and CLI reconstruction logic must stay aligned with the event protocol

## Implementation Notes

The accepted Day 5 implementation has these properties:

1. `AgentLoop` creates spans for:
   - `agent_run`
   - `step`
   - `llm_call`
   - `tool_dispatch`
2. `ToolDispatcher` creates one `tool_call` span per `ToolUseContent`, including rejected and failed calls.
3. `LLMCallCompleted` carries provider/model/cost metadata for direct UI rendering.
4. `sessions.total_cost_usd` tracks accumulated estimated LLM spend.
5. `agent-core sessions` and `agent-core trace` read from SQLite.
6. `export_session_as_otel_json()` converts SQLite records into an OTel-compatible JSON structure.

## Alternatives Considered

### Alternative A: OTel backend as the only trace source

Rejected because:

- local development would require extra infrastructure
- exporter output is not guaranteed to remain locally queryable
- offline inspection would become much harder

### Alternative B: keep only events, no OTel integration

Rejected because:

- runtime hierarchy and span tooling are valuable in production
- external observability platforms expect span semantics
- performance analysis benefits from standard telemetry tooling

### Alternative C: derive traces only at read time, never at runtime

Rejected because:

- live span propagation and correlation would be unavailable
- logs could not automatically inherit active trace context
- production backends would not see real-time span data

## Conclusion

`agent-core` adopts a dual-pipeline observability design:

- SQLite events are the durable execution fact log
- OpenTelemetry spans are the runtime hierarchical observability view

This preserves local-first developer ergonomics while still enabling production-grade telemetry integration.
