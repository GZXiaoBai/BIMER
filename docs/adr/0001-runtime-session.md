# ADR 0001: One owned deployment runtime

Status: Accepted

## Context

CLI, Gradio, and M2 acceptance previously assembled or retained a bare
`DialogueAnalyzer`. Verification, cache clearing, request context, and encoder
shutdown were spread across adapters.

## Decision

`RuntimeSession` owns the analyzer lifecycle and is the internal runtime
interface. `build_runtime_session()` verifies a deployment before loading
models. The historical `build_runtime()` function remains a compatibility
wrapper returning the underlying analyzer.

## Consequences

- Adapters share device fallback, cache, verification, and shutdown behavior.
- Text-edit reanalysis can reuse the last video and language safely within one
  session.
- Long-running adapters must call `close()` or use the context manager.
- The public `analyze_dialogue()` signature does not change.
