# Bytes Architecture Next (Memory + Guardrails MVP)

## Goal
Upgrade the hybrid RAG + agentic assistant into a more production-style GenAI system by adding:
- opt-in session memory,
- explicit guardrail checkpoints,
- richer evaluation metrics for interview-grade evidence.

## Baseline vs Upgraded
### Baseline
- RAG path from local FAISS corpus.
- Agentic path with intent classification, strategy selection, live read-only tools, synthesis, citations.
- Deterministic scenario evaluator.

### Upgraded
- New memory subsystem (`memory_store.py`) with FAISS + JSONL sidecar.
- New guardrail subsystem (`guardrails.py`) for:
  - input prompt-injection detection,
  - output citation gating for freshness-sensitive answers,
  - memory text sanitization before persistence.
- Runtime graph adds new nodes for memory and guardrails.
- UI exposes memory controls (`Enable Memory`, `Session ID`, `Clear Memory`).
- Evaluation includes memory and guardrail metrics.

## Runtime Flow
1. `classify_intent`
2. `input_guardrail`
3. `retrieve_memory` (only if memory enabled)
4. `decide_strategy`
5. `retrieve_or_tool_call`
6. `synthesize_answer`
7. `cite_sources`
8. `output_guardrail`
9. `persist_memory` (only if memory enabled)

## Memory Design
- Storage:
  - vector index: `memory_data/index/`
  - metadata sidecar: `memory_data/memory_records.jsonl`
- Record fields:
  - `memory_id`, `session_id`, `text`, `tags`, `created_at`, `expires_at`, `source`, `pii_redacted`
- TTL:
  - default 30 days
- Session isolation:
  - retrieval and clear operations are scoped by `session_id`
- Dedupe:
  - normalized text match prevents duplicate writes in active session memory

## Guardrail Design
### Input guardrail
- Detects known injection patterns such as "ignore previous instructions" and "reveal system prompt".
- High-severity matches force `reject_guardrail` strategy.

### Output guardrail
- If query/intent is freshness-sensitive, citations are required.
- Missing citations triggers safe constrained response.

### Memory safety filter
- Sanitizes memory candidate text.
- Redacts sensitive patterns (token/key/password/email-like content).
- Blocks unsafe instruction-like memory payloads.

## Public Interface Changes
- `run_agent(query, enabled_sources, model_name, session_id="default", memory_enabled=False, memory_top_k=3)`
- Tool registry adds:
  - `search_memory` (read)
  - `write_memory` (post-synthesis write path)

## Evaluation Additions
- New CLI args:
  - `--memory on|off`
  - `--session-id <id>`
- New metrics:
  - `memory_hit`
  - `guardrail_triggered`
  - `citation_gate_passed`
  - `personalization_correctness`

## Tradeoffs
- Local-first memory is easier to demo and deterministic but single-instance only.
- Hash-embedding fallback avoids hard dependency on API availability but is lower semantic quality than hosted embeddings.
- Heuristic guardrails are fast to implement and explain, but should be replaced by policy-as-code + stronger classifiers in production.
