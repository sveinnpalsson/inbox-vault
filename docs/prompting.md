# Local LLM Prompting Strategy

This project uses centralized prompt constructors in `src/inbox_vault/prompts.py` for:

- enrichment/classification (`build_enrichment_messages`)
- contact profile generation (`build_profile_messages`)
- model redaction (`build_redaction_messages`)

## Design principles

1. **Explicit role separation**
   - System messages define role + privacy/safety constraints.
   - User messages carry task-specific input and output contract.

2. **Strict output contracts**
   - Enrichment/profile prompts embed deterministic JSON schema contracts.
   - Prompts include explicit formatting rules (single JSON object, no markdown/code fences).
   - Parsing is resilient via `extract_first_json` (scans for first valid JSON object).

3. **Privacy and utility guidance**
   - Prompts avoid requesting chain-of-thought.
   - Instructions emphasize: use provided content only, no external inference, preserve triage utility.

4. **Deterministic local-model cues**
   - `chat_json(..., temperature=0.0)` for enrichment/profile tasks.
   - Enrichment prompts include `Control: /no_think`; profile prompts rely on strict JSON/output-contract rules without forcing `/no_think`.
   - Stable formatting instructions to reduce output variance.

5. **Chunk-aware handling**
   - Redaction prompts include `chunk_index/chunk_total` and a rule to redact each chunk independently.
   - Enrichment explicitly labels when body text is truncated (`Body (truncated, max_chars=...)`).

## Tuning knobs

- `body_max_chars` (enrichment prompt constructor call site)
- `max_samples` + `sample_chars` (profile constructor)
- `[redaction].chunk_chars` in config (chunk size for model redaction)
- `[llm].timeout_seconds` and model selection in config

## Compatibility notes

- Existing fallback behavior remains:
  - enrichment/profile continue on model errors
  - redaction falls back to regex when model mode fails
  - hybrid mode still applies regex pass after model redaction
- Chat completion extraction now tolerates OpenAI-compatible variants where:
  - `message.content` is a string **or** segmented array
  - `message.content` is empty but `message.reasoning_content` contains the model text
- `chat_json` requests JSON mode (`response_format={"type":"json_object"}`) when available, then retries once with a stricter reminder if first parse fails.
- JSON extraction still handles malformed-prefix + nested-object cases via `extract_first_json`.

## Current prompt flow (summary + redaction)

1. **Ingest first, prompts later**: Gmail sync writes normalized rows to `messages` before any LLM call.
2. **Summary/classification (`enrich`)**:
   - `enrich_pending` selects rows missing `message_enrichment`.
   - `build_enrichment_messages(...)` assembles system+user prompts with strict JSON schema contract.
   - `chat_json(...)` extracts first valid JSON object; only then does `upsert_enrichment(...)` write a row.
   - If model output is empty/non-JSON/error, that message is skipped (no enrichment row).
   - Diagnostics counters are tracked in both enrichment/profile flows: `attempted`, `succeeded`, `http_failed`, `parse_failed`.
3. **Redaction (`index-vectors`)**:
   - Source text is composed from `subject/snippet/body`.
   - `redact_text(...)` runs per selected mode (`regex`, `model`, `hybrid`) and stores both raw and redacted text.
   - Chunk vectors repeat the same redaction flow per chunk (`chunk_index/chunk_total` passed to prompt).

## Practical improvement areas

- **Redaction consistency checks**: optionally score redacted-vs-raw leakage and log warning when high-risk tokens survive.
- **Configurable truncation by model**: expose `body_max_chars`/`sample_chars` in config for easier tuning by context window.
