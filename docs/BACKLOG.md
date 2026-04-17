# Backlog

## Current priorities

- Add optional weighted hybrid knobs (dense/lexical channel weighting) while preserving deterministic ordering.

## Up next

- Phase 1 for [stream-aware ingestion triage](ingestion-triage-stream-aware.md): add metadata-first stream fingerprinting, additive triage/reputation tables, and observe-only diagnostics before any fetch/index gating.
- Add offline prompt-eval fixtures (gold JSON outputs + adversarial malformed outputs) and regression scoring for enrichment/profile contracts.
- Validate mail-specific leakage/over-redaction tradeoff against the canonical `llm-vault` redaction contract and benchmark harness instead of creating a separate benchmark program here.
- Keep a small mail-only validation slice here for bridge/runtime regressions, but feed benchmark-worthy cases back into `llm-vault` rather than growing a second report track.
- Add optional chunk-level lexical index for long-message phrase recall beyond message-level FTS.
- Consider storing chunk-level provenance in search output when `--clearance full` is requested.
