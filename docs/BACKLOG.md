# Backlog

## Current priorities

- Add optional weighted hybrid knobs (dense/lexical channel weighting) while preserving deterministic ordering.

## Up next

- Add offline prompt-eval fixtures (gold JSON outputs + adversarial malformed outputs) and regression scoring for enrichment/profile contracts.
- Evaluate redaction leakage/over-redaction tradeoff with labeled test set across regex/model/hybrid modes.
- Add optional chunk-level lexical index for long-message phrase recall beyond message-level FTS.
- Consider storing chunk-level provenance in search output when `--clearance full` is requested.
