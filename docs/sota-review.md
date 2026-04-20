# Retrieval and privacy review

Date: 2026-03-15

This document is a short operator-facing review of the current design.

## Current strengths

- Local-first operation with SQLCipher at-rest encryption
- Sync and processing stages are separated (`backfill/update/repair`, `enrich`, `build-profiles`, `index-vectors`)
- Incremental processing is idempotent for normal reruns
- Retrieval supports `dense`, `lexical`, and `hybrid`
- Date-window filters are available on retrieval commands

## Current limits

- Large mailboxes can still be slow when indexing/searching many rows
- Redaction remains best-effort and should be validated on real data samples
- Retrieval quality depends heavily on local embedding model quality and endpoint stability

## Practical next steps

1. Keep indexing scope tight (INBOX/SENT; exclude noise labels)
2. Tune `max_index_chars` and batch sizes for stable local runs
3. Maintain small local qrels files and run `eval-retrieval` before major config/model changes
4. Periodically review redaction outputs for missed sensitive tokens

## Quick validation checklist

- `inbox-vault --help` works
- `inbox-vault status --json` works with local config
- `pytest` passes (or documented known failures)
- `.gitignore` still protects local configs, data, logs, and artifacts
