# Stream-Aware Ingestion Triage

## Context

Status today:
- Inbox Vault already ships additive ingest triage tables, `status --json` summary output, and config-gated `observe` / `enforce` modes.
- The current implementation derives stream fingerprints after payload normalization and can downgrade downstream work for low-value streams, but it does not yet do metadata-first Gmail fetch avoidance.

Inbox Vault currently treats ingestion as a message-first pipeline:

1. list message ids from Gmail
2. fetch the full payload for each new id
3. normalize and store message metadata/body
4. run enrichment
5. run redaction and vector indexing in later stages

That keeps the system simple and idempotent, but it also means recurring low-value mail still consumes fetch, storage, and downstream processing budget before the system can decide whether the message was worth the effort.

This is increasingly expensive for high-volume inboxes with repetitive digests, promo mail, social updates, and list traffic. The waste is not only in embeddings. It also shows up in local LLM latency, redaction workload, larger databases, and noisier retrieval.

## Next goals

1. Add a cheap triage layer before expensive processing.
2. Make decisions at the stream level, not only the sender level.
3. Preserve safety by preferring downgrade over hard skip.
4. Use a local LLM only where it adds value.
5. Keep future rollouts additive and reversible, without breaking the existing ingest commands.

## Non-goals for the next slice

- replacing the current Gmail sync/update entry points
- changing clearance behavior or redaction contracts
- permanently deleting or hiding messages based on a first-pass guess
- requiring an LLM call for every message

## Why stream-aware, not sender-only

A sender or domain can emit both low-value and high-value mail.

Examples:
- `linkedin.com` can send a weekly digest and a security alert
- `amazon.com` can send marketing mail and an urgent account notice
- a bank can send statement reminders and fraud alerts

A sender-only reputation score would suppress the wrong things. The unit of policy should instead be a **message stream** that captures recurring mail shape.

## Proposed stream identity

Each message gets a derived `stream_id` built from cheap observable features. The stream identity should be stable enough to group recurring mail, but specific enough to separate digest traffic from account/security traffic.

Candidate inputs:
- normalized sender address
- normalized sender domain
- normalized subject family or subject template
- presence/value of bulk/list headers such as `List-Id`, `List-Unsubscribe`, `Precedence`, `Auto-Submitted`
- unsubscribe URL host
- coarse body-shape fingerprint from cheap text features
- label hints from Gmail categories when they are already available

Suggested normalization rules:
- lowercase addresses/domains
- strip obvious subject noise like counters, dates, and tracking ids where possible
- derive subject families such as `linkedin jobs digest`, `weekly roundup`, `password reset`, `invoice available`
- keep security/account-action families distinct from marketing families even for the same sender domain

The implementation should treat `stream_id` as a deterministic fingerprint, not as a user-facing concept.

## Target per-message flow

### Current flow

Today the path in `src/inbox_vault/ingest.py` is effectively:

1. list Gmail ids
2. fetch full message payload
3. convert payload with `payload_to_record(...)`
4. upsert message/raw rows
5. later run enrichment, profiles, redaction, and vectors

### Proposed future flow

For each candidate message id:

1. **List candidate ids** through the current update/backfill flow.
2. **Fetch metadata-only view first** when Gmail supports it cheaply enough for the chosen path.
3. **Compute triage inputs**:
   - sender/domain
   - subject family
   - snippet
   - bulk/list headers
   - unsubscribe host
   - prior stream reputation
4. **Run deterministic triage** to produce:
   - `stream_id`
   - signal vector
   - novelty flags
   - tentative tier
   - confidence
5. **Optionally call local LLM** only when the message is borderline, novel, or the stream has too little history.
6. **Choose an ingest tier**: `full`, `light`, `minimal`, or `suppressed`.
7. **Apply the selected path**:
   - `full`: current behavior, plus all downstream work
   - `light`: fetch/store body and metadata, but skip some expensive downstream steps
   - `minimal`: store headers, snippet, and triage diagnostics only
   - `suppressed`: record existence and stream counters only
8. **Update stream reputation** based on the observed result and any later evidence of usefulness.

The shipped rollout does not gate Gmail fetches yet. It computes the tier and diagnostics after payload normalization, then continues with normal ingest while collecting evidence or applying safe downstream downgrades.

## Deterministic signals

The triage pass should be cheap and explainable. Initial signals can include:

### Bulk and mailing-list signals
- `List-Id` present
- `List-Unsubscribe` present
- `Precedence: bulk`
- `Auto-Submitted` present
- known marketing or newsletter sender patterns

### Subject-pattern signals
- recurring subject template match
- low-information subjects such as digests, top stories, job alerts, promos
- security or account-action keywords that should force promotion
- first message in a new subject family

### Body/snippet-shape signals
- repeated boilerplate language
- high ratio of link-heavy or CTA-heavy text
- recurring footer blocks
- very small snippet diversity across recent messages in the same stream

### Stream-history signals
- total observations for this stream
- low-value ratio over time
- number of past promotions to `full`
- retrieval/usefulness counts if the message or stream was later surfaced in search/results
- days since last high-value message in the stream

## Where the local LLM fits

The local LLM should be a policy judge for uncertain cases, not a mandatory per-message stage.

Use it for:
- first few messages in a new stream
- borderline cases where deterministic signals disagree
- messages that look novel inside an otherwise low-value stream
- classification of subject family when heuristics are weak

Do not use it for:
- every repetitive digest after the stream has stabilized
- obvious bulk/list traffic with strong deterministic evidence
- operator-visible diagnostics that can be produced from rules alone

The LLM prompt should be small and structured, using metadata and short previews rather than full message bodies when possible. Any future triage client should preserve the existing chat-completions consistency choice in `src/inbox_vault/llm.py`, including `chat_template_kwargs={"enable_thinking": False}` unless there is a deliberate reason to diverge.

## Tiered outcomes

### `full`
Use the current normal path.

Includes:
- full fetch/store
- enrichment
- redaction
- vector indexing
- profile contribution

Use for:
- personal correspondence
- transactional mail
- account/security/action-needed mail
- novel or ambiguous mail

### `light`
Keep the message available with reduced downstream cost.

Likely behavior:
- fetch/store full message
- allow redaction/storage
- skip embeddings for now
- optionally skip or simplify enrichment

Use for:
- routine but occasionally useful operational mail
- receipts and confirmations
- medium-confidence low-value streams that should stay retrievable

### `minimal`
Capture that the message existed without paying the full cost.

Likely behavior:
- store message id, account, timestamps, sender, subject, snippet, stream_id, triage diagnostics
- keep full fetch behavior unchanged in the current implementation; true metadata-only persistence remains future work
- no enrichment or vectors

Use for:
- stable recurring digests
- clear promo/social streams with low retrieval value

### `suppressed`
Record the stream event only.

Likely behavior:
- store message id and enough metadata to avoid repeated reconsideration
- increment stream counters
- skip expensive processing entirely

Use only after the stream has a long and stable low-value history. This should be a later rollout stage, not a phase-1 default.

## Reputation model over time

Each `stream_id` should accumulate reputation and diagnostics.

Suggested tracked fields:
- `first_seen_at`
- `last_seen_at`
- `seen_count`
- `full_count`
- `light_count`
- `minimal_count`
- `suppressed_count`
- `promoted_count`
- `low_value_count`
- `high_value_count`
- `retrieved_count`
- `actionable_count`
- `last_llm_judged_at`
- `current_default_tier`
- `current_confidence`

Suggested threshold examples for auto-downgrade:
- at least 10 observations
- low-value ratio >= 0.8
- retrieval/usefulness ratio <= 0.05
- no recent promoted-important events

These numbers are starting points only. Phase 1 should log them before they control behavior.

## Novelty and promotion safety valves

The system should bias toward promoting messages when anything looks materially different.

Promotion triggers can include:
- subject family changes inside a known stream
- security, billing, password, verification, fraud, account, payment, renewal, or action-needed keywords
- changed sender subtype for the same domain
- first message after long inactivity
- snippet/body-shape deviation from stream baseline
- a prior suppressed stream suddenly receiving user retrieval hits

The safety principle is:

- **downgrade first**
- **suppress only after evidence**
- **promote aggressively on novelty**

## Schema additions

Phase 1 should use additive tables only.

### `message_ingest_triage`
Per-message triage diagnostics.

Suggested columns:
- `msg_id`
- `account_email`
- `stream_id`
- `tier_proposed`
- `tier_applied`
- `confidence`
- `novelty_flags_json`
- `signals_json`
- `llm_used`
- `llm_reason_json`
- `created_at`
- `updated_at`

### `message_stream_reputation`
Longer-lived stream state.

Suggested columns:
- `stream_id`
- `account_email`
- `stream_kind`
- `sender_domain`
- `subject_family`
- `first_seen_at`
- `last_seen_at`
- `seen_count`
- `low_value_count`
- `high_value_count`
- `retrieved_count`
- `promoted_count`
- `default_tier`
- `confidence`
- `signals_snapshot_json`
- `updated_at`

## Config additions

A dedicated config block is clearer than overloading `[indexing]`.

Suggested section:

```toml
[ingest_triage]
enabled = false
mode = "observe"
metadata_first = true
llm_judge_enabled = true
llm_judge_max_observations = 5
auto_min_observations = 10
auto_low_value_ratio = 0.8
auto_usefulness_ratio = 0.05
allow_minimal = false
allow_suppressed = false
promotion_keywords = ["security", "verification", "fraud", "invoice", "payment", "renewal"]
```

Current modes:
- `disabled`: current default behavior
- `observe`: compute diagnostics only, do not change ingest behavior
- `enforce`: allow safe tier-based behavior changes after payload normalization

## Rollout plan

### Shipped foundation
- stream fingerprinting and deterministic signal extraction are in place
- per-message and per-stream diagnostics are persisted
- `status --json` surfaces aggregate triage counts
- current fetch/store behavior remains unchanged

### Next phase, metadata-first triage
- move the earliest triage pass closer to Gmail metadata fetches
- avoid unnecessary full payload retrieval only when confidence is high
- preserve novelty promotion and conservative fallbacks

### Later phase, safe downgrade
- enable `light` recommendations for high-confidence low-value streams
- keep full fetch/store available
- skip only selected downstream expensive steps

### Later phase, metadata-only mode
- allow `minimal` for very stable streams
- measure false-negative rate and manual overrides

### Later phase, true suppression
- allow `suppressed` only for mature, extremely stable low-value streams
- require novelty promotion checks before suppression applies

## Recommended next implementation slice

1. Move the earliest triage step toward Gmail metadata fetches instead of post-normalization payload handling.
2. Keep additive DB/state compatibility with the existing `message_ingest_triage` and `message_stream_reputation` tables.
3. Gather evidence on when metadata-only confidence is high enough to avoid full fetches safely.
4. Expand operator-visible diagnostics only where they help validate downgrade decisions.
5. Collect more real usage data before letting `minimal` or `suppressed` affect fetch behavior.

This keeps the next slice measurable, reversible, and aligned with the current pipeline architecture.
