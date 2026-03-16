# Redaction Architecture v2 (Deterministic Map-Based)

## Context

Current model/hybrid redaction can be expensive because it chunks message bodies and calls the LLM per chunk.
That improves coverage, but cost and latency scale with message volume and chunk size.

This proposal introduces a deterministic, map-driven redaction architecture that can be run quickly on demand,
while preserving stable placeholders and enabling optional controlled unredaction.

## Threat model and goals

### Threat model
- **Primary risk:** PII leakage via indexed/searchable text, logs, or downstream retrieval outputs.
- **Attacker capabilities considered:**
  - access to redacted retrieval outputs
  - accidental exposure of vector-store text snippets
  - accidental operator access to runtime logs/reports
- **Out of scope for this phase:**
  - cryptographic escrow/HSM-backed key wrapping for reversible maps
  - fine-grained policy engine for per-user/per-field unredaction controls

### Goals
1. **Fast redaction at indexing/search time** using deterministic rules.
2. **Stable placeholders** preserving distinction and coreference:
   - same entity in same scope → same placeholder
   - different entities → distinct placeholders
3. **Structured mapping layer** that can later support authorized unredaction.
4. **Compatibility-first rollout**: keep current `regex|model|hybrid` flows intact.

## Proposed pipeline

### 1) Detection + normalization pass
- Default path: regex-based detectors for common sensitive categories.
- Initial categories (implemented in scaffolding):
  - email
  - phone
  - URL
  - account-like tokens (keyword-prefixed and long numeric identifiers)
- Normalize matches to canonical keys before mapping:
  - email: lowercase
  - phone: digits-only, normalize leading country prefix
  - URL: lowercase + trim trailing slash
  - account-like: alphanumeric-only lowercase

Optional future extension:
- run periodic low-frequency LLM-assisted pattern discovery over sampled corpora to propose new regexes,
  then human-review and promote into deterministic detectors.

### 2) Entity map generation (stable per-scope IDs)
- Build/maintain a map object per configured scope (examples: message thread, mailbox/account, run).
- Data model:
  - `entity_to_placeholder[category][normalized_value] -> <REDACTED_CATEGORY_TOKEN>`
  - `placeholder_to_entity[placeholder] -> first_observed_raw_value`
  - per-category counters for deterministic ordinal assignment (`A, B, ...`)
- Placeholder format:
  - `<REDACTED_EMAIL_A>`
  - `<REDACTED_PHONE_A>`
  - `<REDACTED_URL_B>`
  - `<REDACTED_ACCOUNT_C>`

### 3) Deterministic replacement engine
- Apply category detectors in deterministic order.
- Replace matched entities with placeholders allocated from the scope map.
- Reuse the same map instance across chunks/documents in scope to preserve coreference.
- Output is deterministic and O(n) over text length (regex pass + dict lookups).

### 4) Optional unredaction (authorized contexts)
- If an authorized context has access to map storage, placeholders can be resolved back.
- Current scaffolding includes map-level `unredact()` helper.
- Production-grade rollout should enforce policy checks before unredaction.

## Placeholder and scope strategy

### Placeholder strategy
- Keep placeholders structured and readable for downstream LLM/retrieval use.
- Category + stable token improves semantic continuity while preserving privacy.

### Scope boundaries (configurable)
Recommended defaults:
- **Indexing run scope**: reuse map during a single indexing pass for continuity.
- **Account scope**: optional long-lived map per account for stronger cross-run coreference.
- **Message scope**: strongest minimization, weakest cross-document linking.

Trade-off:
- larger scope = better cross-document consistency, larger blast radius if map leaks.

## Storage model proposal

For persistent reversible mode (future), store map metadata separately from redacted text:

- `redaction_scopes`
  - `scope_id`
  - `scope_type` (run/account/thread)
  - `created_at`, `expires_at`
  - policy metadata (owner, retention class)

- `redaction_entities`
  - `scope_id`
  - `category`
  - `normalized_hash` (preferred for joins/lookup)
  - `encrypted_raw_value` (or wrapped secret)
  - `placeholder`
  - `first_seen_at`, `last_seen_at`

Security/retention guidance:
- Keep map storage encrypted at rest (same or stronger posture than primary DB).
- Separate access controls from generic search path.
- Apply TTL/retention based on scope type and compliance needs.
- Minimize logs: never print raw map values.

## Performance model vs chunk-by-chunk LLM redaction

### Current chunk LLM path
- Cost roughly proportional to chunk count × model latency/token cost.
- Adds network/endpoint variability and failure modes.

### Deterministic map path
- Cost proportional to regex scans + dictionary operations in-process.
- No per-chunk model calls.
- Predictable runtime and low marginal cost for repeated requests.

Expected impact:
- substantial reduction in indexing latency and local model load,
- improved determinism and repeatability of redacted outputs.

## Migration path

1. **Phase 0 (current):**
   - add deterministic map module + tests + docs, no CLI breaking changes.
   - keep existing redaction modes untouched.
2. **Phase 1:**
   - add optional config mode (e.g., `deterministic`) and wire map lifecycle in indexing.
   - define default scope (`run` vs `account`) and persistence toggle.
3. **Phase 2:**
   - persist map tables with retention + access control.
   - add audited unredaction command for authorized operators.
4. **Phase 3:**
   - optional regex discovery workflow using periodic LLM-assisted suggestions + human approval.

## Incremental implementation in this repo

Implemented scaffolding:
- `src/inbox_vault/redaction_map.py`
  - deterministic detectors + placeholder mapper
  - stable cross-chunk replacement via reusable map object
  - `to_dict()/from_dict()` serialization hooks
  - `unredact()` helper for authorized-path future use
- `src/inbox_vault/redaction.py`
  - non-breaking utility wrapper: `deterministic_map_redact_text(...)`
- `tests/test_redaction_map.py`
  - coverage for stable placeholders, category handling, cross-chunk consistency, and serialization round-trip

No changes were made to default CLI behavior or existing `regex|model|hybrid` modes.
