# PR-D6g-2: 100%-hit short-circuit for /batch cache prefilter

## Why this slice exists

Behavior half of the /batch cache prefilter feature scaffolded by
PR-D6g-1 (#439). When every item in a customer's batch is already
in their exact cache, atlas marks the batch ``status='ended'`` at
INSERT time without calling Anthropic. The customer pays nothing
to Anthropic for that batch; their /api/v1/llm/usage rollup picks
up the cache savings.

Real workload that benefits: a customer re-submits an identical
deterministic batch (e.g., retry of a job whose results were
already produced + cached). Today they pay Anthropic full price
for the resubmission.

## Scope (this PR)

**100%-hit case only.** Partial-hit handling defers to PR-D6g-3
because it requires coordinated changes to the refresh / poll
path that this slice does not touch.

### Touches `services/llm_gateway_batch.py`

- New `_CACHE_HIT_BATCH_USAGE_INSERT_SQL` constant -- direct INSERT
  (not via `trace_llm_call`) because the tracer's `_store_local`
  short-circuits on all-zero token rows. Mirrors the chat-side
  `_CACHE_HIT_USAGE_INSERT_SQL` from PR-D6b but tagged with
  `'llm_gateway.batch'` span/endpoint.
- New `_CACHE_SAVINGS_METADATA_KEY` constant -- shared name so the
  read site (/api/v1/llm/usage rollup) and write site can't drift.
- New `_try_prefilter_batch_through_cache` helper -- builds an
  envelope per item using the same `build_request_envelope` shape
  as /chat, looks each up in the customer's exact cache. Returns
  the list of hit-info dicts iff EVERY item hits; otherwise None
  (caller falls through to normal Anthropic submission).
- New `_insert_all_cache_hit_batch` helper -- INSERTs the batch row
  with `status='ended'`, `cache_prefiltered_items=N`,
  `total_items=N`, `completed_items=N`, `usage_tracked=TRUE`,
  `submitted_at=NOW()`, `completed_at=NOW()`, no
  `provider_batch_id`. Writes a per-item zero-token `llm_usage`
  row in the same transaction.
- `submit_customer_batch` calls the prefilter after the
  idempotency-replay and resume-from-crash checks but before the
  normal INSERT-then-Anthropic flow. Short-circuit fires only when
  `row is None` (no replay, no resume) AND every item hits.

### Cross-endpoint cache sharing

Same namespace as /chat (`"llm_gateway.chat"`). A customer's prior
/chat call with the same prompt populates the cache; a later
/batch with the same prompt hits. Same in reverse: a /chat after
a previously-stored batch item benefits.

The cache key is built from the request envelope (provider, model,
messages, max_tokens, temperature) -- identical requests share a
key regardless of which endpoint produced or consumed them.

## Intentional (looks wrong but is deliberate)

- **100%-hit only.** Partial-hit (some items hit, some miss)
  requires coordinated changes to:
  - The Anthropic submit shape (send only misses).
  - `_persist_batch_usage` (already handles per-item rows; need
    to merge cache-hit accounting alongside Anthropic-completed
    accounting).
  - `refresh_customer_batch_status` (the row's `total_items` vs
    `completed_items` accounting must stay consistent as the
    Anthropic side reports incremental progress on a smaller
    actual batch than the customer's logical batch).
  Each is its own decision. Ship the 100%-hit case first; partial
  in PR-D6g-3.
- **Prefilter happens AFTER replay/resume checks.** A retry of an
  idempotency-keyed batch must always replay the original record,
  not create a new ``ended`` row even if the cache state would now
  satisfy every item. Order matters.
- **Fail-open on lookup error.** If `lookup_cached_text` raises
  (DB transient, schema drift), the helper returns None and the
  caller continues normal submission. The customer pays Anthropic
  rather than seeing a 500 from a cache implementation detail.
- **`provider_batch_id` is NULL** on the short-circuit row. There
  is no Anthropic batch -- the customer's GET `/batch/{id}` will
  show `status='ended'` with `provider_batch_id=null` and
  `cache_prefiltered_items=N` so they can tell why no Anthropic
  side exists.
- **Lazy import of cache helpers** (`build_request_envelope` etc.)
  inside the prefilter helper. Avoids widening this module's
  import surface for callers that don't trigger the short-circuit.
- **All-zero token usage row direct-INSERT** (not via
  `trace_llm_call`). Same Codex-P1 lesson from PR-D6b: the tracer
  drops zero-token rows.
- **`usage_tracked=TRUE` at INSERT time.** All per-item rows are
  written in the same transaction; there is nothing for the
  refresh path to reconcile later.

## Deferred (looks missing but is on purpose)

- Partial-hit path (PR-D6g-3).
- `Cache-Control: no-store` on /batch (symmetric with PR-D6f on
  /chat).
- /batch/{id}/results endpoint -- the missing endpoint that would
  let customers retrieve cache-hit response text. Until it lands,
  the 100%-hit case is "you saved $X but here are no responses to
  retrieve" -- accounting-only feature. This PR ships the savings
  side; retrieval is a separate slice.

## Verification

- New regression tests in
  `tests/test_llm_gateway_batch_cache_100pct_hit.py`:
  - Helper / SQL constants exist.
  - Prefilter helper imports cache helpers lazily.
  - Submit flow calls prefilter only when `row is None`.
  - Short-circuit returns early via `_insert_all_cache_hit_batch`.
  - INSERT shape: `status='ended'`, all three counts equal, no
    provider_batch_id.
  - Cache namespace shared with /chat.
- `python3 -m py_compile atlas_brain/services/llm_gateway_batch.py`
  -> clean.
- `bash scripts/check_ascii_python.sh` -> passed.

## Conflict check

No file overlap with any open PR. Builds on the schema landed in
PR-D6g-1 (#439, merged).

## Diff size

- Source: ~140 LOC added to `services/llm_gateway_batch.py`
  (prefilter helper + insert helper + short-circuit hook + SQL
  constants + metadata key constant).
- Tests: ~110 LOC, source-text contract assertions.
- Plan doc: ~135 LOC.

## After this lands

Customers re-submitting identical batches (deterministic retry
workloads) get a free batch -- Anthropic isn't called and the
customer's bill doesn't move. The savings show up in
`/api/v1/llm/usage` `total_cache_savings_usd` (PR-D6c surface).
The batch row's `cache_prefiltered_items` shows the count.

Partial-hit (PR-D6g-3) is the next slice -- bigger lift because
it requires the refresh path to know about the cache-hit subset.
