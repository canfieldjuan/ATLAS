# PR-Content-Ops-Reasoning-Upsert

## Why this slice exists

The DB-backed Content Ops reasoning context repository can read
`campaign_reasoning_contexts`, but `save_context` only inserts. Re-running a
host ETL for the same account, target mode, and selector set can leave duplicate
rows where read behavior depends on `updated_at` instead of a deterministic
write contract.

This slice makes the write path replay-safe before more hosts depend on the
Postgres reasoning provider.

## Scope (this PR)

1. Add a stable selector-key migration for `campaign_reasoning_contexts`.
2. Change `PostgresCampaignReasoningContextRepository.save_context` to upsert
   by account, target mode, and selector key.
3. Add focused tests for upsert SQL, selector-key order independence, and
   migration ownership.
4. Claim the slice in the coordination ledger.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Upsert.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/campaign_reasoning_postgres.py`
- `extracted_content_pipeline/storage/migrations/278_campaign_reasoning_context_upsert.sql`
- `extracted_content_pipeline/manifest.json`
- `tests/test_extracted_campaign_reasoning_postgres.py`
- `tests/test_extracted_campaign_manifest.py`

## Mechanism

`save_context` already normalizes selectors into case-as-given and lowercase
forms. This PR derives a stable MD5 selector key from the sorted cleaned
selector set, writes it to `selector_key`, and uses:

```sql
ON CONFLICT (account_id, target_mode, selector_key)
DO UPDATE SET selectors = EXCLUDED.selectors,
              payload = EXCLUDED.payload,
              updated_at = NOW()
```

Migration 278 adds and backfills `selector_key`, collapses pre-existing
duplicates to the newest row for each replay key, then creates the unique index
that makes replay writes deterministic.

## Intentional

- Existing read semantics are unchanged: reads still match with
  `selectors && $2::text[]` and keep selector-priority ordering.
- The selector key is internal persistence metadata; it is not exposed through
  any public API or response model.
- MD5 is used as a compact deterministic key, not as a security primitive. It
  also keeps the SQL backfill extension-free.
- Pre-existing duplicates are collapsed during migration with newest row
  winning. That matches the existing read tie-breaker and prevents the unique
  index from failing on already-replayed ETL data.

## Deferred

- No UI for reasoning context seed/update history.

## Verification

```bash
python -m pytest tests/test_extracted_campaign_reasoning_postgres.py
python -m pytest tests/test_extracted_campaign_manifest.py
python -m py_compile extracted_content_pipeline/campaign_reasoning_postgres.py tests/test_extracted_campaign_reasoning_postgres.py tests/test_extracted_campaign_manifest.py
bash scripts/local_pr_review.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Upsert.md` | 87 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/campaign_reasoning_postgres.py` | 23 |
| `extracted_content_pipeline/storage/migrations/278_campaign_reasoning_context_upsert.sql` | 52 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `tests/test_extracted_campaign_reasoning_postgres.py` | 91 |
| `tests/test_extracted_campaign_manifest.py` | 9 |
| **Total** | **~269** |
