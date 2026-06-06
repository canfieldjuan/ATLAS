# PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Cleanup

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Search-Seeder added a real seed path for the
synthetic B2B SaaS FAQ demo, and the reviewer independently confirmed that
deleting the seeded FAQ row cascades the projected search rows. The seeder still
leaves cleanup as a manual SQL step, which is workable for one local proof but
awkward and risky for repeated shared-environment demo seeding.

This slice adds explicit cleanup by FAQ id and account id to the same operator
script. It keeps the cleanup narrow: no broad account cleanup, no corpus-wide
delete, and no route changes.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Add `--cleanup-faq-id` support to `seed_content_ops_faq_saas_demo.py`.
2. Delete only the requested FAQ id within the requested account id.
3. Parse the asyncpg `DELETE N` command tag and fail closed when it is malformed
   or when the rowcount is not exactly one.
4. Add focused positive and negative cleanup tests in the existing enrolled SaaS
   demo corpus test file.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Cleanup.md` | Plan contract for this cleanup hardening slice. |
| `scripts/seed_content_ops_faq_saas_demo.py` | Adds explicit FAQ-id cleanup mode and result payload. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Adds cleanup success and failure coverage. |

## Mechanism

The CLI keeps seed mode as the default. When `--cleanup-faq-id <id>` is passed,
`_run(...)` calls `cleanup_saas_demo_faq(...)` instead of the seed path.

Cleanup executes:

```sql
DELETE FROM ticket_faq_markdown
 WHERE id = $1::uuid
   AND account_id = $2
```

Migration 327 already makes `ticket_faq_search_documents.faq_id` cascade from
`ticket_faq_markdown.id`, so the source row delete is the correct cleanup
integration point. The script parses the returned command tag, requires
`DELETE 1`, and returns a compact JSON payload with the requested FAQ id,
deleted row count, raw delete status, and error if any.

## Intentional

- Cleanup is by explicit FAQ id plus account id only. Broad demo-account cleanup
  is deferred so this slice cannot delete unrelated FAQ rows.
- No migration or live route validation is embedded in this script.
- Search-document rowcount is not queried here; the database FK cascade is the
  existing integration contract.

## Deferred

- Future PR: route-level live validation against a deployed host after seeding
  and cleaning up the SaaS demo in the target environment.
- Future PR: optional cleanup by manifest file if the demo flow starts seeding
  multiple FAQ ids per run.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q - 12 passed.
- python -m py_compile scripts/seed_content_ops_faq_saas_demo.py tests/test_content_ops_faq_saas_demo_corpus.py - passed.
- python scripts/seed_content_ops_faq_saas_demo.py --database-url postgresql://example --account-id acct-demo --cleanup-faq-id '' --json - returned exit 1 with the expected cleanup validation error.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Cleanup.md - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . - passed.
- git diff --check - passed.
- Live DB cleanup not run: this checkout does not expose `EXTRACTED_DATABASE_URL` or `DATABASE_URL`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Cleanup.md` | 87 |
| `scripts/seed_content_ops_faq_saas_demo.py` | 66 |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | 96 |
| **Total** | **249** |
