# PR-Content-Ops-FAQ-SaaS-Demo-Search-Seeder

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Corpus added a synthetic B2B SaaS support-ticket
corpus, and PR-Content-Ops-FAQ-SaaS-Demo-Artifact added the generated Markdown
FAQ artifact. The remaining demo handoff gap is search: the hosted FAQ search
route can only return the SaaS demo FAQ after a host operator has a repeatable
way to seed that generated FAQ into the existing ticket FAQ tables and search
projection.

This slice adds the thinnest write path for that handoff. It does not change the
route or search repository; it uses the existing FAQ producer, FAQ repository,
approval/status transition, and search repository projection path.

The diff is over the 400 LOC soft target because the vertical slice needs the
operator script, fake-boundary orchestration tests, and plan together to prove
the real flow without adding runtime API code. Splitting the script from its
tests would leave an unproven DB-write handoff.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Vertical slice

1. Add a CLI script that generates the SaaS FAQ from the checked CSV, saves it as
   a ticket FAQ draft, approves it, and verifies the projected search result.
2. Return a compact JSON summary with the seeded FAQ id, corpus id, generated
   item count, projected document count, and verification search result.
3. Add focused tests in the already-enrolled SaaS demo corpus test file proving
   the generated draft projects into searchable documents and the seeder
   orchestrates save -> approve/project -> search.
4. Keep hosted route code, repository code, migrations, and frontend code
   unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Search-Seeder.md` | Plan contract for this vertical demo seeding slice. |
| `scripts/seed_content_ops_faq_saas_demo.py` | Operator CLI for seeding the generated SaaS FAQ into DB-backed FAQ search. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Existing enrolled test file extended with seeder/search-projection coverage. |

## Mechanism

The script reads `support_ticket_saas_demo_sources.csv`, normalizes it with
`source_rows_to_campaign_opportunities`, and calls `build_ticket_faq_markdown`
with the same deterministic parameters as the checked Markdown artifact.

It then creates a `TicketFAQDraft` with `metadata.corpus_id`, persists it through
`PostgresTicketFAQRepository.save_drafts`, and calls `update_status(...,
"approved")`. The existing repository status transition is the integration point
that replaces search projection rows via `build_ticket_faq_search_documents`.
After that, the script calls `PostgresTicketFAQSearchRepository.search` with a
known SaaS query and fails closed if no result comes back for the requested
account/corpus/status.

The tests avoid a live DB by faking the repository boundary, while still using
the real generator and search projection/search code. A live DB run remains an
operator command:

```bash
python scripts/seed_content_ops_faq_saas_demo.py \
  --database-url "$EXTRACTED_DATABASE_URL" \
  --account-id "$ATLAS_FAQ_SEARCH_ACCOUNT_ID" \
  --json
```

## Intentional

- No hosted route change. This slice seeds data the existing route can read.
- No migration runner is embedded in the seeder; operators should run the
  existing extracted pipeline migrations before seeding.
- No cleanup command lands here. The script returns the seeded FAQ id so cleanup
  can use the existing FAQ/search tables or a later cleanup helper.

## Deferred

- Future PR: route-level live validation against a deployed host after seeding
  the SaaS demo corpus in the target environment.
- Future PR: optional cleanup helper for demo FAQ ids if operators need repeated
  reseeding in shared environments.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q - 7 passed.
- python -m py_compile scripts/seed_content_ops_faq_saas_demo.py tests/test_content_ops_faq_saas_demo_corpus.py - passed.
- python scripts/seed_content_ops_faq_saas_demo.py --database-url '' --account-id '' --corpus-id '' --target-id '' --status '' --query '' --limit 0 - returned exit 1 with the expected fail-closed validation errors.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Search-Seeder.md - passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . - passed.
- git diff --check - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- Live DB seed not run: this checkout does not expose `EXTRACTED_DATABASE_URL` or `DATABASE_URL`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Search-Seeder.md` | 106 |
| `scripts/seed_content_ops_faq_saas_demo.py` | 288 |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | 163 |
| **Total** | **557** |
