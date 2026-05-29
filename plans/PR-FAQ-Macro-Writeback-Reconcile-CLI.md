# PR-FAQ-Macro-Writeback-Reconcile-CLI

## Why this slice exists

PR-FAQ-Macro-Writeback-Pending-Reconcile made pending Zendesk macro mappings
fail safe: no match or ambiguous exact-title matches stay
`zendesk_macro_mapping_pending_reconcile`. That protects customers from
wrong-macro attachment, but operators still need a direct way to inspect and
retry pending rows without waiting for another publish request. This slice adds
that narrow operator path over the same repository and Zendesk reconciliation
logic.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Add a repository method for tenant-scoped pending macro writeback mappings.
2. Add a public Zendesk pending-mapping reconcile method that reuses the
   unique exact-title search behavior.
3. Add a CLI that lists pending mappings, attempts reconciliation, and emits a
   JSON summary.
4. Add focused tests for pending-row listing, single-match reconciliation, and
   ambiguous/no-match CLI output.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Reconcile-CLI.md` — plan for this slice.
- `extracted_content_pipeline/faq_macro_writeback.py` — repository contract.
- `extracted_content_pipeline/faq_macro_writeback_postgres.py` — pending mapping query.
- `extracted_content_pipeline/faq_macro_writeback_zendesk.py` — public pending reconcile result.
- `scripts/reconcile_content_ops_faq_macro_writebacks.py` — operator CLI.
- `scripts/run_extracted_pipeline_checks.sh` — CI enrollment for the new CLI test.
- `tests/test_extracted_ticket_faq_macro_writeback_postgres.py` — pending query coverage.
- `tests/test_extracted_ticket_faq_macro_writeback_zendesk.py` — public reconcile coverage.
- `tests/test_extracted_ticket_faq_macro_writeback_reconcile_cli.py` — CLI summary coverage.

## Mechanism

The Postgres adapter exposes `list_pending_mappings(...)` for one
`TenantScope`, one platform, and a bounded limit. It returns rows where the
mapping is still pending and has no external id.

`ZendeskMacroPublishProvider.reconcile_pending_mapping(...)` takes one pending
mapping, reads the reserved title from mapping metadata, searches Zendesk by
that title, and upserts only when exactly one normalized title match exists.
Missing credentials, missing title, no match, ambiguous match, and persistence
failures are returned as explicit result statuses/errors.

The CLI wires those pieces with the existing `EXTRACTED_DATABASE_URL` /
`DATABASE_URL` convention and centralized host config. By default it is a dry
run that reports what would be attempted. `--execute` performs reconciliation.

## Intentional

- The CLI is tenant-scoped by required `--account-id`; there is no all-tenant
  scan in this slice.
- Dry run is the default because this is an operator recovery command touching
  live support-tool mappings.
- Reconcile still requires a unique exact-title match. Ambiguous rows remain
  pending and visible in the summary.
- The CLI backfills mappings only. It does not update macro body content; the
  normal publish path still owns content updates once the mapping is recovered.

## Deferred

- `PR-FAQ-Macro-Writeback-Tenant-Credentials`: tenant-scoped encrypted
  credential storage for multi-customer live writeback.
- `PR-FAQ-Macro-Writeback-Publish-UI`: review UI action for the macro publish
  route.

Parked hardening: none

## Verification

- python -m pytest tests/test_extracted_ticket_faq_macro_writeback_postgres.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_extracted_ticket_faq_macro_writeback_reconcile_cli.py -q — 23 passed.
- python -m py_compile extracted_content_pipeline/faq_macro_writeback.py extracted_content_pipeline/faq_macro_writeback_postgres.py extracted_content_pipeline/faq_macro_writeback_zendesk.py scripts/reconcile_content_ops_faq_macro_writebacks.py tests/test_extracted_ticket_faq_macro_writeback_reconcile_cli.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- python scripts/check_extracted_imports.py — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py — passed.
- python scripts/smoke_extracted_pipeline_imports.py — passed.
- python scripts/smoke_extracted_pipeline_standalone.py — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-reconcile-cli.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~101 |
| Contract / adapters | ~158 |
| CLI | ~172 |
| CI enrollment | ~1 |
| Tests | ~281 |
| Total | ~713 |

This is over the 400 LOC soft cap because the operator path needs a real
contract method, a host CLI, and tests at all three integration points to avoid
another under-enforced safety claim.
