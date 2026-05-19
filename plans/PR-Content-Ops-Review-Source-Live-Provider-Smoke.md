# PR-Content-Ops-Review-Source-Live-Provider-Smoke

## Why this slice exists

The review-source Postgres smoke currently proves G2 review rows can be
exported, inspected, imported into `campaign_opportunities`, and persisted as
offline deterministic drafts. The deferred gap is live provider generation over
those imported review-source rows. This slice adds that operator path without
changing the default offline smoke.

## Scope (this PR)

1. Add an opt-in provider mode to the review-source Postgres smoke.
2. Let the shared source-row Postgres smoke helper accept injected LLM and skill
   ports while preserving the existing offline defaults.
3. Document the live-provider flag at the existing review-source smoke command
   sites.
4. Add focused tests for default offline behavior and live-provider wiring.

### Files touched

- `scripts/content_ops_source_postgres_smoke_helpers.py`
- `scripts/smoke_content_ops_review_source_postgres.py`
- `tests/test_content_ops_source_postgres_smoke_helpers.py`
- `tests/test_smoke_content_ops_review_source_postgres.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Review-Source-Live-Provider-Smoke.md`

## Mechanism

`generate_imported_target_drafts` keeps constructing
`DeterministicCampaignLLM` and `StaticCampaignSkillStore` when no ports are
supplied. The review-source smoke adds `--llm offline|pipeline`; `offline` is
the default, and `pipeline` passes the product LLM client plus skill registry
through the same helper into
`generate_campaign_drafts_from_postgres`.

## Intentional

- The default remains offline so CI and host readiness checks do not require
  provider credentials.
- The CFPB Postgres smoke keeps using the shared helper defaults and does not
  need a CLI change in this slice.
- Tests monkeypatch the provider factory and skill registry; they do not make a
  live provider call.

## Deferred

- A real operator run against a host database with provider credentials remains
  a manual smoke after merge.
- Broader `--llm pipeline` parity for the CFPB smoke can land separately if the
  support-ticket path needs live-provider proof.

## Verification

- Focused smoke/helper tests -> `16 passed`.
- Python compile check for edited scripts/tests -> passed.
- ASCII check on edited Python/test files -> passed.
- `git diff --check` -> passed.
- `scripts/validate_extracted_content_pipeline.sh` -> passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` -> passed.
- `scripts/audit_extracted_standalone.py` -> passed.
- `scripts/check_ascii_python.sh` -> passed.
- `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Smoke helper and CLI | ~45 |
| Tests | ~65 |
| Docs and coordination | ~35 |
| Plan | ~55 |
| **Total** | **~200** |

This is below the 400 LOC review budget.
