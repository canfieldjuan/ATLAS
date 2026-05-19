# PR-Content-Ops-CFPB-Live-Provider-Smoke

## Why this slice exists

PR #628 added live-provider mode to the G2 review-source Postgres smoke and
left CFPB/support-ticket-like parity as the next narrow follow-up. CFPB is the
public support-ticket-like source path, so it should be able to prove the same
imported source rows can run through the product `PipelineLLMClient` seam
without changing the default offline smoke.

## Scope (this PR)

1. Add `--llm offline|pipeline` to the CFPB Postgres smoke.
2. Wire `--llm pipeline` through the existing shared Postgres smoke helper.
3. Document the live-provider option at the CFPB smoke command sites.
4. Add focused tests for default offline behavior and provider-port wiring.
5. Replace the stale merged #628 in-flight row with this slice's claim.

### Files touched

- `scripts/smoke_content_ops_cfpb_source_postgres.py`
- `tests/test_smoke_content_ops_cfpb_source_postgres.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-CFPB-Live-Provider-Smoke.md`

## Mechanism

The CFPB smoke keeps `offline` as the default. When `--llm pipeline` is passed,
it constructs the product LLM client and skill registry, then passes them into
`generate_imported_target_drafts`. That helper already defaults to
deterministic offline ports, so existing CFPB smoke behavior remains unchanged.

## Intentional

- No new generator, importer, or source adapter.
- Tests monkeypatch provider factories; they do not call a real LLM provider.
- This does not run the manual live smoke because this environment does not
  have `EXTRACTED_DATABASE_URL` or provider credentials loaded.

## Deferred

- A real CFPB live-provider run remains an operator smoke after merge.
- Hosted UI/source-run controls for choosing live vs offline generation remain
  separate product work.

## Verification

- Focused CFPB smoke tests -> `9 passed`.
- Python compile check for edited script/test -> passed.
- `git diff --check` -> passed.
- ASCII check on edited script/test -> passed.
- `scripts/validate_extracted_content_pipeline.sh` -> passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` -> passed.
- `scripts/audit_extracted_standalone.py` -> passed.
- `scripts/check_ascii_python.sh` -> passed.
- `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| CFPB smoke CLI | ~25 |
| Tests | ~50 |
| Docs and coordination | ~20 |
| Plan | ~55 |
| **Total** | **~150** |

This is below the 400 LOC review budget.
