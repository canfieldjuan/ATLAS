# PR-Deflection-Request-ID-Redaction

## Why this slice exists

The 2026-06-15 historical request-id sweep found two raw
`content-ops-<32 hex>` request IDs still committed in current `origin/main`.
Both were live artifact capabilities before the sweep and were relocked through
the existing signed Stripe revocation path. The root cause is that older proof
docs predated the paid-artifact redaction discipline introduced by the later
full-funnel proof. This change fixes the repository exposure at the source by
redacting the committed docs/fixture and adding a regression test so the same
capability-shaped IDs cannot be recommitted silently.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Redact historical raw deflection request IDs and result URLs from the
   committed validation docs/fixture that still contain them.
2. Update stale proof wording to record the 2026-06-15 revocation sweep: both
   historical IDs were `200` before revocation and `403` afterward.
3. Redact the remaining stale personal email and local absolute path from the
   same proof artifacts.
4. Add and enroll a focused regression test for strict
   `content-ops-<32 hex>` request IDs in committed files, plus proof-artifact
   checks for local home paths and non-placeholder emails.

### Review Contract

Acceptance criteria:

- No committed file in this PR's tree contains a strict
  `content-ops-<32 hex>` request ID.
- The touched proof artifacts contain no local `/home/...` paths and no
  non-placeholder email addresses.
- The affected validation docs retain proof value through stable sanitized
  labels / SHA prefixes instead of runnable request IDs or result URLs.
- The new test fails if a raw strict deflection request ID is reintroduced in a
  committed file, and the test is enrolled in the extracted-checks CI runner.

Affected surfaces:

- Historical Content Ops deflection validation docs and one summary fixture.
- Test-only repository hygiene coverage.
- Extracted-checks CI runner enrollment list.

Risk areas:

- Over-redacting the docs so the historical proof loses useful status context.
- Under-redacting URLs, personal emails, or local paths that still carry stale
  proof capabilities/context.

Triggered rules:

- R1 requirements match, R2 test evidence, R8 docs/artifact truthfulness, R13
  class fix, R14 codebase verification.

### Files touched

- `docs/extraction/validation/content_ops_faq_deflection_portfolio_hosted_smoke_2026-05-30.md`
- `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md`
- `docs/extraction/validation/fixtures/deflection_full_volume_live_proof_20260614/summary.json`
- `plans/PR-Deflection-Request-ID-Redaction.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_docs_no_raw_deflection_request_ids.py`

## Mechanism

The docs replace each raw request ID with a deterministic
`content-ops-[redacted:<sha12>]` label. Result URLs that previously embedded the
raw ID become redacted URL placeholders with the same SHA label. The JSON
fixture keeps scalar status evidence but stores a redacted request-id label plus
the SHA prefix instead of the live request ID.

The regression test asks git for tracked files via `git ls-files` and scans
those files from the repository root for the strict request-id shape
`content-ops-[0-9a-f]{32}`. It also scans the touched proof artifacts for local
home paths and non-placeholder email addresses. The test fails with file paths
and counts rather than printing matched sensitive values. The test is enrolled
in `scripts/run_extracted_pipeline_checks.sh` next to the deflection proof-doc
tests so CI runs the guard.

## Intentional

- The proof docs keep hosted status values, byte/row counts, and command shape,
  but commands are no longer copy-paste runnable against historical requests.
  That is intentional: these request IDs are capabilities when paired with the
  tenant's auth context.
- The test targets the strict request-id shape rather than every
  `content-ops-*` token so branch names, plan slugs, route names, and prose
  remain allowed.
- The email/path checks are proof-artifact scoped, not repo-wide, because the
  repository contains legitimate examples and fixtures that would make a global
  email ban noisy.

## Deferred

None.

Parked hardening: none.

## Verification

- Command: `python -m pytest` targeting
  `tests/test_docs_no_raw_deflection_request_ids.py` -- 6 passed.
- Sanitized working-tree scan over tracked files -- 0 strict raw request IDs.
- Sanitized all-text working-tree scan including untracked PR files -- 0 strict raw request IDs.
- Sanitized proof-artifact scan -- 0 non-placeholder emails and 0 local home
  paths across touched proof docs/fixture.
- Extracted pipeline check runner `scripts/run_extracted_pipeline_checks.sh` --
  passed; content-ops pytest block reported 4,291 passed / 10 skipped, and
  extracted reasoning-core checks reported 295 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/content_ops_faq_deflection_portfolio_hosted_smoke_2026-05-30.md` | 14 |
| `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md` | 23 |
| `docs/extraction/validation/fixtures/deflection_full_volume_live_proof_20260614/summary.json` | 15 |
| `plans/PR-Deflection-Request-ID-Redaction.md` | 125 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_docs_no_raw_deflection_request_ids.py` | 140 |
| **Total** | **318** |
