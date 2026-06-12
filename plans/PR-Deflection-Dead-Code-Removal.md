# PR-Deflection-Dead-Code-Removal

## Why this slice exists

The 2026-06 deflection-lane dead-code audit (issues #1500, #1501, #1502)
confirmed two zero-consumer subsystems with grep-proven evidence at commit
7dd668c:

1. `assign_support_ticket_clusters` in
   `extracted_content_pipeline/support_ticket_clustering.py` was kept as a
   backward-compat delegate when PR-Deflection-Cluster-Preview-Skip introduced
   `assign_support_ticket_clusters_with_diagnostics`. Its only callers were
   one signature-pinning test and an internal `_ensure_clustered` call; no
   production module, script, or MCP surface imports it.

2. The snapshot `teaser` + `locked_questions` subsystem in
   `extracted_content_pipeline/faq_deflection_report.py` was built across
   three slices (PR-Deflection-Snapshot-Teaser, PR-Deflection-Teaser-Top-Answer,
   PR-Deflection-Teaser-Locked-Dedupe) for a portfolio rendering that every
   slice deferred and that never landed: `git log -S 'teaser' -- portfolio-ui/`
   returns no commits, the portfolio proxy `projectSnapshot()` strips both keys
   fail-closed before any browser sees them, and the readonly MCP server uses a
   separate projection. The producer ran on every snapshot for nothing, and the
   teaser was the only path that exposed a full answer body pre-payment.

Keeping either would preserve misleading API surface, a documented contract no
client honors, and tests pinning behavior with zero consumers. Operator decided
remove-fully over render (the paid-report reframe already moved teaser/FOMO
copy into the report itself).

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Dead-code removal (audit follow-through)

1. Remove the `assign_support_ticket_clusters` delegate and its `__all__`
   entry; `_ensure_clustered` calls the diagnostics entry point directly.
2. Remove `DeflectionSnapshot.teaser` / `DeflectionSnapshot.locked_questions`,
   the `_snapshot_teaser` / `_teaser_*` / `_is_teaser_eligible` /
   `_select_full_teaser_item` helper cluster, the now-orphaned
   `_RESOLUTION_EVIDENCE_SCOPE_SCOPED` constant, and
   `DEFAULT_DEFLECTION_TEASER_PREVIEW_COUNT`.
3. Remove the `deflection_snapshot_teaser_preview_count` config field, its
   validation branch, and the threading through
   `_gate_deflection_report_artifacts` in
   `extracted_content_pipeline/api/control_surfaces.py`.
4. Update the frontend contract docs and checked-in snapshot example so the
   documented shape matches the producer exactly: `summary` + `top_questions`
   only, no answer bodies before paid unlock.
5. Retarget or delete the pinning tests; add
   `test_deflection_snapshot_never_exposes_answer_bodies` to pin the
   strengthened privacy invariant (snapshot contains zero answer bodies).
6. Drop stale `locked_questions` fixture keys from the paid-postgres smoke
   script and submit-handoff test fixture.

### Review Contract

- Acceptance criteria:
  - [ ] `assign_support_ticket_clusters` no longer exists anywhere in the
        package, `__all__`, or tests; `_ensure_clustered` produces identical
        annotations via the diagnostics entry point.
  - [ ] `DeflectionSnapshot.as_dict()` serializes exactly `summary` and
        `top_questions`; no `teaser`, `locked_questions`, or answer-body text
        appears in any snapshot payload, gated execute result, or checked-in
        example.
  - [ ] `ContentOpsControlSurfaceApiConfig` no longer accepts or validates
        `deflection_snapshot_teaser_preview_count`; no caller threads a
        teaser count.
  - [ ] Contract docs and `content_ops_faq_deflection_snapshot_example.json`
        match the producer shape byte-for-byte under the contract-docs test.
  - [ ] No production behavior changes outside the deflection snapshot
        projection; the readonly MCP server suite passes unchanged.
- Affected surfaces: content-ops API response shape (unpaid snapshot),
  extracted-package public symbols, frontend contract docs, tests, one smoke
  script. No DB, auth, or scheduler changes.
- Risk areas: backcompat (removed public symbols and snapshot keys; mitigated
  by grep-proven zero consumers in #1501/#1502), stale stored snapshot JSONB
  rows still carrying old keys (no read path selects them).
- Reviewer rules triggered: R1, R2, R5, R10 (extracted-package + API-surface
  path triggers), R14.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/support_ticket_clustering.py`
- `docs/frontend/content_ops_faq_report_contract.md`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `scripts/smoke_content_ops_deflection_paid_postgres.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_ops_live_execute_harness.py`
- `tests/test_extracted_support_ticket_clustering_scale.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`
- `plans/PR-Deflection-Dead-Code-Removal.md`

## Mechanism

`build_deflection_snapshot` drops the `teaser_preview_count` parameter and the
teaser/locked construction; `DeflectionSnapshot.as_dict()` serializes only
`summary` and `top_questions`. The snapshot privacy posture strictly tightens:
the teaser full answer was the single pre-payment answer-body exposure, so
after this slice no answer text, steps, evidence, or source IDs exist anywhere
in the snapshot payload. Tests that asserted the intentional teaser leak now
assert its absence (`"Open Analytics" not in gated_payload`, visible answer
count 0). `_ensure_clustered` keeps identical behavior by unpacking the
diagnostics tuple and discarding the diagnostics dict, exactly what the
deleted delegate did.

## Intentional

- No API versioning shim for the removed snapshot keys: the only production
  consumer (portfolio proxy) already projects them away, so no shipped client
  observes the change. Stored historical snapshots in
  `content_ops_deflection_reports.snapshot` JSONB may still contain the old
  keys; readers tolerate extra keys, and no read path selects them.
- `deflection_snapshot_content_opportunities` (MCP projection) is untouched;
  it never read the removed keys.
- The unrelated "teaser" marketing copy in `docs/report_catalog.md` and the
  digest skills is out of scope (different product surface).
- The companion stale-deferral docs cleanup flagged in #1500 (1,000-row
  fixture text in an archived plan) stays archived untouched; archives are
  historical records.

## Deferred

- If a free-tier conversion teaser is ever wanted again, it should be designed
  against the paid-report-reframe surface, not restored from this code; the
  removed shape is recoverable from git history and issue #1502.
- Backfill/normalization of historical snapshot JSONB rows (removing stale
  `teaser`/`locked_questions` keys) is unnecessary for correctness and left
  out.

## Verification

- `pytest tests/test_content_ops_deflection_report.py tests/test_extracted_support_ticket_clustering_scale.py -q` - 38 passed.
- `pytest tests/test_content_ops_faq_report_contract_docs.py tests/test_smoke_content_ops_deflection_submit_handoff.py -q` - 29 passed.
- `pytest tests/test_extracted_content_control_surface_api.py tests/test_extracted_content_ops_live_execute_harness.py -q` - 155 passed, 1 skipped.
- `pytest tests/test_mcp_content_ops_deflection_readonly.py -q` - 13 passed (MCP boundary unaffected).
- `pytest tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_content_ops_deflection_resolution_live_proof.py tests/test_content_ops_faq_deflection_live_upload_fixture.py tests/test_smoke_content_ops_support_ticket_package.py -q` - 154 passed.
- `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_deflection_submit.py tests/test_extracted_support_ticket_input_package.py -q` - all passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - clean.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - 0 findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `ruff check` over all touched Python files - clean.
- `git grep` residue sweep for `teaser`/`locked_questions`/`assign_support_ticket_clusters`
  across py/ts/js/json/md - only negative-assertion test lines and unrelated
  marketing copy remain.

## Estimated diff size

~520 changed lines, overwhelmingly deletions (35 insertions / 481 deletions
before the plan doc). Over the 400 LOC soft cap because the slice is a pure
removal of an audited dead subsystem plus its pinning tests; splitting it
would leave the package importing deleted symbols between PRs.
