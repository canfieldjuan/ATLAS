# PR-Deflection-Provider-Export-Intake-Guidance

## Why this slice exists

#1384 is nearly closed. The parser/admission mechanics now handle BOMs,
encodings, delimiters, HTML, zero-usable rejects, partial-coverage warnings,
Zendesk scalar comment/history columns, private-note exclusion, and provider
full-thread fixtures. The remaining root cause is documentation drift: the
operator-facing repo guidance still does not clearly say which help-desk export
shape can produce publishable FAQ answers and which shape can only produce a
gap-list/preview.

This change fixes the root at the repo guidance layer. It does not treat a
parser symptom: the parser behavior is already locked by enrolled tests, and
the missing piece is making the proven contract visible so future intake copy,
fixtures, and support instructions do not drift back to "any ticket CSV is
enough."

## Scope (this PR)

Ownership lane: content-ops/deflection-parser/provider-ingestion
Slice phase: Production hardening

1. Document the provider-export contract in the packaged fixture README:
   full-thread exports with public customer wording and public agent replies
   can support publishable answers; ticket-index-only exports remain gap-list
   preview inputs.
2. Document private/internal note handling and the public atlas-portfolio
   handoff boundary: buyer uploads raw bytes, but ATLAS owns parsing/admission.
3. Tie the guidance to the CI-enrolled proof tests so future changes know
   which behavior is contractual.

### Review Contract

- Acceptance criteria:
  - [ ] Guidance distinguishes full-thread exports from ticket-index-only CSVs.
  - [ ] Guidance says private/internal notes are ignored for customer wording
        and must not become customer-facing examples or answer proof.
  - [ ] Guidance names the active parse boundary: atlas-portfolio uploads raw
        bytes, ATLAS parses/admissions the CSV/full-thread source.
  - [ ] Existing provider export smoke tests continue to pass and remain
        enrolled in extracted CI.
- Affected surfaces: docs, provider fixture contract, parser/operator guidance.
- Risk areas: misleading buyer/operator instructions, product promise drift,
  stale documentation.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/examples/support_ticket_provider_exports/README.md`
- `plans/PR-Deflection-Provider-Export-Intake-Guidance.md`

## Mechanism

The change updates the packaged support-ticket provider-export documentation
where the fixture contract lives, then links the higher-level extracted package
README to those fixtures. The tests are unchanged because the behavior already
exists:

- `test_support_ticket_package_smoke_accepts_provider_full_thread_exports`
  proves Zendesk/Freshdesk/Help Scout/Intercom full-thread shaped CSVs map
  customer wording and resolution evidence.
- `test_provider_full_thread_exports_generate_publishable_deflection_items`
  proves those fixtures generate resolution-evidence FAQ items.
- `test_support_ticket_package_smoke_marks_ticket_index_only_export_gap_list_only`
  proves a Zendesk ticket-index-only CSV does not create publishable-answer
  evidence.
- `test_support_ticket_package_uses_zendesk_public_comments_not_internal_notes`
  and `test_support_ticket_package_skips_private_comment_objects_in_history`
  prove internal/private notes do not leak into customer wording.

## Intentional

- No parser alias or admission-code change: the current code already handles
  the #1384 residual cases. This slice writes the contract down rather than
  perturbing working parser logic.
- No atlas-portfolio UI copy change in this repo. The live buyer page lives in
  `canfieldjuan/atlas-portfolio`; this PR records the ATLAS-side contract that
  the buyer UI should mirror.
- Fixture files remain synthetic/sanitized. Real customer/provider exports
  should not be committed unless explicitly sanitized and approved.

## Deferred

- atlas-portfolio intake copy can mirror this contract in a buyer-facing UI
  slice if the operator wants the upload page to name full-thread export
  requirements directly.
- Provider-specific live export instructions beyond the generic fixture
  contract remain product copy, not parser behavior.

Parked hardening: none.

## Verification

- pytest tests/test_smoke_content_ops_support_ticket_package.py -q -- 20 passed.
- python scripts/sync_pr_plan.py plans/PR-Deflection-Provider-Export-Intake-Guidance.md -- updated files/diff table from the actual diff.
- bash scripts/run_extracted_pipeline_checks.sh -- 4659 passed, 10 skipped, 1 warning.
- Pending before push: bash scripts/local_pr_review.sh through scripts/push_pr.sh

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/README.md` | 10 |
| `extracted_content_pipeline/examples/support_ticket_provider_exports/README.md` | 34 |
| `plans/PR-Deflection-Provider-Export-Intake-Guidance.md` | 108 |
| **Total** | **152** |
