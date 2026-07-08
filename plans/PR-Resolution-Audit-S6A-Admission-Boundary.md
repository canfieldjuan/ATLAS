# PR-Resolution-Audit-S6A-Admission-Boundary

## Why this slice exists

#2042 extracts the privacy/admission boundary from the paused #2037 evidence
branch into a small shipping slice for #1993. Current `origin/main` admits
support-ticket text from two independent paths: generic CSV/package rows and
Zendesk full-thread rows. Both paths only understand the literal Python boolean
shape `public is False` for private comments, and the package row normalizer
reads row title/body/comment text before any row-level private/internal marker
can reject the row. That means string/number privacy markers, private-note
aliases, internal-comment aliases, and ambiguous privacy markers can be
admitted as customer evidence.

This fixes the root, not a symptom: admission is moved to the upstream
normalization boundary shared by both support-ticket ingestion paths before
private/internal source text is flattened into `text` or `resolution_text`.
The diff is over the 400 LOC soft cap because the required plan doc and
same-class tests ship with the shared boundary; the code/test diff excluding the
plan remains narrowly scoped to this one admission concern.

### Problem-derived contract

- Root cause: support-ticket private/internal admission is split across
  package-row normalization, package comments, Zendesk full-thread flattening,
  and private-description duplicate suppression. A package-only predicate
  cannot protect comments already flattened by the Zendesk path, and row-level
  private markers can be read after source text extraction.
- Correct fix must touch/change:
  1. Add one shared support-ticket private/internal predicate with marker
     normalization for boolean strings, decimal booleans, private/internal
     aliases, explicit public labels, and ambiguous markers.
  2. Reject row-level private/internal support-ticket rows before
     `_normalize_ticket_row` extracts title/body/comment text.
  3. Route package comment admission through the shared predicate.
  4. Route Zendesk full-thread comment admission through the same predicate
     before `_row_from_entry` flattens customer/resolution text.
  5. Keep private-first-comment duplicate suppression aligned with the same
     raw text key used for Zendesk `description` comparison.
  6. Add focused same-class tests for boolean strings, decimal booleans,
     private-note aliases, explicit public labels, ambiguous markers,
     row-level private rows, and Zendesk full-thread comments.
- Must not change:
  - HTML/tag line-preserving hygiene or `support_ticket_clustering.py`.
  - Scalar `ticket_history` / `conversation_history` signature or quote
    handling.
  - Evidence-tier/customer-wording semantics beyond excluding private/internal
    source text from admission.
  - Status/outcome diagnostics.
  - Report/snapshot/email/PDF/landing/checkout/product shape.
  - Final-output scrubber grammar.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Vertical slice

1. Add a shared support-ticket privacy/admission predicate and apply it at the
   two current support-ticket ingestion boundaries: generic package rows and
   Zendesk full-thread rows.
2. Add targeted tests that exercise the real row normalizers and prove private
   markers are excluded while public/markerless comments still pass.

### Review Contract

- Acceptance criteria:
  - Private/internal rows and comments are excluded before source text is
    admitted.
  - Markerless customer comments and explicit public labels still pass.
  - Ambiguous privacy markers fail closed.
  - Zendesk private-first-comment duplicate suppression keeps the public
    `description` out when the same private first comment is mirrored there.
  - The proof covers the class with multiple marker variants, not only one
    cited example.
- Affected surfaces: `extracted_content_pipeline` support-ticket CSV/package
  normalization and Zendesk full-thread normalization.
- Risk areas: PII/privacy leakage, false-positive dropping of public customer
  text, and drift between package and Zendesk comment filters.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R3
  privacy/security boundary, R8 contract drift, R10 guard/checker behavior,
  R13 same-class fix, R14 codebase verification.

### Files touched

- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/support_ticket_privacy.py`
- `extracted_content_pipeline/support_ticket_zendesk_thread.py`
- `plans/PR-Resolution-Audit-S6A-Admission-Boundary.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`
- `tests/test_support_ticket_privacy.py`

## Mechanism

The slice adds a small package-owned helper for support-ticket privacy marker
normalization. It exposes row and comment admission wrappers backed by the same
normalizer: public markers with false-like values and private/internal markers
are private, explicit public values are public, and ambiguous privacy markers
fail closed. Strict privacy labels such as `visibility` and `privacy` still
fail closed, while ordinary metadata labels such as `access` and `audience`
classify by value so routing phrases like "Account access" do not drop public
rows. Object-shaped markers normalize through the same label vocabulary.
Restricted value labels and private/internal reply kind labels are treated as
private. Case/punctuation variants of the same marker key are grouped before
classification; conflicting grouped values become ambiguous and fail closed
instead of using last-writer-wins. Package rows call the row wrapper before any
`source_title`, body, or comment extraction. Package comments and Zendesk
comments use the comment wrapper before their text is appended. Zendesk
private-description duplicate suppression still compares raw normalized text
keys, so filtering does not depend on a separate sanitizer shape.

## Intentional

- This slice does not add HTML line-preserving hygiene. That is #2043/S6B; doing
  it here would mix a text-cleanup concern into the privacy/admission boundary.
- This slice does not change scalar history parsing. That is #2044/S6C.
- This slice does not change final report/snapshot/email/PDF or landing shape.
  The operator has not approved a product-shape change for this slice, and the
  privacy fix does not require one.

## Deferred

- #2043/S6B: line-preserving HTML/tag hygiene.
- #2044/S6C: scalar history signature/quote guard.
- #2045/S6D: QA scorecard/runtime integration and reconciliation defects.

Parked hardening: none.

## Verification

- Passed:
  - `python -m pytest tests/test_support_ticket_privacy.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_extracted_support_ticket_input_package.py -q` (`231 passed`)
  - `python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --baseline tests/maturity_sweep/baseline_extracted_content_pipeline.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/webhook*' --sensitive-glob '**/payment*' --sensitive-glob '**/*deletion*'`
  - `python -m pytest tests/test_smoke_content_ops_support_ticket_package.py tests/test_extracted_support_ticket_input_package.py -q` (`150 passed`)
  - `bash` `scripts/validate_extracted_content_pipeline.sh`
  - `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - `bash` `scripts/check_ascii_python.sh`
  - `bash` `scripts/run_extracted_pipeline_checks.sh` (`5197 passed, 21 skipped`)
  - `python scripts/sync_pr_plan.py plans/PR-Resolution-Audit-S6A-Admission-Boundary.md --check`
  - `git diff --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 8 |
| `extracted_content_pipeline/support_ticket_privacy.py` | 231 |
| `extracted_content_pipeline/support_ticket_zendesk_thread.py` | 21 |
| `plans/PR-Resolution-Audit-S6A-Admission-Boundary.md` | 158 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_support_ticket_input_package.py` | 194 |
| `tests/test_smoke_content_ops_support_ticket_package.py` | 119 |
| `tests/test_support_ticket_privacy.py` | 120 |
| **Total** | **855** |
