# PR-Resolution-Audit-S6C-Scalar-History

## Why this slice exists

Issue #2044 splits scalar-transcript cleanup from paused evidence PR #2037. `_comments_text()` compacts all three scalar-history aliases as single comments, erasing the boundaries needed to exclude signatures and old replies without losing later follow-ups. This hardening slice fixes that root at the input-admission seam.

### Problem-derived contract

- Root cause: one-message compaction runs before a multi-message scalar history can distinguish customer text from signatures, footers, and quoted history.
- Correct fix: sanitize only the three scalar-history aliases; preserve plain/HTML blanks; recognize bounded signature, mobile, `On ... wrote:`, and Outlook separators; use mode-specific resume evidence; feed one line-preserved result to the junk gate and compact only emitted output.
- Proof: exercise `build_support_ticket_input_package` with positive, negative, composed, and held-out transcript shapes.
- Must not change: ordinary bodies/comments, structured-comment privacy, S6B/S6E/S6D semantics, resolution evidence, or customer-facing product shape.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 3

1. Add the bounded scalar-history state machine at the input-package choke point.
2. Add real-entrypoint boundary-composition and junk-admission regressions.
3. Treat paused #2037 as read-only evidence and build from current main.

### Review Contract

- Acceptance: later customer messages survive; signature/footer and old-reply runs do not; false `wrote:` near-matches remain; ordinary bodies and non-history comments retain behavior.
- Affected surfaces: scalar-history normalization and its package tests only.
- Risks: false resume publishes old/footer text; false skip loses customer text.
- Reviewer rules triggered: R1, R2, R10, R13, R14, including a boundary probe.
- Reachability: package-entrypoint tests assert emitted `source_material` text or the junk-gate result.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S6C-Scalar-History.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

Before S6B extraction, the sanitizer marks text/HTML blanks. A two-mode scanner skips signatures or quoted history: quote mode resumes only on explicit/first-person customer evidence; signature mode also admits a bounded failure cue after a blank. It returns lines for S6E, then final assembly compacts once.

## Intentional

- Structured comment containers keep their existing privacy/text path.
- `On ... wrote:` needs date/time/email evidence; Outlook uses its anchored separator; arbitrary `wrote:` prose is not a boundary.
- Quote blanks prove nothing; signature blanks are only one part of the stronger resume rule.
- No shared hygiene framework or customer-facing output-shape change is added.

## Deferred

- #2050 owns status/outcome synonyms; #2045 owns evidence-tier semantics.
- #2037 stays paused and must not be revived or merged.

Parked hardening: none.

## Verification

- Command: pytest -q tests/test_extracted_support_ticket_input_package.py -k 's6c_scalar_history' — 15 passed, 244 deselected.
- Command: pytest -q tests/test_extracted_support_ticket_input_package.py — 259 passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh — passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — clean.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt — zero findings.
- Command: bash scripts/check_ascii_python.sh — passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py — 201 tests enrolled.
- Command: bash scripts/run_extracted_pipeline_checks.sh — 10,745 passed, 21 skipped; one pre-existing pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 140 |
| `plans/PR-Resolution-Audit-S6C-Scalar-History.md` | 74 |
| `tests/test_extracted_support_ticket_input_package.py` | 178 |
| **Total** | **392** |
