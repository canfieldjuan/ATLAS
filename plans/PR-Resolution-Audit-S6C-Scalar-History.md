# PR-Resolution-Audit-S6C-Scalar-History

## Why this slice exists

Issue #2044 splits scalar-transcript cleanup from paused evidence PR #2037. `_comments_text()` compacts all three scalar-history aliases as single comments, erasing the boundaries needed to exclude signatures and old replies without losing later follow-ups. This hardening slice fixes that root at the input-admission seam.

### Problem-derived contract

- Root cause: one-message compaction erased transcript structure; the generated oracle still tested sender and admission rules on only one matching production, allowing parallel timestamped-pronoun and explicit-evidence paths to receive the wrong semantic verdict.
- Correct fix: keep bounded reply/signature grammars and one skip policy, while applying pronoun, explicit-evidence, and bounded identity-line rules across every history alias and matching production; narrow grammar decisions to those semantic roles.
- Proof: exercise `build_support_ticket_input_package` with the full generated parity/semantic cross-product plus titled/untitled all-quote admission outcomes.
- Must not change: ordinary bodies/comments, structured-comment privacy, S6B/S6E/S6D semantics, resolution evidence, or customer-facing product shape.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 3

1. Add the bounded scalar-history state machine and real-entrypoint composition/junk regressions at the input-package choke point, treating paused #2037 as read-only evidence.

### Review Contract

- Acceptance: headings/contact-first details survive; every reply-sender production requires real person evidence; explicit evidence keeps quote-empty rows; bounded company lines compose only with signature evidence; ordinary comments retain behavior.
- Affected surfaces/risks: scalar-history normalization/tests only; false resume publishes old/footer text and false skip loses customer text.
- Reviewer rules triggered: R1, R2, R10, R13, R14 with a boundary probe; reachability uses package-entrypoint emitted text or junk-gate results.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S6C-Scalar-History.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

Before S6B extraction, the sanitizer marks Unicode/HTML blanks. A reply grammar requires date plus email or non-pronoun time/name and delimiter/multi-token-person evidence. A signature grammar scans bounded identity lines for contact/company evidence and feeds one skip policy. Hygiene-empty scalar history clears title only when no explicit evidence keeps the row.

## Intentional

- Structured comment containers retain their privacy/text path; no shared hygiene framework or customer-facing shape change is added.
- `On ... wrote:` needs date plus grammar-recognized non-pronoun sender evidence and is terminal. Contact-first details and role/company lines never prove a signature alone; bounded person plus terminal evidence must confirm it.

## Deferred

- #2050 owns status/outcome synonyms; #2045 owns evidence tiers; #2037 stays paused and must not be revived or merged.

Parked hardening: none.

## Verification

- Commands: focused scalar-history pytest — 2 passed/244 deselected with 4,727 generated parity/semantic/admission cases; owning file — 246 passed; exact maturity ratchet — passed.
- Commands: validate_extracted_content_pipeline.sh — passed; forbid/audit-standalone/ASCII audits — clean; CI enrollment — 201 tests enrolled.
- Command: bash scripts/run_extracted_pipeline_checks.sh — 10,732 passed, 21 skipped; one pre-existing pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 219 |
| `plans/PR-Resolution-Audit-S6C-Scalar-History.md` | 62 |
| `tests/test_extracted_support_ticket_input_package.py` | 118 |
| **Total** | **399** |
