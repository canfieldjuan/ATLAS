# PR-Resolution-Audit-S6C-Scalar-History

## Why this slice exists

Issue #2044 splits scalar-transcript cleanup from paused evidence PR #2037. `_comments_text()` compacts all three scalar-history aliases as single comments, erasing the boundaries needed to exclude signatures and old replies without losing later follow-ups. This hardening slice fixes that root at the input-admission seam.

### Problem-derived contract

- Root cause: one-message compaction erased transcript structure; the first generated oracle closed under-exclusion but under-modeled over-exclusion roles, allowing pronoun senders, contact-first details, legal footers, help requests, and titled hygiene-empty rows to receive the wrong semantic verdict.
- Correct fix: keep bounded reply/signature grammars and one skip policy, while extending the independent oracle across sender arity, contact-first detail, operational continuation versus footer, and hygiene-empty title variants; narrow grammar decisions to those semantic roles.
- Proof: exercise `build_support_ticket_input_package` with the full generated parity/semantic cross-product plus titled/untitled all-quote admission outcomes.
- Must not change: ordinary bodies/comments, structured-comment privacy, S6B/S6E/S6D semantics, resolution evidence, or customer-facing product shape.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 3

1. Add the bounded scalar-history state machine and real-entrypoint composition/junk regressions at the input-package choke point, treating paused #2037 as read-only evidence.

### Review Contract

- Acceptance: headings/contact-first details survive; reply senders require real person evidence; operational help/failure resumes while legal footer prose stays skipped; signature sequences compose across blanks/roles; titled and untitled all-quote rows count as no-new-content; ordinary comments retain behavior.
- Affected surfaces/risks: scalar-history normalization/tests only; false resume publishes old/footer text and false skip loses customer text.
- Reviewer rules triggered: R1, R2, R10, R13, R14 with a boundary probe; reachability uses package-entrypoint emitted text or junk-gate results.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S6C-Scalar-History.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

Before S6B extraction, the sanitizer marks Unicode/HTML blanks. A reply grammar requires date plus email, time/name, or delimiter/multi-token-person evidence. A signature grammar requires person context before contact/company evidence, ignores identity blanks, and feeds one skip policy whose continuation roles separate operational requests/failures from footer prose. Hygiene-empty scalar history clears title only for junk admission.

## Intentional

- Structured comment containers retain their privacy/text path; no shared hygiene framework or customer-facing shape change is added.
- `On ... wrote:` needs date plus grammar-recognized sender evidence and is terminal; pronoun/single-token prose remains content. Contact-first details and role lines never prove a signature alone; bounded person plus terminal evidence must confirm it.

## Deferred

- #2050 owns status/outcome synonyms; #2045 owns evidence tiers; #2037 stays paused and must not be revived or merged.

Parked hardening: none.

## Verification

- Commands: focused scalar-history pytest — 2 passed/244 deselected with 3,521 generated parity/semantic/admission cases; owning file — 246 passed; exact maturity ratchet — passed.
- Commands: validate_extracted_content_pipeline.sh — passed; forbid/audit-standalone/ASCII audits — clean; CI enrollment — 201 tests enrolled.
- Command: bash scripts/run_extracted_pipeline_checks.sh — 10,732 passed, 21 skipped; one pre-existing pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 213 |
| `plans/PR-Resolution-Audit-S6C-Scalar-History.md` | 62 |
| `tests/test_extracted_support_ticket_input_package.py` | 122 |
| **Total** | **397** |
