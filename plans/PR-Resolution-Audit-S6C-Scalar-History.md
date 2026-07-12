# PR-Resolution-Audit-S6C-Scalar-History

## Why this slice exists

Issue #2044 splits scalar-transcript cleanup from paused evidence PR #2037. `_comments_text()` compacts all three scalar-history aliases as single comments, erasing the boundaries needed to exclude signatures and old replies without losing later follow-ups. This hardening slice fixes that root at the input-admission seam.

### Problem-derived contract

- Root cause: one-message compaction erased transcript structure; within the replacement scanner, token-level regex matches still act as semantic line roles without sufficient delimiter, sender, container, or neighboring-line evidence.
- Correct fix: classify Unicode blanks, sender-bearing reply headers, signature-tail shapes, and operational customer cues before mode policy; make quoted tails terminal, preserve ordinary `>`/dash details, and feed one line-preserved result to the junk gate before compacting output.
- Proof: exercise `build_support_ticket_input_package` with positive, negative, composed, and held-out transcript shapes.
- Must not change: ordinary bodies/comments, structured-comment privacy, S6B/S6E/S6D semantics, resolution evidence, or customer-facing product shape.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 3

1. Add the bounded scalar-history state machine and real-entrypoint composition/junk regressions at the input-package choke point, treating paused #2037 as read-only evidence.

### Review Contract

- Acceptance: customer thresholds/details and bounded post-footer requests survive; senderless/header/footer near-matches and every old-reply tail do not; Unicode/HTML blanks normalize; ordinary bodies/comments retain behavior.
- Affected surfaces: scalar-history normalization and its package tests only.
- Risks: false resume publishes old/footer text; false skip loses customer text.
- Reviewer rules triggered: R1, R2, R10, R13, R14, including a boundary probe.
- Reachability: package-entrypoint tests assert emitted `source_material` text or the junk-gate result.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S6C-Scalar-History.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

Before S6B extraction, the sanitizer marks Unicode/HTML blanks. Reply headers require date plus email or time-followed-by-name evidence and enter terminal quote mode. Ambiguous dashes inspect only the immediate tail for name/contact shape, never vocabulary. Signature mode admits bounded requests/failures without accepting legal disclaimers; normal mode preserves `>` lines.

## Intentional

- Structured comment containers retain their privacy/text path; no shared hygiene framework or customer-facing shape change is added.
- `On ... wrote:` needs date plus real sender evidence and is terminal; arbitrary `wrote:` prose, quote labels/blanks/questions prove no new boundary, while signature blanks remain only one part of a customer-shaped resume rule.

## Deferred

- #2050 owns status/outcome synonyms; #2045 owns evidence tiers; #2037 stays paused and must not be revived or merged.

Parked hardening: none.

## Verification

- Command: pytest -q tests/test_extracted_support_ticket_input_package.py -k 's6c_scalar_history' — 16 passed, 244 deselected.
- Command: pytest -q tests/test_extracted_support_ticket_input_package.py — 260 passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh — passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — clean.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt — zero findings.
- Command: bash scripts/check_ascii_python.sh — passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py — 201 tests enrolled.
- Command: bash scripts/run_extracted_pipeline_checks.sh — 10,746 passed, 21 skipped; one pre-existing pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 164 |
| `plans/PR-Resolution-Audit-S6C-Scalar-History.md` | 69 |
| `tests/test_extracted_support_ticket_input_package.py` | 165 |
| **Total** | **398** |
