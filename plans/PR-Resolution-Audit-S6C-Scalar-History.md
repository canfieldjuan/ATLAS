# PR-Resolution-Audit-S6C-Scalar-History

## Why this slice exists

Issue #2044 splits scalar-transcript cleanup from paused evidence PR #2037. `_comments_text()` compacts all three scalar-history aliases as single comments, erasing the boundaries needed to exclude signatures and old replies without losing later follow-ups. The investigation also proved the current PR implementation made quote mode terminal and tried to infer people from an open-ended sender vocabulary. Both contradict the accepted issue: quote/signature runs must end at blank or likely new-message boundaries, and ambiguous customer prose must remain content. This hardening slice fixes that root at the input-admission seam.

### Problem-derived contract

- Root cause: one-message compaction erased transcript boundaries, while the first state machine substituted open-ended sender-name inference and terminal quote skipping for those missing boundaries. That creates two failure classes: arbitrary title-cased prose can be discarded as a reply, and later customer messages can never resume after a recognized quote.
- Correct fix: canonicalize each scalar-history line once for detection while retaining its original content form; derive one structural event (blank, corroborated reply header, corroborated signature, likely customer-message boundary, or ordinary text); and apply the same explicit state transition table to every scalar-history alias. Reply admission requires affirmative email/original-message/quote-marker evidence, not guessed person identity. Quote and signature states both end at a blank or likely customer-message boundary.
- Safety bias: ambiguous free text is customer content by default. More old-text leakage is acceptable where reply evidence is absent; customer-content loss is not. A Unicode name alone therefore does not prove a reply, while a quote-prefixed Unicode reply header does.
- Proof: exercise `build_support_ticket_input_package` with a grammar-derived event/state cross-product, including quote-marker depth, Unicode sender text, blank and no-blank resumptions, and titled/untitled all-quote admission outcomes.
- Must not change: ordinary bodies/comments, structured-comment privacy, S6B/S6E/S6D semantics, resolution evidence, or customer-facing product shape.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 3

1. Replace sender-vocabulary inference with one bounded structural event/state model at the scalar-history input-package choke point, treating paused #2037 as read-only evidence.

### Review Contract

- Acceptance: quoted and signature runs resume after blank or structural customer boundaries; reply headers need affirmative structural evidence; `>` markers are detection-only and standalone customer `>` lines remain content; original-message blocks accept `Sent:` or `Date:`; ordinary comments retain behavior.
- Affected surfaces/risks: scalar-history normalization/tests only; false resume publishes old/footer text and false skip loses customer text.
- Reviewer rules triggered: R1, R2, R10, R13, R14 with a boundary probe; reachability uses package-entrypoint emitted text or junk-gate results.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S6C-Scalar-History.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

Before S6B extraction, the sanitizer preserves blank events and computes a detection probe with leading quote markers removed. A single event classifier recognizes corroborated reply headers, corroborated signature starts, and structural customer-message boundaries. One state table then emits ordinary content, skips quote/signature runs, and resumes either run on blank or customer-message events. Original-message headers accept `Sent:`/`Date:` parity; `On ... wrote:` requires a date plus email or quote-marker evidence.

## Intentional

- Structured comment containers retain their privacy/text path; no shared hygiene framework or customer-facing shape change is added.
- Customer-content preservation wins ties: uncorroborated Unicode or ASCII name-shaped `On ... wrote:` text remains content instead of invoking open-ended person inference. Quote markers are stripped only for header detection, never from emitted customer text.
- Signature delimiters require bounded contact evidence; first-name, role, and company lines do not prove a signature alone. Mobile signatures remain exact structural markers, but both signature forms resume on the same blank/customer-boundary policy as quotes.

## Deferred

- #2050 owns status/outcome synonyms; #2045 owns evidence tiers; #2037 stays paused and must not be revived or merged.

Parked hardening: none.

## Verification

- Commands: focused scalar-history pytest — 3 passed/243 deselected with 128 generated event/state cases; owning file — 246 passed; exact extracted-content maturity ratchet — passed at the pre-PR file score of 9.
- Commands: validate_extracted_content_pipeline.sh — passed; forbid/audit-standalone/ASCII audits — clean; CI-enrollment audit passed inside the package gauntlet.
- Command: bash scripts/run_extracted_pipeline_checks.sh — 10,732 passed, 21 skipped; one pre-existing pynvml warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 193 |
| `plans/PR-Resolution-Audit-S6C-Scalar-History.md` | 64 |
| `tests/test_extracted_support_ticket_input_package.py` | 130 |
| **Total** | **387** |
