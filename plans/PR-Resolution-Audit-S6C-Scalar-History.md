# PR-Resolution-Audit-S6C-Scalar-History

## Why this slice exists

Issue #2044 splits scalar transcript cleanup out of paused evidence branch
#2037 after the Resolution Audit CSV run exposed a correctness boundary that
ordinary one-message cleanup cannot represent. Current
`_comments_text()` treats a scalar `ticket_history`, `history`, or
`conversation_history` value as one comment and immediately compacts it.
That erases message boundaries and admits signature footers and old quoted
replies as customer evidence. Stopping at the first signature or quote would
instead discard later public customer follow-ups, so that downstream symptom
fix is also incorrect.

This change fixes the root in the scalar-history admission path: it gives that
path an explicit transcript state machine before compact customer text is
assembled. The recent S6B line-preserving extraction is reused as the upstream
text component; no second HTML parser or downstream report scrubber is added.

### Problem-derived contract

- Root cause: scalar transcript fields contain several messages in one string,
  but the current adapter applies the same one-message compaction used for
  ordinary bodies and comment objects. Once compacted, the adapter cannot tell
  customer follow-ups from signatures or quoted reply history.
- Correct fix must touch/change: only scalar `ticket_history`, `history`, and
  `conversation_history` string handling in
  `extracted_content_pipeline/support_ticket_input_package.py`; preserve
  later customer messages after a blank or recognizable new-message boundary;
  suppress signature/footer runs and quoted-reply bodies with or without `>`;
  require date, time, or email evidence before treating an `On ... wrote:` line
  as a reply header; feed the same cleaned scalar transcript to the junk gate
  and emitted customer text; and prove the behavior through the real
  `build_support_ticket_input_package` entrypoint with the cited cases plus
  held-out transcript compositions.
- Must not change: ordinary body fields; scalar one-message comment aliases;
  list/object comment handling; package or Zendesk private/internal admission;
  S6B HTML tag recognition and line extraction; S6E junk admission;
  S6D customer evidence-tier or status/outcome semantics; resolution evidence;
  final-output scrubber grammar; or report, snapshot, email, PDF, landing,
  checkout, and other product shape.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 3

1. Add a bounded scalar-history transcript sanitizer at the support-ticket
   input-package chokepoint and route only scalar history aliases through it.
2. Add real-entrypoint regression coverage for signature and quoted-reply
   state transitions, including held-out variants that prove the class rather
   than the paused branch's exact examples.
3. Leave the paused #2037 branch as read-only evidence and implement from
   current `origin/main` after S6B/S6E.

### Review Contract

- Acceptance criteria:
  - later customer messages survive after signatures and reply quotes;
  - signature/footer lines remain excluded until a blank or recognizable
    customer-message boundary;
  - prefixed and unprefixed old quoted bodies remain excluded;
  - an ordinary customer sentence containing `wrote:` without date/time/email
    reply metadata is preserved;
  - ordinary body and non-history comment paths retain their current behavior.
- Affected surfaces: extracted support-ticket scalar history normalization and
  its package-level tests only.
- Risk areas: a false resume can publish footer/old-reply text as customer
  evidence; an over-broad boundary can discard a real later customer message.
- Reviewer rules triggered: R1 requirements match, R2 two-sided detector
  evidence, R10 maintainability, R13 held-out class proof, and R14 codebase
  verification with a boundary probe.
- Reachability proof: tests call the real `build_support_ticket_input_package`
  entrypoint and assert its emitted `source_material` text; this is an existing
  adapter behavior, not a new runtime or buyer-visible surface.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S6C-Scalar-History.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The input adapter will recognize only scalar values under the three explicit
history aliases. It will pass those strings through the existing S6B
line-preserving text seam, retaining explicit plain-text blank separators for
transcript state. A small state machine emits admitted lines while in normal
mode and switches to signature-skip or quote-skip mode at anchored boundaries.

Signature mode may resume after a blank or strong customer-message boundary
while bounded legal-footer shapes remain excluded. Quote mode stays closed
across blanks and agent-like sentences; only an explicit customer label or
first-person question/failure follow-up resumes it. Quote-prefixed lines stay
excluded. Reply headers require `On ... wrote:` plus a date, time, or email cue,
so ordinary product prose remains admitted. The same sanitized transcript
feeds junk admission and emitted customer text before final compaction.

## Intentional

- The sanitizer is private to the input package rather than a new shared
  hygiene framework. S6A privacy, S6B HTML extraction, and S6E junk admission
  already own their boundaries; widening those modules would reassemble the
  broad #2037 PR this queue deliberately split.
- Structured comment lists keep their current per-comment privacy and text
  handling because their container already preserves message boundaries.
- The recognizable-message predicate is conservative and two-sided. It admits
  question-shaped or explicit customer follow-ups, not arbitrary post-footer
  prose; explicit blank separation remains the generic escape hatch.
- Quote and signature modes deliberately differ: a blank may end a signature,
  but cannot prove an unprefixed quoted reply has ended.
- No customer-facing copy or output shape changes are authorized or required.

## Deferred

- #2050 owns status/outcome synonyms; #2045 owns evidence-tier semantics after S6C.
- #2037 remains paused evidence and must not be revived or merged.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_extracted_support_ticket_input_package.py -k 's6c_' -q
  - Passed: 12 passed, 243 deselected after exact-head review probes.
- Command: python -m pytest tests/test_extracted_support_ticket_input_package.py -q
  - Passed: 255 passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed: mapped files and hard-import checks clean.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed: clean.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed: zero Atlas runtime import findings.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py
  - Passed: 201 matching tests enrolled.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Passed: 10,741 passed, 21 skipped, 1 pre-existing `pynvml` warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 123 |
| `plans/PR-Resolution-Audit-S6C-Scalar-History.md` | 148 |
| `tests/test_extracted_support_ticket_input_package.py` | 128 |
| **Total** | **399** |
