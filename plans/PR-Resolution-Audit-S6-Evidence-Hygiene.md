# PR-Resolution-Audit-S6-Evidence-Hygiene

## Why this slice exists

Issue #1993 / `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
keeps S6 open for text, comment, and outcome evidence hygiene. The current code
has useful pieces, but the boundary is split: HTML compaction lives in
`support_ticket_plain_text`, comments only skip exact `public is False`, and
outcome diagnostics inherit incomplete status normalization.

### Problem-derived contract

- Root cause: ticket text is admitted into clustering through several narrow
  local rules instead of one explicit hygiene boundary. That lets privacy
  variants (`is_private`, `is_internal`, string `"false"`) and mechanical
  thread junk depend on the specific export shape, while outcome status
  diagnostics are only as complete as the status synonym set admitted upstream.
- Correct fix must touch/change: add one support-ticket input hygiene chokepoint
  before clustering; route subject/body/comment text through it; make comment
  privacy detection explicit and shared for all comment object variants; expand
  status normalization only for concrete helpdesk synonyms that still map to
  the existing canonical buckets; prove the behavior through the real
  `build_support_ticket_input_package` -> `build_ticket_faq_markdown` path and
  focused support-ticket package fixtures.
- Must not change: no report-model, snapshot, landing, email, PDF, checkout, or
  buyer-visible section/label shape changes; no final-output scrubber grammar
  changes; no clustering algorithm changes beyond cleaner input text; no S7 date
  parsing/window policy; no S8 runtime QA scorecard wiring; no changes to other
  lanes or open PRs.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Vertical slice

1. Normalize support-ticket subject/body/comment text through one explicit
   pre-clustering hygiene helper.
2. Treat private/internal comment markers as private even when exports use
   `is_private`, `is_internal`, `private`, `internal`, `public: "false"`, or
   equivalent visibility/type labels.
3. Add bounded status synonyms that preserve the existing canonical
   `resolved`/`reopened`/`cancelled`/`open`/`other` buckets.
4. Add tests for junk auto-reply/signature/quoted-thread/NUL handling,
   private/internal comment variants, near-miss legitimate customer wording, and
   real FAQ outcome diagnostics.

### Review Contract

- Acceptance criteria: private/internal comments are excluded while markerless
  customer comments stay; mechanical reply junk is removed while near-misses
  remain; status synonyms reach package summaries and real FAQ diagnostics;
  report/snapshot/email/PDF shape does not change.
- Affected surfaces: support-ticket input normalization, FAQ consumption through
  existing normalized rows, S6 tracker docs, and focused tests.
- Risk areas: over-stripping wording, leaking private comments, relabeling
  unknown statuses, or changing report output shape.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R3
  privacy/safety boundary, R7 product-shape consent, R10 fixtures/producers,
  R13 class-fix probes, R14 codebase verification.
- Reachability proof: real package builder -> `build_ticket_faq_markdown` ->
  observable `outcome_diagnostics` and admitted text.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S6-Evidence-Hygiene.md`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`

## Mechanism

Add a support-ticket hygiene layer in `support_ticket_input_package.py`: a text
helper that wraps `support_ticket_plain_text`, strips embedded NULs and obvious
mechanical reply sections, a comment privacy helper for boolean-ish markers and
visibility/type strings, and a bounded status synonym update. The FAQ builder
keeps consuming the same normalized rows; tests prove the row-level status state
flows into `outcome_diagnostics`.

## Intentional

- This PR does not add new report fields or expose new buyer-facing sections;
  S6 is an input hygiene fix, not a product-shape slice.
- Unknown statuses still map to `other`; final PII scrubber grammar is untouched.
- `support_ticket_zendesk_thread._comment_text` is a separate parser helper;
  this PR changes only the support-ticket package admission boundary.

## Deferred

- S7 owns date parsing, partial date-window coverage, and dateless run-rate
  semantics.
- S8 owns runtime QA scorecard wiring.
- Broader customer-facing report/snapshot/email/PDF copy or shape changes remain
  operator-approval gated.

Parked hardening: none.

## Verification

- `pytest tests/test_smoke_content_ops_support_ticket_package.py tests/test_extracted_ticket_faq_markdown.py -q` - 455 passed.
- `scripts/run_extracted_pipeline_checks.sh` - 5166 passed, 21 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 20 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 145 |
| `plans/PR-Resolution-Audit-S6-Evidence-Hygiene.md` | 112 |
| `tests/test_extracted_ticket_faq_markdown.py` | 58 |
| `tests/test_smoke_content_ops_support_ticket_package.py` | 62 |
| **Total** | **397** |
