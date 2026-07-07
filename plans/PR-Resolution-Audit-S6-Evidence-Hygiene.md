# PR-Resolution-Audit-S6-Evidence-Hygiene

## Why this slice exists

Issue #1993 / `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
keeps S6 open for text, comment, and outcome evidence hygiene. The current code
has useful pieces, but the boundary is split: HTML compaction lives in
`support_ticket_plain_text`, the package builder and Zendesk full-thread
flattener used separate privacy checks, and outcome diagnostics inherit
incomplete status normalization.

This slice is over the 400 LOC target because the first-pass package-only fix
missed the same privacy defect in the Zendesk full-thread path. Keeping the fix
root-cause-correct requires a shared helper, manifest ownership, and same-class
tests across both admission paths in one PR.

### Problem-derived contract

- Root cause: ticket text is admitted into clustering through several narrow
  local rules instead of one explicit hygiene boundary. That lets privacy
  variants (`is_private`, `is_internal`, string/decimal booleans, ambiguous
  present markers) and mechanical thread junk depend on the specific export
  shape. Zendesk full-thread comments can be flattened before the package
  boundary, so a package-only privacy fix cannot protect that primary path.
  Outcome status diagnostics are only as complete as the status synonym set
  admitted upstream.
- Correct fix must touch/change: add one shared support-ticket hygiene
  chokepoint; route package subject/body/comment text through it; route Zendesk
  full-thread comment admission through the same private-comment predicate
  before flattening; preserve HTML line breaks before line-junk stripping; keep
  legitimate customer questions about auto-reply/out-of-office features; fail
  closed on present ambiguous private/public markers; expand status
  normalization only for concrete helpdesk synonyms that still map to the
  existing canonical buckets; prove the behavior through the real
  `build_support_ticket_input_package` -> `build_ticket_faq_markdown` path and
  focused package/Zendesk fixtures.
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
   `is_private`, `is_internal`, `private`, `internal`, `public: "false"`,
   decimal booleans, ambiguous present markers, or equivalent visibility/type
   labels.
3. Add bounded status synonyms that preserve the existing canonical
   `resolved`/`reopened`/`cancelled`/`open`/`other` buckets.
4. Add tests for junk auto-reply/signature/quoted-thread/NUL/HTML handling,
   private/internal comment variants, the Zendesk full-thread flattener path,
   near-miss legitimate customer wording, and real FAQ outcome diagnostics.

### Review Contract

- Acceptance criteria: private/internal comments are excluded on both package
  and Zendesk full-thread paths while markerless/customer-public comments stay;
  mechanical reply junk is removed after HTML line breaks are preserved while
  near-misses remain; status synonyms reach package summaries and real FAQ
  diagnostics; report/snapshot/email/PDF shape does not change.
- Affected surfaces: support-ticket input normalization, Zendesk full-thread
  row flattening, FAQ consumption through existing normalized rows, S6 tracker
  docs, and focused tests.
- Risk areas: over-stripping wording, leaking private comments, relabeling
  unknown statuses, or changing report output shape.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R3
  privacy/safety boundary, R7 product-shape consent, R10 fixtures/producers,
  R13 class-fix probes, R14 codebase verification.
- Reachability proof: real package builder -> `build_ticket_faq_markdown` ->
  observable `outcome_diagnostics` and admitted text.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/support_ticket_text_hygiene.py`
- `extracted_content_pipeline/support_ticket_zendesk_thread.py`
- `plans/PR-Resolution-Audit-S6-Evidence-Hygiene.md`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`

## Mechanism

Add a shared support-ticket hygiene layer used by both the support-ticket
package builder and the Zendesk full-thread flattener. The shared layer strips
embedded NULs, preserves HTML line breaks before removing mechanical reply
sections, keeps legitimate customer questions about auto-reply/out-of-office
features, and centralizes private/internal comment detection for boolean-ish,
decimal, visibility, and type markers. Zendesk full-thread rows filter through
that same privacy predicate before flattening comments into customer or
resolution text. The FAQ builder keeps consuming the same normalized rows;
tests prove the row-level status state flows into `outcome_diagnostics`.

## Intentional

- This PR does not add new report fields or expose new buyer-facing sections;
  S6 is an input hygiene fix, not a product-shape slice.
- Unknown statuses still map to `other`; final PII scrubber grammar is untouched.
- Zendesk full-thread parsing remains a flattener; this PR only routes its
  private/internal comment admission through the same shared S6 boundary and
  does not otherwise change role assignment or auto-ack suppression.

## Deferred

- S7 owns date parsing, partial date-window coverage, and dateless run-rate
  semantics.
- S8 owns runtime QA scorecard wiring.
- Broader customer-facing report/snapshot/email/PDF copy or shape changes remain
  operator-approval gated.

Parked hardening: none.

## Verification

- `pytest tests/test_smoke_content_ops_support_ticket_package.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py -q` - 548 passed.
- `scripts/run_extracted_pipeline_checks.sh` - 5167 passed, 21 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 33 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 47 |
| `extracted_content_pipeline/support_ticket_text_hygiene.py` | 182 |
| `extracted_content_pipeline/support_ticket_zendesk_thread.py` | 5 |
| `plans/PR-Resolution-Audit-S6-Evidence-Hygiene.md` | 139 |
| `tests/test_extracted_support_ticket_input_package.py` | 63 |
| `tests/test_extracted_ticket_faq_markdown.py` | 58 |
| `tests/test_smoke_content_ops_support_ticket_package.py` | 75 |
| **Total** | **605** |
