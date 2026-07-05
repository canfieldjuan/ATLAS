# Resolution Audit CSV Current-Code Remediation Arc

Issue: https://github.com/canfieldjuan/ATLAS/issues/1993

Status: tracking source of truth as of 2026-07-05. This doc supersedes the
older issue-body finding dump for implementation planning. The older audit
files remain evidence/history; code remains ground truth.

## Ground Rules

- Do not accept any audit finding as given. Reconstruct behavior from current
  code before coding.
- Treat stale issue paths as stale. Current report/submit code lives mostly in
  `extracted_content_pipeline/`, with delivery/PDF surfaces in `atlas_brain/`.
- No user-facing report, snapshot, landing, email, or PDF shape changes without
  explicit operator approval first.
- Vertical-first: fix the smallest root that proves real behavior. Park
  non-blocking hardening instead of widening a slice.
- Every remediation PR must update this doc and link the fixing PR.

## Gaps In The Existing Issue Body

- Several cited paths are stale. `atlas_brain/faq_deflection_report.py` and
  `atlas_brain/api/control_surfaces.py` are not current paths; the live code is
  in `extracted_content_pipeline/faq_deflection_report.py` and
  `extracted_content_pipeline/api/control_surfaces.py`.
- F7 is narrower than the issue wording. Per-call `embedding_port` is discarded
  by `FAQDeflectionReportService.generate(..., **kwargs)`, but service-level
  embedding injection exists through `_content_ops_services._build_ticket_faq_service`.
- P5 "no memoization anywhere" is too broad. `campaign_source_adapters` has
  `_SourceFieldLookup`; the still-live problem is the support-ticket and FAQ
  markdown paths that repeatedly normalize row keys.
- M6 "no PII stripping" is false for final report output because the report
  scrubber handles common identifiers. The live gap is input conversation
  hygiene: signatures, quote chains, junk auto-replies, and embedded NULs.
- P5-7 "QA scorecard has zero runtime callers" is stale. QA scorecard runners
  and workflow enrollment now exist; any remaining concern must be framed as
  live-observer quality, not absence.
- C3 is precise, not universal. Ambiguous day/month dates can silently
  transpose under the US parser; day-first values with day > 12 become
  unparseable and disable the date window.

## Confirmed Current-Code Fix Queue

### S1 - CSV Admission And Data Loss

Status: open.

Root: CSV header admission accepts weak fallbacks and row-shape failures are
handled inconsistently.

Confirmed components:

- Missing/headerless CSV can treat the first data row as fallback header.
- Duplicate header columns overwrite earlier values with last-wins semantics.
- One over-wide row raises for the whole file.
- Blank rows do not count in `source_row_count`; likely acceptable if
  documented.
- Legacy encoding fallback can accept clean CP1252/Latin-1 without an explicit
  admission warning.

Expected fix:

- Add header-confidence admission with warnings/errors for low-confidence
  headers.
- Detect duplicate headers and fail or disambiguate explicitly.
- Decide over-wide policy once: reject with clear row detail or quarantine rows
  with an explicit warning.
- Add fixture tests for headerless, duplicate header, over-wide, short-row, and
  legacy-encoding cases.

### S2 - Submit Row Cap And Heavy Inline Build

Status: open.

Root: submit limits conflate bytes and rows, then synchronous report build can
run over very large input.

Confirmed components:

- `_MAX_DEFLECTION_SUBMIT_ROWS` equals the 50 MiB byte cap.
- Missing `limit` returns raw row count.
- Upload/blob loaders stage to temp files before parsing; memory multiplier
  should be profiled before repeating the older numeric claim.

Expected fix:

- Define an explicit row cap separate from bytes.
- Keep the cap visible in diagnostics.
- Move or gate heavy report work behind the existing async job pattern if the
  cap still allows slow runs.

### S3 - Money Formatting Unification

Status: open.

Root: money rendering has multiple helpers with different rounding semantics.

Confirmed components:

- Markdown/model formatting uses half-up integer dollars.
- PDF formatting uses half-up integer dollars.
- Email formatting uses Python `:.0f`, which is banker/half-even rounding.

Expected fix:

- One shared money display helper or one shared rounding policy.
- Tests that render the same value through report model, PDF, and email paths.
- Keep the `$13.50 assisted-contact basis` label single-sourced.

### S4 - Clustering Correctness Spike And First Fix

Status: open.

Root: token-set labels are promoted into hard partitions before question-level
similarity has a chance to unify or split intents.

Confirmed components:

- Same-intent tickets can fragment across topic/evidence partitions.
- Distinct surface-similar intents can over-merge through overlap/shared-anchor
  logic.
- Embedding rescue currently only considers singleton components.
- Representative question choice and some output ordering depend on row order.

Expected fix:

- Run a calibration spike before changing thresholds.
- Avoid single-link semantic merge behavior; prove same-intent and
  distinct-intent fixtures.
- Include order-shuffle tests.
- Keep user-facing grouping/report shape unchanged until approved.

### S5 - Text Chokepoint Hygiene

Status: open.

Root: input normalization preserves useful customer language but does not strip
conversation artifacts before clustering.

Confirmed components:

- Subject/body/comments are concatenated.
- HTML is compacted to text, but signatures, quoted chains, junk auto-replies,
  and low-ratio embedded NULs are not governed by one explicit chokepoint.
- The final report scrubber exists, so this is not a total PII-output absence.

Expected fix:

- Add one ticket-text hygiene chokepoint before clustering.
- Test junk auto-reply, signature, quoted thread, NUL, and near-miss legitimate
  customer wording.
- Keep final-output scrub tests intact.

### S6 - Date Parsing And Date Window Policy

Status: open.

Root: date parsing is ISO + US-only, and the source date window fails all-or
nothing when any included row lacks a parseable date.

Confirmed components:

- Ambiguous non-US dates can silently transpose.
- Day-first dates with day > 12 become unparseable.
- One unparseable/missing date can disable the full date window.
- Dateless annualized run-rate stores `repeat_ticket_count * 12`; prose now
  says "if monthly pace," but the data field is still a strong assumption.

Expected fix:

- Decide locale/day-first policy explicitly.
- Add diagnostics for ambiguous vs unparseable dates.
- Decide whether partial date windows can be used with coverage confidence.
- Rename or separate dateless run-rate fields if needed; user-facing wording
  requires operator approval.

## Product-Surface Follow-Ups Requiring Operator Approval

These are not coding-start items until the operator approves the product shape:

- Whether SEO targets should appear before ranked opportunities in the paid
  report.
- Whether priority fix queue/top unresolved repeats should appear in markdown
  prose or remain structured-only for web/PDF/email consumers.
- Whether snapshot locked rows should expose cost/topic metadata.
- Whether annualized/dateless run-rate should remain in the free snapshot or
  be withheld/renamed.

## Closed Or Reframed Claims

- P5-7 zero runtime callers: closed as stale; runner/workflow callers exist.
- P3 subclusterer linear positive: keep as observation, not remediation.
- L13 short-row truncation: keep as safe observation unless S1 changes row
  policy.
- F7 per-call embedding discard: keep open only if a current caller actually
  needs per-call override; otherwise document service-level injection as the
  supported path.

## Tracking Checklist

- [ ] S1 CSV admission/data-loss PR linked here.
- [ ] S2 submit row cap/heavy-build PR linked here.
- [ ] S3 money helper PR linked here.
- [ ] S4 clustering spike PR linked here.
- [ ] S4 clustering first implementation PR linked here.
- [ ] S5 text chokepoint hygiene PR linked here.
- [ ] S6 date/window policy PR linked here.
- [ ] Operator-approved product-surface issue/PR linked here, if any.
