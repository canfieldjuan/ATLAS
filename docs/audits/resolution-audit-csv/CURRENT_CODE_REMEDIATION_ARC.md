# Resolution Audit CSV Current-Code Remediation Arc

Issue: https://github.com/canfieldjuan/ATLAS/issues/1993

Status: living, non-exhaustive tracking source as of 2026-07-05. This doc
supersedes the older issue-body finding dump for implementation planning. The
older audit files remain evidence/history; code remains ground truth.

## Ground Rules

- Do not accept any audit finding as given. Reconstruct behavior from current
  code before coding.
- Treat stale issue paths as stale. Current report/submit code lives mostly in
  `extracted_content_pipeline/`, with delivery/PDF surfaces in `atlas_brain/`.
- No user-facing report, snapshot, landing, email, or PDF shape changes without
  explicit operator approval first.
- Vertical-first: fix the smallest root that proves real behavior. Park
  non-blocking hardening instead of widening a slice.
- Do not perfect this ledger before shipping fixes. Each remediation slice must
  reconstruct the item it touches and update this doc with what current code
  proves.
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
  markdown paths that repeatedly normalize row keys. That live problem remains
  queued below as its own fix.
- M6 "no PII stripping" is over-broad for scrubbed submit/store paths, but the
  final-output boundary is uneven. The live gaps are input conversation hygiene,
  comment privacy admission, and CLI/customer-facing output paths that can write
  generated markdown, summary, or evidence without the submit scrub.
- P5-7 needs a narrower correction, not retirement. QA scorecard tests,
  standalone runners, and workflow enrollment exist, but the scorecard is not
  wired into product/runtime report generation. Keep the runtime/load-bearing
  wiring gap open.
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
- Rows without stable IDs fall back to `ticket-{row_index}`, so duplicate rows
  with missing/changing IDs can inflate repeat volume and cost.

Expected fix:

- Add header-confidence admission with warnings/errors for low-confidence
  headers.
- Detect duplicate headers and fail or disambiguate explicitly.
- Decide over-wide policy once: reject with clear row detail or quarantine rows
  with an explicit warning.
- Decide missing-ID dedup identity or diagnostics before cost/repeat counts are
  trusted for those rows.
- Add fixture tests for headerless, duplicate header, over-wide, short-row, and
  legacy-encoding cases.

### S2 - Row-Key Normalization Cache

Status: complete. Fixing PR:
[#2029](https://github.com/canfieldjuan/ATLAS/pull/2029).

Root: support-ticket and FAQ markdown paths repeatedly normalize row keys while
scanning every lookup, even though a cached field-lookup pattern already exists
elsewhere.

Confirmed components:

- `campaign_source_adapters._SourceFieldLookup` proves the cached pattern
  exists.
- `support_ticket_input_package._first_value` still normalizes raw row keys
  during each lookup.
- `ticket_faq_markdown._field_value` still scans and normalizes raw row keys
  during each lookup.

Expected fix:

- Reuse or mirror the cached per-row lookup pattern in the support-ticket and
  FAQ markdown hot paths.
- Add parity fixtures proving output does not change.
- Add a focused runtime/perf guard or benchmark artifact before claiming the
  cache removed the hot path.

Implementation notes:

- Support-ticket input package now carries a per-row lookup wrapper through the
  normalization hot path while keeping ordinary `row.get(...)` exact.
- FAQ markdown now wraps each opportunity/evidence row once before the helper
  graph calls `_field_value`.
- Focused tests assert cached-vs-raw lookup parity and that repeated cached
  lookups do not rescan the raw mapping.

### S3 - Submit Row Cap And Heavy Inline Build

Status: complete. Fixing PR:
[#2030](https://github.com/canfieldjuan/ATLAS/pull/2030).

Root: submit limits conflate bytes and rows, then synchronous report build can
run over very large input.

Confirmed components:

- `_MAX_DEFLECTION_SUBMIT_ROWS` equals the 50 MiB byte cap.
- Missing `limit` returns raw row count.
- Full-thread JSON upload/blob loaders parse without a `max_rows` bound before
  later slicing, so `limit` does not cap staging/parsing work.
- Upload/blob loaders stage to temp files before parsing; memory multiplier
  should be profiled before repeating the older numeric claim.

Expected fix:

- Define an explicit row cap separate from bytes.
- Apply the cap to CSV and full-thread JSON importer paths.
- Keep the cap visible in diagnostics.
- Move or gate heavy report work behind the existing async job pattern if the
  cap still allows slow runs.

Implementation notes:

- This slice derives the sync cap from the existing 1,000-row execute/source
  material contract so sync submit cannot reach the known heavy inline build
  path. Full-volume/background submit remains a separate large-upload slice.
- #2030 merged and applied the cap to CSV, upload/blob, and Zendesk full-thread
  source-entry paths while keeping truncation diagnostics honest.

### S4 - Money Reconciliation Contract

Status: in review. Fixing PR:
[#2031](https://github.com/canfieldjuan/ATLAS/pull/2031).

Root: paid artifacts compute and render money through multiple paths, so helper
rounding, row totals, headline totals, and baked money strings can contradict
each other.

Confirmed components:

- Markdown/model formatting uses half-up integer dollars.
- PDF formatting uses half-up integer dollars.
- Email formatting uses Python `:.0f`, which is banker/half-even rounding.
- Rounded row totals can diverge from rounded headline totals even if helpers
  share a rounding mode.
- Money strings baked into model summaries can drift from raw/cents totals.

Expected fix:

- Define one money reconciliation contract: raw/cents values are authoritative,
  display helpers are shared, and rounded rows reconcile to the headline by an
  explicit rule.
- Remove or isolate baked money strings from model summaries where they can
  drift.
- Tests that render the same values through report model, PDF, and email paths,
  including row-total-vs-headline reconciliation.
- Keep the `$13.50 assisted-contact basis` label single-sourced.

Implementation notes:

- This slice keeps the report model shape unchanged and fixes the current
  display defect by routing support-cost displays through one cents renderer.
  Stored raw numeric fields remain numeric; adding stored cents fields remains
  approval-gated because it would change hosted payload shape.

### S5 - Clustering Correctness Spike And First Fix

Status: open.

Root: token-set labels are promoted into hard partitions before question-level
similarity has a chance to unify or split intents.

Confirmed components:

- Same-intent tickets can fragment across topic/evidence partitions.
- Distinct surface-similar intents can over-merge through overlap/shared-anchor
  logic.
- Embedding rescue currently only considers singleton components.
- Representative question choice and some output ordering depend on row order.
- Large token-set uploads above 2,000 rows deliberately skip token-set
  clustering and leave those rows uncategorized, with only a warning/metadata
  signal.

Expected fix:

- Run a calibration spike before changing thresholds.
- Avoid single-link semantic merge behavior; prove same-intent and
  distinct-intent fixtures.
- Include the large-upload token-set skip in the spike, or ensure S3 caps
  full-submit rows below that skip threshold.
- Include order-shuffle tests.
- Keep user-facing grouping/report shape unchanged until approved.

### S6 - Text, Comment, And Outcome Evidence Hygiene

Status: open.

Root: input normalization preserves useful customer language, but text/comment
and outcome evidence are admitted through narrow local rules instead of one
explicit hygiene boundary.

Confirmed components:

- Subject/body/comments are concatenated.
- HTML is compacted to text, but signatures, quoted chains, junk auto-replies,
  and low-ratio embedded NULs are not governed by one explicit chokepoint.
- `_comment_text` only skips comments where `public is False`; private/internal
  markers such as `is_private`, `is_internal`, or string `"false"` can still
  flow into ticket text.
- Status values outside the small resolved/open/reopened/cancelled sets
  normalize to `other`, so resolved outcome evidence can be undercounted.
- The final report scrubber exists for scrubbed paths, but CLI/customer-facing
  markdown, summary, and evidence outputs still need boundary review.

Expected fix:

- Add one ticket-text hygiene chokepoint before clustering.
- Test junk auto-reply, signature, quoted thread, NUL, and near-miss legitimate
  customer wording.
- Add private/internal comment fixtures and status synonym fixtures.
- Keep final-output scrub tests intact.

### S7 - Date Parsing And Date Window Policy

Status: open.

Root: date parsing is ISO + US-only, and the source date window fails all-or
nothing when any included row lacks a parseable date.

Confirmed components:

- Ambiguous non-US dates can silently transpose.
- Day-first dates with day > 12 become unparseable.
- One unparseable/missing date can disable the full date window.
- Dateless annualized run-rate stores `repeat_ticket_count * 12`; prose now
  says "if monthly pace," but the hosted report-model/consumer payload still
  exposes annualized fields as a strong assumption.

Expected fix:

- Decide locale/day-first policy explicitly.
- Add diagnostics for ambiguous vs unparseable dates.
- Decide whether partial date windows can be used with coverage confidence.
- Rename or separate dateless run-rate fields if needed; user-facing wording
  requires operator approval.

### S8 - Runtime QA Scorecard Wiring

Status: open.

Root: QA scorecard logic is exercised by tests and standalone scripts, but it is
not load-bearing in product/runtime report generation.

Confirmed components:

- `build_deflection_full_report_qa_scorecard` has test callers and deterministic
  harness callers.
- `build_pdf_export_scorecard` has standalone script/test callers.
- No product/runtime report generation path currently invokes the scorecard as a
  guard before delivery or persistence.

Expected fix:

- Decide the runtime boundary: generation, storage, delivery, or release gate.
- Wire the scorecard where a generated paid report can actually be blocked or
  flagged.
- Add an end-to-end test that proves a bad report artifact trips the runtime
  guard, not just a standalone checker.

## Product-Surface Follow-Ups Requiring Operator Approval

These are not coding-start items until the operator approves the product shape:

- Whether SEO targets should appear before ranked opportunities in the paid
  report.
- Whether priority fix queue/top unresolved repeats should appear in markdown
  prose or remain structured-only for web/PDF/email consumers.
- Whether snapshot locked rows should expose cost/topic metadata.
- Whether annualized/dateless run-rate belongs in hosted report-model consumer
  payloads, PDF/email/page delivery, or any future snapshot projection. Current
  free snapshot fields do not expose annualized values.

## Closed Or Reframed Claims

- P5-7 zero runtime callers: reframed, not closed. Standalone runners and tests
  exist, but product/runtime wiring remains open as S8.
- P3 subclusterer linear positive: keep as observation, not remediation.
- L13 short-row truncation: keep as safe observation unless S1 changes row
  policy.
- F7 per-call embedding discard: keep open only if a current caller actually
  needs per-call override; otherwise document service-level injection as the
  supported path.

## Tracking Checklist

- [x] S1 CSV admission/data-loss and missing-ID stability PR linked here: #2026.
- [x] S2 row-key normalization cache PR linked here: #2029.
- [x] S3 submit row cap/heavy-build PR linked here: #2030.
- [ ] S4 money reconciliation PR linked here: #2031.
- [ ] S5 clustering spike PR linked here.
- [ ] S5 clustering first implementation PR linked here.
- [ ] S6 text/comment/outcome hygiene PR linked here.
- [ ] S7 date/window policy PR linked here.
- [ ] S8 runtime QA scorecard wiring PR linked here.
- [ ] Operator-approved product-surface issue/PR linked here, if any.
