# PR-Deflection-Outcome-Risk-Diagnostics

## Why this slice exists

Issue #1419/#1507 deliberately separated ingestion from product consumption:
#1510 normalized ticket status / reopened / CSAT fields, and #1535 proved the
Zendesk full-thread path can draft publishable answers from public agent
resolution evidence. The remaining vertical gap is buyer-visible consumption
of the status/CSAT/reopen signals. Today those fields sit in normalized rows and
input-provider metadata, but the paid report does not tell the buyer which
questions were "resolved on paper" but still risky because tickets reopened or
CSAT was negative.

This slice adds the thinnest paid-report diagnostic lane for that signal
without weakening the proven-answer gate: status/CSAT/reopen can flag outcome
risk, but they still cannot create `resolution_evidence` or publishable answer
copy by themselves.

The synced diff estimate is above 400 LOC because it includes this plan doc
and the review-fix probes. The item diagnostics, report rendering, raw-alias
normalization, and negative gate tests are one indivisible vertical slice.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Carry per-question outcome diagnostics from normalized support-ticket rows
   into FAQ items: status summary, reopened count, CSAT-present count, negative
   CSAT count, and numeric CSAT average when available.
2. Normalize direct raw-row status/CSAT aliases (`ticket_status`, `status`,
   `satisfaction_rating`, etc.) through the same semantics as the input package.
3. Count diagnostic totals by distinct source ticket, not evidence row, so one
   ticket with multiple snippets cannot inflate reopened/CSAT counts.
4. Aggregate those diagnostics into the paid report summary.
5. Render a paid-report "Resolution Outcome Diagnostics" section that explains
   the signal as review guidance, not answer proof.
6. Add focused extracted tests covering positive diagnostics, textual Zendesk
   `bad` CSAT, numeric low CSAT, direct raw aliases, duplicate evidence rows,
   and the negative case where diagnostics alone do not turn a
   draft-needs-review item into `resolution_evidence`.
7. Regenerate the committed resolution-evidence proof artifacts so the live
   proof fixture reflects the new paid-report diagnostics section.

### Review Contract

Acceptance criteria:
- Rows with `ticket_status_state="reopened"` or negative CSAT surface as
  outcome-risk diagnostics in FAQ items, report summary, and paid Markdown.
- Textual Zendesk CSAT (`bad`) and numeric low CSAT (`<=2`) both count as
  negative; positive textual/numeric CSAT does not.
- Direct report rows with raw `ticket_status`/`status` and textual/numeric CSAT
  aliases produce the same outcome diagnostics as input-package-normalized rows.
- Duplicate evidence rows with the same `source_id` count as one ticket for
  diagnostic totals, risk totals, status mix, reopened count, and CSAT count.
- The diagnostic lane explicitly says these signals do not prove a publishable
  answer.
- A no-resolution item with only status/CSAT diagnostics remains
  `answer_evidence_status == "draft_needs_review"` and appears in the
  no-proven-answer section, not the publishable-answer section.
- The free snapshot does not expose new answer text/evidence; this PR only
  changes paid report item/summary/Markdown details.

Affected surfaces:
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `tests/test_content_ops_deflection_report.py`

Risk areas:
- Accidentally treating status/CSAT/reopen as resolution evidence.
- Over-capturing CSAT: textual `good` or numeric `4/5` must not become a
  negative signal.
- Row-vs-ticket counting drift when one support ticket contributes multiple
  evidence snippets.
- Direct CSV/script path drift from the input-package normalized path.
- Adding paid-report summary fields that drift from item-level counts.
- Changing the free snapshot contract.

Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json`
- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Outcome-Risk-Diagnostics.md`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

`build_ticket_faq_markdown(...)` already groups support-ticket evidence into
per-question FAQ items. This PR adds a small deterministic helper that inspects
those grouped rows for fields #1510 already emits:

- `ticket_status_state`
- `csat`
- `csat_score`

For the direct report path, the helper also preserves raw status/CSAT aliases
from the opportunity row and normalizes them with the same status/CSAT helpers
used by `support_ticket_input_package.py`.

The helper first collapses diagnostics by `source_key`/`source_id`, then
aggregates counts. That keeps ticket-level numbers coherent when a single
ticket contributes multiple evidence snippets.

When at least one status/CSAT field is present, the helper attaches an
`outcome_diagnostics` dict to the item. The dict is diagnostic only; it is not
consulted by the answer-evidence gate.

`deflection_report_summary(...)` then aggregates item diagnostics into compact
counts, and `render_deflection_report(...)` inserts a paid-only section between
ranked opportunities and publishable answers. The section is skipped when no
diagnostics are present.

## Intentional

- This PR does not invent a universal CSAT scale. Numeric `<=2` is treated as
  negative because it is the common 1-5 low-score band; textual Zendesk `bad`
  is treated as negative; textual `good` only counts as CSAT-present.
- Open/pending status is displayed as context, but the risk counts focus on
  reopened and negative CSAT. Open tickets may be unresolved rather than
  failed resolutions.
- Raw direct-row status/CSAT aliases reuse the input-package normalizers rather
  than maintaining a second set of semantics in the report path.
- The section is paid-report only. The pre-payment snapshot stays focused on
  safe counts and locked questions.

## Deferred

- Tier-2 churn-risk scoring and buyer prioritization remain follow-up product
  work. This PR only surfaces the raw diagnostics in the paid report.
- Live Zendesk operator export rerun is still outside CI; this PR uses
  deterministic fixture-level tests.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_deflection_report.py -q` — 41 passed.
- `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py -q` — 302 passed.
- `pytest tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py -q` — 343 passed.
- `pytest tests/test_content_ops_deflection_resolution_live_proof.py -q` — 3 passed.
- `pytest tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py tests/test_content_ops_deflection_resolution_live_proof.py -q` — 355 passed after rebasing onto #1536.
- bash `scripts/run_extracted_pipeline_checks.sh` — 4101 passed, 10 skipped after rebasing onto #1536.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 16 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json` | 9 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json` | 9 |
| `extracted_content_pipeline/faq_deflection_report.py` | 116 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 156 |
| `plans/PR-Deflection-Outcome-Risk-Diagnostics.md` | 160 |
| `tests/test_content_ops_deflection_report.py` | 236 |
| **Total** | **702** |
