# Deflection Resolution-Evidence Live Proof

Date: 2026-06-09

Issue: #1419

> Re-baselined 2026-06-10 for #1460 (measured repetition): FAQ grouping now
> requires the same question to repeat, not just shared topic membership. The
> unresolved-lane fixture rows were reworded so each theme genuinely repeats
> the same question (SSO certificate upload x3, CRM sync pause x2), and the
> committed report/summary/result artifacts were regenerated with the same
> documented command. Headline counts are unchanged; the summary gained
> `non_repeat_ticket_count` / `non_repeat_question_count` (both 0 here).

## What Ran

This validation proves the paid FAQ deflection report can produce publishable
answer groups when the uploaded support-ticket export includes real agent
resolution evidence. It complements the CFPB real-data proof, which correctly
proved the no-proven-answer lane for complaint data with no support
resolutions.

Command:

```bash
python scripts/build_content_ops_deflection_report.py \
  docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv \
  --source-format csv \
  --output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md \
  --summary-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json \
  --result-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json \
  --require-output-checks \
  --json
```

## Proof Artifacts

- Source CSV:
  `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv`
- Generated report sample:
  `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- Summary:
  `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json`
- Result envelope:
  `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json`

## Result

| Metric | Value |
|---|---:|
| Source rows | 12 |
| Ranked questions | 4 |
| Publishable answers drafted from proven resolutions | 2 |
| Questions still needing approved resolutions | 2 |
| Resolution evidence present | true |
| Resolution evidence count | 2 |

The generated report includes both required customer-facing lanes inside
`## Question Details and Evidence`:

- Two resolved-answer drafts:
  - attribution/report export, backed by 4 resolved tickets;
  - invoice download, backed by 3 resolved tickets.
- Two no-proven-answer themes:
  - SSO certificate upload, asked by 3 tickets but no resolution text;
  - CRM sync pause after a field-mapping change, asked by 2 tickets but no
    resolution text.

## Launch Implication

This closes the deterministic paid-report proof gap from #1419: when the
provider export includes scoped resolution text, the report produces
publishable help-center copy instead of only a gap list. When rows lack
resolution text, those questions still stay in the no-proven-answer lane.

This proof does not exercise Stripe, PDF/email delivery, or a production
operator upload. It proves the report artifact that those downstream surfaces
unlock and deliver.
