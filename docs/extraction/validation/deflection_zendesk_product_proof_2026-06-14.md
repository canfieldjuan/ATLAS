# Deflection Zendesk Product Proof

Date: 2026-06-14

Issues: #1419, #1440

## What Ran

This validation replaces the four-row Zendesk fixture as the product-shaped
proof corpus for the paid FAQ deflection report. It exported a seeded Zendesk
trial tenant through the live Zendesk API, normalized the ticket/comment
threads with the full-thread importer, built the support-ticket input package,
and generated the deterministic FAQ deflection report artifact locally.

The source role is Zendesk product/integration evidence:

- ticket descriptions and public requester comments became customer wording;
- public agent replies became candidate resolution evidence;
- `public=false` internal notes stayed out of customer-facing report excerpts;
- Zendesk `status` and `satisfaction_rating` populated diagnostics without
  creating publishable answers by themselves.

This is not a full-volume stress proof. CFPB remains the scale corpus; this
Zendesk trial export is the buyer-shaped corpus that proves the support desk
object model, public/private role handling, status diagnostics, CSAT
diagnostics, and proven-answer lane on a realistic Zendesk thread shape.

## Proof Artifacts

- Sanitized summary:
  `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/summary.json`
- Sanitized report excerpt:
  `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/report_excerpt.md`
- Raw Zendesk export: not committed. The local run wrote it under
  `tmp/zendesk_product_proof/raw_zendesk_export.json` so credentials, raw
  customer text, and full comment bodies do not enter git history.

## Result

| Metric | Value |
|---|---:|
| Tickets exported | 50 |
| Comments exported | 126 |
| Public comments | 122 |
| Private comments | 4 |
| Normalized rows | 50 |
| Normalization warnings | 0 |
| Answer drafts from proven resolutions | 7 |
| Questions still needing proven answers | 1 |
| Resolution evidence present | true |
| Status diagnostics present | true |
| CSAT diagnostics present | true |

Observed Zendesk status distribution:

| Status | Tickets |
|---|---:|
| open | 16 |
| pending | 5 |
| solved | 29 |

Observed Zendesk satisfaction distribution:

| Satisfaction score | Tickets |
|---|---:|
| good | 24 |
| bad | 5 |
| unoffered | 21 |

The generated report drafted answer candidates for the seeded refund, MFA/login,
replacement shipment, localized UI, password reset, and admin-step themes. The
committed excerpt includes representative generated samples, and the committed
summary records:

```json
{
  "publishable_answer_count": 7,
  "no_proven_answer_count": 1,
  "private_note_leak_checks": {
    "auto_ack_present_in_markdown": false,
    "private_marker_present_in_markdown": false
  }
}
```

## Output-Quality Boundary

These are ingestion/proven-resolution **drafts**, not buyer-ready FAQ entries.
The committed excerpt intentionally preserves two visible output-quality
defects:

- Synthetic subject prefixes leak into generated question headings, for example
  `[Atlas seed 10] Login and MFA access issue ...` and `[Atlas seed 22] Shipping
  or replacement question ...`. That means this corpus proves the Zendesk
  object model and public/private role handling, but not real-ticket
  question-label polish.
- The degraded question label `What should I do about atla?` appears three
  times in `top_questions` (ranks 4, 6, and 8; ticket counts 4, 1, and 1).
  That is a label degradation and clustering-quality gap, not a Zendesk
  ingestion or private-note leak failure.

Those defects are deliberately documented here instead of hidden behind the
answer count. A later quality slice must clean up synthetic subject-prefix
pollution and the degraded `atla` label before this proof can be treated as a
buyer-ready FAQ quality pass.

## Exporter Boundary

The first proof attempt exposed a real Zendesk API mismatch in the exporter:
the incremental cursor endpoint rejected the old URL shape that included
`page[size]`.

```text
/api/v2/incremental/tickets/cursor?start_time=0&page%5Bsize%5D=50
```

The exporter URL fix and the committed 50-ticket corpus now live in #1567,
which is the canonical source for Zendesk cursor-endpoint behavior. This proof
doc does not claim page-size live validation or own exporter behavior; it
records the sanitized report artifact produced from the seeded Zendesk trial
export.

## Boundaries

- The raw export is intentionally excluded from git. Only sanitized scalar
  metrics and short answer excerpts are committed.
- The run did not mutate Stripe state, unlock a paid artifact, send email, or
  run the public portfolio wrapper.
- The hosted Atlas export route at the configured `ATLAS_API_BASE_URL` was not
  available for this proof window (`/api/v1/content-ops/zendesk-export/full-thread`
  returned 404/405, and the unversioned route returned 502). The product proof
  therefore uses the same exporter code directly against Zendesk credentials.
- Local portfolio wrapper prerequisites were also missing:
  `BLOB_READ_WRITE_TOKEN` and `ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN`.
- `output_checks.resolution_evidence_scoped` was false in the generated report
  summary because two one-ticket seeded clusters had missing question scope.
  That is a report-quality boundary, not a private-note or Zendesk-ingestion
  failure; the publishable answer count remains based on proven public agent
  replies.
- The output-quality boundary above means the report is not buyer-ready as-is:
  it has synthetic subject-prefix pollution and a repeated degraded question
  label.

## Launch Implication

This closes the Zendesk ingestion/private-note proof gap for the full-thread
path: a live Zendesk API export with public agent replies can produce answer
drafts while private notes and diagnostics stay out of customer-facing copy. It
does **not** close buyer-ready question-label quality because the committed
samples still show subject-prefix pollution and the degraded `atla` label. Pair
this evidence with the CFPB stress proof and a follow-up output-quality fix
before claiming the whole #1440 funnel is proven at buyer-shaped quality and
full-volume size.
