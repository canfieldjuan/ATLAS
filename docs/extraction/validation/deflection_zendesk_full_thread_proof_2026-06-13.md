# Deflection Zendesk Full-Thread Proof

Date: 2026-06-13

Issue: #1419

## What Ran

This validation proves the Zendesk full-thread import shape can drive the paid
FAQ deflection report's publishable-answer lane. The source fixture is a
captured Zendesk trial API artifact shaped as `tickets + comments`:

- ticket description and public requester comments become customer wording;
- public agent replies become resolution evidence;
- `public=false` internal notes are excluded;
- Zendesk `status` and `satisfaction_rating` remain diagnostics.

Source role: this is Zendesk-shaped product/integration evidence. It proves the
support-ticket object model, public/private comment handling, and
resolution-evidence lane are pointed at the buyer's actual support workflow. It
is not a full-volume stress proof by itself because the committed fixture is
small.

CI proof:

```bash
pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob -q
```

## Proof Artifacts

- Source fixture:
  `tests/fixtures/zendesk_full_thread_seed_sample.json`
- Generated summary sample:
  `docs/extraction/validation/fixtures/deflection_zendesk_full_thread_proof_20260613/summary.json`
- Generated report excerpt:
  `docs/extraction/validation/fixtures/deflection_zendesk_full_thread_proof_20260613/report_excerpt.md`

## Result

| Metric | Value |
|---|---:|
| Source rows | 4 |
| Ranked questions | 1 |
| Publishable answers drafted from proven resolutions | 1 |
| Questions still needing approved resolutions | 0 |
| Resolution evidence present | true |
| Resolution evidence count | 1 |

The generated paid artifact contains the public refund-answer resolution:

> We confirmed the duplicate billing event and refunded the extra charge.

The paid artifact does not contain the private internal note, the boilerplate
auto-ack, or the reopened customer's "still broken" follow-up as publishable
copy. Status and CSAT are still present in the input-provider metadata, but do
not create extra publishable answers by themselves.

## Launch Implication

This closes the deterministic #1419 proof gap for the Zendesk full-thread
path: an API-shaped Zendesk export with public agent replies can unlock
publishable help-center copy, while private notes and diagnostics do not leak
into customer-facing answers.

Pair this proof with CFPB stress/scale evidence before claiming the whole #1440
funnel is proven at both product-quality shape and full-volume size.
