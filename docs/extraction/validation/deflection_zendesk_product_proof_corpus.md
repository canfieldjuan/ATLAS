# Deflection Zendesk Product-Proof Corpus

Date: 2026-06-14

Issue: #1440

## Source role

This is the **Zendesk product-shaped proof corpus**: real support-ticket shape
(ticket + comment object model, public requester/agent wording, `public=false`
internal notes, resolution evidence) captured from the operator's `finetunelab`
Zendesk trial via the Zendesk API.

It is **separate from the CFPB full-volume stress proof**
(`deflection_full_volume_live_proof_2026-06-14.md`). CFPB proves the funnel
survives scale on messy long-form narratives; it cannot judge product quality,
because it is not real support-ticket shape. This corpus exists to judge product
quality qualitatively: does deflection cluster repeated questions, draft the
public resolution as the publishable answer, exclude private notes, and refrain
from publishing reopened non-resolutions.

The `full-volume-cfpb` gate thresholds (e.g. the 25,000 repeat-ticket minimum)
do **not** apply here. This is a small real-shape corpus (50 tickets), sized
to test clustering, resolution extraction, private-note exclusion, and
reopened-ticket handling -- not statistical volume.

The 4-row `tests/fixtures/zendesk_full_thread_seed_sample.json` remains a tiny
smoke only; this corpus does not replace it.

## How the corpus is captured

```bash
# operator, with finetunelab ATLAS_CONTENT_OPS_ZENDESK_* credentials set:
.venv/bin/python scripts/capture_zendesk_product_proof_corpus.py --dry-run   # preview
.venv/bin/python scripts/capture_zendesk_product_proof_corpus.py \
  --run-tag atlas_product_proof_20260614 \
  --out docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json
```

The capture script projects raw `tickets + comments` to a whitelisted, scrubbed
shape: no credentials, emails, phones, identifier numbers, raw Zendesk
ticket/user IDs, or URLs. Ticket IDs become local `zd-proof-NNN` tokens, and
requester/author identity is replaced by `requester`/`agent` role pseudonyms so
the importer still separates customer wording from agent resolution without
committing real IDs (proven by a `rows_from_zendesk_full_thread` round-trip
test). Automation/`system` comments are excluded so they cannot become
resolution_text. `has_private_note` is derived;
expected-outcome labels (`cluster_theme`, `should_publish_answer`, `reopened`,
`unresolved`) are filled by a reviewer-drafted, operator-corrected labeling pass.

## Result

Captured + labeled from the `finetunelab` trial via the live export
(`--limit 200`); verified PII-clean (0 emails / phones / identifier numbers /
raw IDs / secrets).

| Metric | Value |
|---|---:|
| Tickets | 50 |
| Comments | 126 |
| Distinct cluster themes | 14 |
| `should_publish_answer = true` | 36 |
| `unresolved = true` | 7 |
| `reopened = true` | 4 |
| `has_private_note = true` | 4 |

The themes include deliberate negative cases: "Unanswered product question" (6),
"Answered but reopened risk signal" (4), and "Private note must not become
answer" (3), alongside repeat-question clusters (duplicate charge x8, login/MFA
x7, API/webhook x6, shipping x5). Labels are reviewer-drafted from the theme +
structure and are **pending operator correction**.

Still deferred: the funnel run itself -- feeding this corpus through deflection
and scoring publishable-answer precision against `should_publish_answer`,
private-note exclusion, and reopened/unresolved handling. Until then this corpus
is captured-and-labeled but not yet a qualitative product pass.

## Pairing

A full #1440 funnel claim needs both classes: CFPB stress (survives volume) +
this Zendesk corpus (real product shape). Neither alone proves the buyer-facing
funnel.
