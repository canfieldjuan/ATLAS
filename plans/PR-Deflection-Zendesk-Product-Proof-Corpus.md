# PR-Deflection-Zendesk-Product-Proof-Corpus

## Why this slice exists

The #1440 funnel has a full-volume CFPB stress proof, but CFPB is the wrong
*shape* of data: it is long-form regulatory complaint narrative, not support
tickets. It proves the funnel survives scale; it cannot prove the deflection
product produces *good* output, because real support queues cluster repeated
questions and CFPB does not. Product quality needs real ticket shape.

The operator has ~29 synthetic Zendesk tickets in the `finetunelab` trial,
created via the Zendesk API. This slice captures those into a sanitized,
committed **product-shaped eval corpus** with expected-outcome labels, so the
deflection output can be judged qualitatively (does it cluster the variants,
draft the public resolution, exclude private notes, and not publish reopened
non-resolutions). It is explicitly separate from the CFPB stress proof.

## Scope (this PR)

Ownership lane: content-ops/deflection-product-proof
Slice phase: Functional validation

1. `scripts/capture_zendesk_product_proof_corpus.py`: operator-run capture that
   wraps the existing live export client and projects raw `tickets + comments`
   into a sanitized fixture (whitelist fields; scrub emails/phones/identifier
   numbers/URL tokens; replace raw ticket/user IDs with local stable tokens and
   role pseudonyms so identity is dropped but the importer's role-split is
   preserved).
2. Expected-outcome metadata per ticket: `cluster_theme`,
   `should_publish_answer`, `has_private_note` (derived), `reopened`,
   `unresolved`. Derived facts are filled; judgment fields are labeled by a
   human pass.
3. Proof doc declaring this the Zendesk product-shaped proof corpus, separate
   from CFPB stress.
4. Keep `tests/fixtures/zendesk_full_thread_seed_sample.json` (4 rows) as a tiny
   smoke only.
5. Unit tests for the sanitizer (pure, no live credentials).
6. Fix the live export client (`support_ticket_zendesk_export.py`): the
   incremental cursor endpoint rejects `page[size]` (Zendesk 400), so drop it
   from the tickets URL. Found running the live dry-run; the existing test had
   encoded the buggy URL, so it is corrected and a regression guard added.

### Review Contract

- Acceptance criteria:
  - [ ] The committed fixture contains no credentials, tokens, emails, phone
        numbers, identifier-length numbers, raw Zendesk ticket/user IDs, or
        ticket URLs. Ticket IDs are local stable tokens (`zd-proof-001`).
  - [ ] Roles are preserved as pseudonyms (`requester`/`agent`/`system`) so the
        importer's `author_id == requester_id` split still separates customer
        wording (`description`) from agent resolution (`resolution_text`); a
        round-trip test through `rows_from_zendesk_full_thread` proves it.
  - [ ] Ticket projection keeps only `id`, `requester_id` (pseudonym),
        `subject`, `description`, `status`, `satisfaction_rating` (score only),
        `comments[{public, author_id (role), body}]`, and `expected`.
  - [ ] `has_private_note` is derived from `public=false` comments; judgment
        labels are present (may be null until the human labeling pass).
  - [ ] The proof doc names this product-shaped and separate from CFPB stress.
  - [ ] The 4-row fixture stays a tiny smoke; this corpus does not replace it.
  - [ ] The new test is enrolled in `run_extracted_pipeline_checks.sh` and the
        extracted-pipeline workflow path filters (script + test paths).
- Affected surfaces: validation scripts/fixtures/docs + CI enrollment + the
  Zendesk export client URL builder. No runtime generation, clustering, Stripe,
  or portfolio change; the importer is reused unchanged (confirmed by round-trip,
  not edited).
- Risk areas: PII/secret leakage into a committed artifact, importer role-split
  breakage, live export request shape, source-role drift, collision with #1565.
- Reviewer rules triggered: R1, R2, R3, R10, R14.

### Files touched

- `scripts/capture_zendesk_product_proof_corpus.py`
- `tests/test_capture_zendesk_product_proof_corpus.py`
- `scripts/run_extracted_pipeline_checks.sh` (enroll the new test)
- `.github/workflows/extracted_pipeline_checks.yml` (path filters for the new script + test)
- `extracted_content_pipeline/support_ticket_zendesk_export.py` (drop page[size])
- `tests/test_extracted_support_ticket_zendesk_export.py` (correct URL + regression)
- `docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json` (captured + labeled corpus)
- `docs/extraction/validation/deflection_zendesk_product_proof_corpus.md`
- `plans/PR-Deflection-Zendesk-Product-Proof-Corpus.md`

The corpus (50 tickets) is captured and labeled in this PR. The
expected-outcome labels are reviewer-drafted from the real tickets and remain
pending operator correction.

## Mechanism

`sanitize_zendesk_export` is a pure function: it takes the raw
`{"tickets": [{"ticket", "comments"}]}` artifact from
`export_zendesk_full_thread_artifact` and returns the committed shape. Only
whitelisted keys survive, so identity/metadata fields are dropped by
construction rather than blacklisted.

**Role preservation, not identity.** The importer splits customer wording from
agent resolution by comparing `comment.author_id` to `ticket.requester_id`, so
dropping those IDs outright collapses both into `resolution_text`. Instead the
sanitizer derives a role from the raw IDs at capture time and emits stable
pseudonyms in the same field names: `requester_id` becomes `"requester"`, and
each `author_id` becomes `"requester"` (matches the requester), `"system"` (an
automation `via.channel`), or `"agent"`. The importer's comparison then works
unchanged with no raw IDs committed -- a round-trip test through
`rows_from_zendesk_full_thread` confirms customer wording lands in `description`
and agent public replies in `resolution_text`. Ticket IDs become local
`zd-proof-NNN` tokens; the ticket `url` (which embeds the raw id) is dropped.

Retained text (subject, description, comment bodies) is scrubbed for emails,
phone-shaped runs, 6+ digit identifier numbers, and URL token parameters; short
numbers (amounts, years) are preserved. `_assert_no_secrets` is a final guard.

The capture CLI builds `ZendeskMacroCredentials` from the centralized
`content_ops_zendesk_*` config and is **operator-run** (it needs live
credentials). Writing is opt-in: the run previews (summary only) unless `--out`
is given, and `--dry-run` forces preview even with `--out`.

## Intentional

- **Build strictly on top of merged #1565.** #1565 owns the portfolio-ui submit
  live smoke (`faq-deflection-submit-live-smoke.mjs`); this slice touches only
  the export/capture path and does not modify that surface.
- **Capture the real 29 first; do not generate a parallel synthetic corpus.**
  Missing-case fill (toward ~30-40 tickets) is deferred and only if the 29 leave
  gaps in the needed cases.
- **Not the deterministic ground-truth lane.** This is a real-shape qualitative
  eval corpus; `scripts/build_synthetic_support_tickets.py` stays the separate
  zero-LLM clustering-correctness generator.
- **Sanitized artifact only.** No credentials, tokens, raw Zendesk IDs, or
  requester/author identity ever enter the committed fixture; only role
  pseudonyms and local IDs.

## Deferred

- Operator correction of the reviewer-drafted expected-outcome labels (the
  corpus is captured + labeled here; the labels are a draft for review).
- The funnel-run proof metrics: running this corpus through deflection and
  recording publishable-answer precision against `should_publish_answer`,
  private-note exclusion, and reopened/unresolved handling.
- The Zendesk pusher (dry-run/idempotent) and any missing-case generation,
  deferred unless the 50 leave gaps.

Parked hardening: none.

## Verification

- pytest tests/test_capture_zendesk_product_proof_corpus.py - 10 passed
  (includes the `rows_from_zendesk_full_thread` round-trip proving the
  customer/agent split survives sanitization).
- python -m py_compile scripts/capture_zendesk_product_proof_corpus.py
  tests/test_capture_zendesk_product_proof_corpus.py - passed.
- pytest tests/test_extracted_support_ticket_zendesk_export.py - 4 passed
  (existing cases corrected off the buggy URL + a regression that the
  incremental URL carries no page[size]).
- python scripts/audit_extracted_pipeline_ci_enrollment.py - OK.
- ASCII check - passed (0 non-ASCII in all touched files).
- Live read-only `--dry-run` against `finetunelab` after the export fix:
  50 tickets / 126 comments / 4 with private notes (hit the default limit;
  nothing written). Confirms auth + fetch + sanitize end-to-end.
- Post-export (operator + reviewer): `--out` run with a higher `--limit`,
  sanitization spot-check of the committed fixture, funnel run for qualitative
  output.

## Estimated diff size

| File | LOC |
|---|---:|
| `scripts/capture_zendesk_product_proof_corpus.py` | 210 |
| `tests/test_capture_zendesk_product_proof_corpus.py` | 185 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `extracted_content_pipeline/support_ticket_zendesk_export.py` | 6 |
| `tests/test_extracted_support_ticket_zendesk_export.py` | 18 |
| `docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json` | 1488 |
| `docs/extraction/validation/deflection_zendesk_product_proof_corpus.md` | 45 |
| `plans/PR-Deflection-Zendesk-Product-Proof-Corpus.md` | 165 |
| **Total** | **2122** |

Over the 400 soft cap: ~1488 of those lines are the committed corpus **data**
(the 50-ticket sanitized fixture), not logic. The executable surface (sanitizer
+ role preservation + export fix + test matrix) is ~630 lines and is what
carries review risk.
