# PR-Deflection-Status-CSAT-Ingestion

## Why this slice exists

Discussion #1507 (deflection import-shape + dataset reconnaissance) established that a
real helpdesk export carries two structured outcome signals the deflection pipeline
currently throws away: **ticket status** (resolved vs abandoned / reopened) and
**customer-satisfaction rating**. Verified against the code: ingestion recognizes only
resolution *text* (`_RESOLUTION_TEXT_KEYS`); there is no `_STATUS_KEYS` / `_CSAT_KEYS`,
so a Zendesk `Ticket Status` / `Customer Satisfaction Rating` column (or Mendeley
`issue_status`) is dropped at the door. Both #1419 (prove + gate the publishable-answer
lane) and #1466 (proven-answer gate) need these signals: status separates the Tier-2
"answer-gap" backlog from resolved tickets, and `reopened` / low CSAT are the behavioral
"answered but not actually resolved" signals that are stronger and cheaper than scraping
reply text for instruction-shape.

This slice does the narrow, deterministic ingestion half: **recognize, normalize, and
expose** status and CSAT. It deliberately does **not** change the proven/gap
classification or any buyer-facing report output -- that consumption is #1419's product
decision and is out of scope here (per the no-product-shape-changes rule).

## Scope (this PR)

Ownership lane: deflection/ingestion
Slice phase: Production hardening

1. `extracted_content_pipeline/support_ticket_input_package.py`: add `_STATUS_KEYS` and
   `_CSAT_KEYS`; normalize status into a canonical state bucket (resolved / open /
   reopened / cancelled / other) and parse numeric CSAT; attach both (raw + normalized)
   onto each normalized row; surface aggregate status/CSAT summaries in package metadata.
2. Tests in the already-CI-enrolled `tests/test_extracted_support_ticket_input_package.py`
   proving recognition across header aliases, status bucketization, numeric-CSAT parsing,
   metadata summary correctness, and that absent columns leave the package unchanged.

### Review Contract
- Acceptance criteria: a row carrying `status` / `ticket_status` / `issue_status` / `state`
  gets `ticket_status` (raw) + `ticket_status_state` (canonical bucket); a row carrying a
  satisfaction column gets `csat` (raw) + `csat_score` (numeric when parseable); package
  metadata reports `ticket_status_present(_count)`, `ticket_status_summary` (per-bucket
  counts), `csat_present(_count)` (from raw `csat`, so textual good/bad reads as present),
  `csat_score_count`, `csat_score_average` (numeric-only); rows without these columns are
  byte-for-byte unchanged and produce no new warnings.
- Affected surfaces: one ingestion module + its unit test. No report output, no DB, no
  network, no generation-input change.
- Risk areas: status vocabulary misclassification (mitigated: unknown values map to
  "other", nothing is silently relabeled) and accidental product-behavior change
  (mitigated: proven/gap logic untouched; new data is additive passthrough + metadata).
- Reviewer rules triggered: R1 (requirements match -- `extracted_*` change), R2
  (failure-branch fixtures: absent / empty / non-numeric CSAT), R10 (maintainability),
  R12 (test runs in CI -- existing enrolled file).

### Files touched
- `extracted_content_pipeline/support_ticket_input_package.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `plans/PR-Deflection-Status-CSAT-Ingestion.md`

## Mechanism

`_STATUS_KEYS` and `_CSAT_KEYS` are matched through the existing `_first_value` /
`_key` machinery, so header aliases differing only in case / spaces / underscores
(`Ticket Status`, `ticket_status`, `issue_status`) resolve to the same key. Status text
is normalized by `_normalize_status_state` against four fixed value sets into one of
`resolved` / `open` / `reopened` / `cancelled` / `other`; unknown vocabulary returns
`other` so nothing is mislabeled. `reopened` is its own bucket because it is the churn
signal #1419/#1466 care about. CSAT is parsed by `_parse_csat_score`, which returns a
float only for numeric values (textual labels like Zendesk good/bad keep their raw value
but no score, because the negative-threshold decision belongs to #1419, not ingestion).
Per-row attachment is additive passthrough; aggregate summaries (`_ticket_status_summary`,
numeric-score list) are computed once and placed in `metadata` only -- generation inputs
are untouched.

## Intentional
- Deterministic only; no LLM/model dependency (matches the deflection positioning).
- Normalize-and-expose, not consume: the proven/gap split, churn-risk flagging, and any
  report-surface change are explicitly deferred to #1419 so the buyer-facing shape does
  not move in an ingestion slice.
- Unknown status vocabulary -> `other` (never silently coerced to resolved/open).
- CSAT negative-threshold is NOT decided here (a 1-2-is-bad rule assumes a 5-point scale;
  Zendesk CSAT is good/bad). Ingestion exposes raw + numeric; the product rule is #1419's.
- Aggregates live in `metadata` (diagnostics/preview), not `inputs` (generation), to keep
  the generation path byte-stable for rows without these columns.

## Deferred
- Consuming `ticket_status_state` / `reopened` / `csat_score` in the proven-vs-gap
  classification and the Tier-2 churn-risk lane (#1419).
- Demoting #1466's text gate to "one of three signals" in the proof logic (#1466).
- Resolution-disposition codes (`Won't Do` / `Duplicate` / `Cannot Reproduce`) as a
  distinct "resolved-on-paper-only" class -- a follow-up key set.
- Metrics-only vs full-mode importer branching and preview self-diagnosis (#1507 contract).

Parked hardening: none.

## Verification
- `tests/test_extracted_support_ticket_input_package.py`: alias recognition, status
  bucketization (closed->resolved, reopened->reopened, unknown->other), numeric-CSAT
  parse + non-numeric -> no score, metadata summary counts/average, and an unchanged-row
  control case.
- `scripts/check_ascii_python.sh` clean.
- Targeted: `pytest tests/test_extracted_support_ticket_input_package.py -q`.

## Estimated diff size
| Area | Est LOC |
|---|---:|
| Plan | ~115 |
| Ingestion module | ~100 |
| Tests | ~90 |
| Total | ~305 |

Within the 400 soft cap.
