# PR-Deflection-Parser-JSON-Message-Guard

## Why this slice exists

#1626 named a deterministic fail-open: a mapped `Message` containing only
machine JSON, such as `{"event":"ticket_created","id":123}`, counts as usable
customer text. The root cause is upstream of the evidence runner:
`source_row_to_campaign_opportunity` treated the first non-empty
`_source_text(...)` alias as usable after stringification, so a machine JSON
alias could both count as customer wording and mask a later human alias. This
fixes that class at the source-row conversion boundary; it does not set the
#1467/#1582 low non-zero coverage threshold.

This exceeds the 400 LOC soft cap because review found the root was broader
than the original scalar-string check: the fix had to replace source-text
candidate selection and add held-out tests for short customer JSON, verbose
machine metadata, later-alias fallback, and structured JSON/JSONL values. The
scope remains one indivisible parser-admission boundary.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Production hardening

1. Treat all-machine JSON payloads in mapped scalar source-text fields as
   unusable source text.
2. Emit a specific warning for skipped machine payload rows.
3. Update the #1626 breakage matrix so the JSON payload case is fail-closed.
4. Drain the matching `HARDENING.md` entry and archive the merged #1626 plan.

### Review Contract

Acceptance criteria:
- The guard runs in source-row conversion/admission, not only in
  `scripts/evaluate_csv_admission_threshold_evidence.py`.
- A row whose mapped `Message` is exactly machine JSON is not converted into an
  opportunity and drives zero-usable CSV admission to `REJECT`.
- Partial CSVs with one human message and one machine JSON payload still
  `ACCEPT` with `partial_source_row_coverage` plus the machine-payload warning.
- Near misses stay accepted: ordinary human text that mentions JSON, short
  customer messages inside message-bearing JSON fields, and structured JSON
  objects carrying customer message values must not be rejected by the
  machine-payload guard.
- Machine JSON in an earlier text alias must not hide later customer wording in
  another mapped source-text alias.
- The #1626 evidence artifact records the JSON case as fail-closed, not
  `known_gap`.
- #1467/#1582 low non-zero threshold policy remains deferred until real partial
  provider CSV evidence exists.

Affected surfaces:
- Source-row CSV parser/admission conversion, diagnostics tests, CSV admission
  proof artifact, and same-lane plan archive/index housekeeping.

Risk areas:
- Over-rejecting real customer text that includes JSON snippets or API errors.
- Hiding the reason a mapped text row stopped counting as usable.
- Accidentally converting this deterministic guard into broad low-coverage
  threshold policy.

Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Files touched

- `HARDENING.md`
- `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md`
- `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Parser-JSON-Message-Guard.md`
- `plans/archive/PR-Deflection-Parser-Breakage-Evidence-Runner.md`
- `scripts/evaluate_csv_admission_threshold_evidence.py`
- `tests/test_evaluate_csv_admission_threshold_evidence.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_ingestion_diagnostics.py`

## Mechanism

- Replace "first non-empty text alias wins" with "first usable customer-text
  candidate wins": source-text selection skips machine JSON aliases, continues
  to later aliases/thread candidates, and only reports
  `machine_source_payload_text` when no usable customer text exists.
- Inspect JSON before stringification for both string payloads and structured
  JSON/JSONL values. Only message-bearing JSON keys such as `message`,
  `customer_message`, `requester_comment`, `body`, and `text` can supply
  customer wording; verbose metadata keys such as `event` do not.
- Have `source_row_to_campaign_opportunity(...)` return no opportunity and a
  `machine_source_payload_text` warning for all-machine JSON rows, reusing the
  existing raw/usable source-row admission count.
- Keep near misses usable: text containing `{...}`, short customer messages in
  JSON, and structured JSON with a customer message value still pass.
- Regenerate the CSV admission proof artifact after the runner observes the
  JSON case as `REJECT`.

## Intentional

- No low non-zero usable-ratio threshold. This PR changes row usability for a
  deterministic machine-payload class only.
- No live provider calls and no real customer CSVs; the bug class is isolated
  by synthetic fixtures.
- The guard is conservative where customer-message keys exist: short
  message-bearing JSON values such as `{"message":"Cannot login"}` remain
  usable, while machine metadata values under non-message keys do not.

## Deferred

- #1467/#1582 low non-zero reject threshold remains blocked on real partial
  provider CSV evidence.
- Provider-specific structured JSON extraction remains out of scope; this guard
  only stops machine payloads from counting as scalar customer wording.

Parked hardening:
- Closes `HARDENING.md`: "CSV source-row admission accepts machine JSON in
  mapped message fields."

## Verification

- `pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_content_ingestion_diagnostics.py tests/test_evaluate_csv_admission_threshold_evidence.py -q`
  - 161 passed in 0.86s.
- `python scripts/evaluate_csv_admission_threshold_evidence.py --json`
  - Passed; generated a 6-case breakage matrix with 0 blocking cases and 0
    known gaps. The JSON message case now observes `REJECT` with
    `machine_source_payload_text`.
- `./scripts/run_extracted_pipeline_checks.sh`
  - 4574 passed, 10 skipped, 1 warning in 75.33s.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 11 |
| `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md` | 4 |
| `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json` | 38 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 245 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Parser-JSON-Message-Guard.md` | 142 |
| `plans/archive/PR-Deflection-Parser-Breakage-Evidence-Runner.md` | 0 |
| `scripts/evaluate_csv_admission_threshold_evidence.py` | 9 |
| `tests/test_evaluate_csv_admission_threshold_evidence.py` | 18 |
| `tests/test_extracted_campaign_source_adapters.py` | 129 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 76 |
| **Total** | **673** |
