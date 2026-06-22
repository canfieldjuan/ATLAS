# PR-Deflection-Action-Identity-Foundation

## Why this slice exists

#1612 and the linked #1316 delta-report plan both call out the same blocker:
the paid `deflection.v1` action report has business signals for monthly deltas,
but its evidence export only has rank-derived `question_id` values (`q001`,
`q002`, ...). Those IDs can change when priority/rank changes, so they are not
safe as month-over-month row identity.

Root cause: action rows and evidence-export questions had display/run-local
identity but no deterministic cross-run identity field. This PR fixes that root
for the paid model/export by adding stable identity fields derived from the
question/topic text, while keeping the free Snapshot projection fail-closed.

## Scope (this PR)

Ownership lane: issue-1612/deflection-full-report-delivery-actionability
Slice phase: Vertical slice

1. Add paid-only `repeat_key`, `cluster_id`, `identity_basis`, and
   `identity_confidence` to action rows and evidence-export question/evidence
   rows.
2. Preserve existing rank-derived `question_id` as a run-local/export display
   identifier for compatibility.
3. Prove the new identity survives rank reorder and remains absent from the
   free Snapshot projection.
4. Refresh the frontend contract/example artifact so downstream report and
   delta work sees the new fields.

### Review Contract

Acceptance criteria:
- Paid action rows and evidence-export rows expose stable identity fields.
- `repeat_key`/`cluster_id` are deterministic across rank reorder and ticket-ID
  rollover for the same question/topic.
- Low-confidence identity is explicit; this slice must not pretend rank is a
  durable fallback.
- Free Snapshot output remains allowlist-only; new paid identity fields do not
  appear in the teaser.
- Existing evidence-export `question_id`/`row_id` compatibility is preserved.

Affected surfaces:
- `extracted_content_pipeline/faq_deflection_report.py`
- `docs/frontend/content_ops_faq_report_contract.md`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

Risk areas:
- Snapshot/privacy projection widening.
- Evidence-export consumer compatibility.
- Future delta matching accidentally relying on rank instead of paid identity.

Reviewer rules triggered: R1, R10, R2, R3, R7, R13, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Action-Identity-Foundation.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

`_action_identity` normalizes the source question/topic text, then hashes those
deterministic parts into a `repeat_*` key. The digest is binary-encoded so the
committed example fixtures do not resemble API keys to secret scanning while
still carrying a deterministic non-PII identity. Source IDs do not participate
when question text exists, so a fresh monthly ticket set does not make the same
repeat look new. The basis and confidence are explicit:

- `question_topic`: high confidence.
- `question`: medium confidence.
- `source_ids`: medium confidence fallback only when no question exists.
- `insufficient_identity`: low confidence.

Action-row builders and evidence-export builders call that same helper. The
export keeps existing `question_id` and `row_id` values unchanged, but adds the
stable identity fields alongside them. The Snapshot projection stays unchanged
because no new `snapshot_safe_fields` are added.

## Intentional

- This is identity foundation only; it does not build the monthly delta report
  or customer-facing delta UI.
- `question_id` remains rank-derived for compatibility. New consumers should
  use `repeat_key`/`cluster_id` for cross-run matching.
- The helper uses deterministic lexical normalization, not source ticket IDs,
  fuzzy matching, or LLM classification. Merge/split/fuzzy identity can be
  layered later without making rank or evidence-set rollover the fallback.
- Codex P2 was valid: ticket IDs must not feed durable monthly identity. This
  update keeps source IDs only as a fallback when question text is absent.
- The `repeat_key` suffix is binary-encoded to satisfy the Gitleaks PR scanner
  on committed fixture values; it is still derived from SHA-256 digest bits and
  is not customer text.

## Deferred

- #1316 D1 monthly delta core: compare persisted paid `deflection.v1` models
  using these identity fields.
- Fuzzy/merge/split identity matching and low-confidence adjudication remain
  delta-lane follow-ups.
- Snapshot-safe exposure of any identity field remains deferred; paid-only is
  the boundary for this slice.

Parked hardening: none.

## Verification

- Focused identity/snapshot/doc pytest command -- 4 passed.
- Affected report/doc pytest files -- 153 passed.
- Python compile for touched Python files -- passed.
- Git whitespace check -- passed.
- Extracted content pipeline validation -- passed.
- Extracted reasoning-import guard -- clean.
- Extracted standalone audit -- Atlas runtime import findings: 0.
- ASCII Python policy check -- passed.
- Pending before push: local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 48 |
| `docs/frontend/content_ops_faq_report_contract.md` | 16 |
| `extracted_content_pipeline/faq_deflection_report.py` | 56 |
| `plans/PR-Deflection-Action-Identity-Foundation.md` | 133 |
| `tests/test_content_ops_deflection_report.py` | 107 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 5 |
| **Total** | **365** |
