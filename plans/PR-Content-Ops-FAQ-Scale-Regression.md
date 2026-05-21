# PR-Content-Ops-FAQ-Scale-Regression

## Why this slice exists

PR-Content-Ops-FAQ-1000-Row-Scale-Run proved and fixed the real local CFPB
1,000-row FAQ generator run, but that proof still depends on ignored `tmp/`
artifacts and a local public archive path. The fixed behavior needs a
repeatable CI regression that does not require the CFPB archive to exist on a
reviewer's machine.

This slice locks the 1,000-row confidence claim into the normal test suite with
synthetic CFPB-shaped source rows that exercise the same failure modes: issue
overflow, source-policy questions, malformed redacted question candidates, and
financial action guidance.

## Scope (this PR)

1. Add one focused 1,000-row CFPB-style regression test for the deterministic
   FAQ Markdown generator.
2. Keep the regression in the existing FAQ Markdown test file so it runs with
   the extracted content pipeline checks.
3. Do not change generator behavior in this slice unless the new regression
   exposes a missed bug.

### Files touched

- `plans/PR-Content-Ops-FAQ-Scale-Regression.md`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

The test constructs 1,000 in-memory support-ticket rows using CFPB-like
`source_title`, `source_id`, and evidence text. The distribution mirrors the
real run's shape closely enough to exercise overflow condensation while avoiding
a checked-in 1,000-row fixture file.

The assertions verify:

- all 1,000 rows are represented by `source_ids`;
- `max_items=12` passes every output check;
- no item uses `topic_fallback`;
- redacted/malformed snippets do not become FAQ headings;
- CFPB account topics do not receive SaaS reporting/export guidance.

## Intentional

- The rows are generated in test code instead of stored as a large fixture file.
  This keeps the PR small while still covering the 1,000-row behavior.
- This is a regression slice only; the behavior fix already landed in PR #705.
- The synthetic data is CFPB-shaped, not CFPB-sourced, so the test is stable and
  does not depend on public archive availability or live network access.

## Deferred

- Add a portfolio-site artifact link only after the product page copy is updated
  to use the fixed run confidently.
- Add live/archive smoke coverage only if the project later standardizes a
  checked-in public-data fixture or cache location.

## Verification

Completed:

- `pytest tests/test_extracted_ticket_faq_markdown.py -q` - 62 passed.
- `scripts/validate_extracted_content_pipeline.sh` via bash - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` - passed.
- `scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` via bash - passed.
- `scripts/run_extracted_pipeline_checks.sh` via bash - 1568 passed, 1 warning.
- Pending final wrapper after commit: `scripts/local_pr_review.sh origin/main`.

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Scale-Regression.md` | 81 |
| `tests/test_extracted_ticket_faq_markdown.py` | 119 |
| **Total** | **200** |

This is under the 400 LOC soft cap.
