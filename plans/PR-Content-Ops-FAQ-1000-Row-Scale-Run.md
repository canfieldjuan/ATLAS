# PR-Content-Ops-FAQ-1000-Row-Scale-Run

## Why this slice exists

The portfolio demo copy was revised to say the FAQ Report can handle
500-1,000+ ticket CSV batches, but that claim needs a real run against the
actual FAQ generator. The previous 46-row demo excerpt is not enough proof.

This slice records a 1,000-row local CFPB archive run through the extracted FAQ
Markdown generator, logs the issues surfaced during the run, and fixes the
generator failures that made the initial fail-closed run unsuitable as a
production proof point.

## Scope (this PR)

1. Extract 1,000 narrative CFPB complaint rows from the local public archive.
2. Run those rows through `scripts/build_extracted_ticket_faq_markdown.py`.
3. Record performance, output-check results, generated artifact paths, and
   quality issues.
4. Fix the generator behaviors surfaced by the run: tail-group condensation,
   topic-fallback questions, malformed redacted question extraction, and
   CFPB/banking action-step mismatches.
5. Re-run the 1,000-row fail-closed command and record the fixed result.

### Files touched

- `plans/PR-Content-Ops-FAQ-1000-Row-Scale-Run.md`
- `docs/extraction/validation/content_ops_faq_1000_row_run_2026-05-21.md`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`

## Mechanism

The run used the local CFPB CSV archive at:

```text
/home/juan-canfield/Downloads/archive (1)/rows.csv
```

A small local extraction command reused
`scripts/export_content_ops_cfpb_sources.py::cfpb_row_to_source_row` to convert
the first 1,000 non-empty complaint narratives into source-row JSONL under
`tmp/content_ops_faq_1000/`. The existing standalone FAQ builder then loaded
that JSONL and generated Markdown from the real deterministic generator path.

The fix keeps the deterministic generator path. When the number of issue groups
exceeds `max_items`, the generator keeps the highest-volume groups and folds the
tail into one overflow FAQ item so every source row is still represented by the
condensed output. Customer-wording extraction now rejects redacted or malformed
question candidates, and topic fallback questions are replaced by source-policy
questions. Action-step selection now has CFPB/banking rules before generic SaaS
reporting/profile rules.

## Intentional

- The large source JSONL and generated Markdown artifacts stay under ignored
  `tmp/` because they are local validation outputs, not source-controlled
  product fixtures.
- The local archive is used instead of CFPB live fetch because the live endpoint
  was previously observed returning 504 during planning.
- The overflow FAQ item can include many source IDs while displaying only the
  configured evidence cap. This matches the existing condensed-output contract:
  metadata tracks all represented ticket sources while Markdown stays readable.

## Deferred

- Add a smaller checked-in CFPB-style fixture that exercises the 1,000-row
  regressions without depending on the local public archive.
- Decide whether the portfolio demo should link the full fixed Markdown artifact
  or keep using curated excerpts.

## Verification

Completed:

- `wc -l '/home/juan-canfield/Downloads/archive (1)/rows.csv'`
- Extracted 1,000 narrative source rows to
  `tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl`.
- Ran `python scripts/build_extracted_ticket_faq_markdown.py ... --max-items 12
  --require-output-checks`; it failed with `condensed` and
  `uses_user_vocabulary`.
- Ran direct builder inspection with `max_items=12`; generated 12 groups from
  1,000 rows in about 0.5s and about 23 MB RSS, but output checks failed.
- Ran direct builder inspection with `max_items=50`; generated 15 groups and
  passed `condensed` and `has_action_items`, but still failed
  `uses_user_vocabulary`.
- Inspected generated Markdown and item metadata for malformed questions,
  fallback questions, and wrong action steps.
- `pytest tests/test_smoke_content_ops_cfpb_faq_markdown.py
  tests/test_extracted_ticket_faq_markdown.py` - 65 passed.
- Re-ran the 1,000-row fail-closed command with `--max-items 12`; exit status
  0, elapsed wall time `0:00.53`, max RSS `23856 KB`.
- Wrote fixed inspection output to
  `tmp/content_ops_faq_1000/cfpb_1000_faq_fixed_result.json`; all output checks
  were true.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 1566 passed, 1 warning.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-1000-Row-Scale-Run.md` | 119 |
| `docs/extraction/validation/content_ops_faq_1000_row_run_2026-05-21.md` | 257 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 99 |
| `tests/test_extracted_ticket_faq_markdown.py` | 192 |
| `tests/test_smoke_content_ops_cfpb_faq_markdown.py` | 13 |
| **Total** | **680** |

This exceeds the 400 LOC soft cap because this PR now includes both the original
1,000-row evidence log and the focused fixes for each surfaced failure. Splitting
would leave the PR in a known fail-closed state after the user explicitly asked
to address the failures now.
