## Why this slice exists

Large SaaS-only live validation proved the deflection funnel can ingest 420
resolution-backed support-ticket rows, rank recurring questions, unlock through
the paid webhook path, and render the full report. It also exposed a correctness
bug in the FAQ draft pipeline: broad intent grouping can merge distinct
customer questions, causing a draft for one question to include resolution
steps from another question's source rows. Copy polish must wait until this
evidence-scoping invariant is fixed, because polished prose over cross-question
evidence would be confidently wrong.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Production hardening

1. Split resolution-backed support-ticket FAQ groups by normalized resolution
   evidence within the intent topic, so broad intent rules cannot merge
   distinct SaaS resolutions into the same draft.
2. Keep existing intent-topic clustering for rows that do not carry resolution
   evidence.
3. Add focused regression coverage proving resolutions from a dashboard-refresh
   question cannot bleed into an SSO/SCIM question.
4. Keep mixed-question overflow buckets review-needed instead of treating one
   leftover question's resolution as a publishable answer for the whole bucket.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `plans/PR-Deflection-Question-Evidence-Scoping.md`

## Mechanism

`build_ticket_faq_markdown` currently keys `groups` by `_topic(...)`, which can
collapse unrelated resolution-backed questions under broad rules such as
`integration setup`. This slice introduces a grouping key that scopes
resolution-backed rows by normalized resolution evidence:

```python
resolution_text = _resolution_text(evidence, opportunity)
if resolution_text:
    return f"resolution:{_compact_key(resolution_text)}"
return f"topic:{_topic(...)}"
```

The user-facing item topic remains the intent topic for labels and scoring, but
the grouping boundary is evidence-scoped when resolution evidence exists. That
keeps repeated phrasings that share the same proven resolution together while
preventing a draft from collecting a neighboring question's different
resolution_text. Rows without resolution evidence retain the existing
intent-topic grouping so non-drafted clustering behavior does not churn.

## Intentional

- This slice does not polish the visible step copy. The existing
  "Use the uploaded resolution evidence:" phrasing is rough, but polishing it
  before fixing evidence boundaries would hide the correctness bug.
- This slice does not add LLM clustering or semantic similarity. The production
  failure came from over-merging different resolutions, so the conservative fix
  is stricter evidence-scoped grouping.
- Mixed-question overflow still preserves source coverage for condensation, but
  it now fails closed to `draft_needs_review` instead of claiming a publishable
  answer from one leftover question's resolution.

## Deferred

- Follow-up slice: `PR-Deflection-Resolution-Copy-Polish` should turn
  resolution_text into clean help-center prose once grouping is safe.
- Follow-up slice: add a stricter artifact/eval gate that fails closed if an
  item step cites resolution evidence outside that item's evidence scope.
- Parked hardening: none.

## Verification

- `pytest tests/test_extracted_ticket_faq_markdown.py -k "resolution_evidence or sharing_resolutions or overflow_resolution or unresolved_overflow" -q`
  - Passed: 5 passed, 146 deselected.
- `pytest tests/test_extracted_ticket_faq_markdown.py -q`
  - Passed: 151 passed.
- `python scripts/build_content_ops_deflection_report.py /home/juan-canfield/Desktop/saas-deflection-large-sample.csv --source-format csv --max-items 8 --result-output /tmp/deflection-large-saas-local-after-scoping.json --summary-output /tmp/deflection-large-saas-local-after-scoping-summary.json --require-output-checks --json`
  - Passed: generated 8; drafted_answer_count 7; no_proven_answer_count 1;
    output checks all true; source_count 420.
- `pytest tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_snapshot_example_matches_producer_shape -q`
  - Passed: 1 passed.
- `env ATLAS_API_BASE_URL= ATLAS_B2B_JWT= ATLAS_TOKEN= ATLAS_ACCOUNT_ID= ATLAS_FAQ_SEARCH_ACCOUNT_ID= ATLAS_DEFLECTION_SUBMIT_BLOB_URL= ATLAS_DEFLECTION_SUBMIT_CSV_FILE= ATLAS_DEFLECTION_COMPANY_NAME= ATLAS_DEFLECTION_CONTACT_EMAIL= ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL= EXTRACTED_DATABASE_URL= DATABASE_URL= pytest tests/test_smoke_content_ops_deflection_submit_handoff.py::test_validate_args_fails_closed_for_missing_and_unsafe_inputs tests/test_smoke_content_ops_deflection_submit_handoff.py::test_validate_args_defaults_to_checked_fixture_when_no_source_is_provided tests/test_smoke_content_ops_deflection_submit_handoff.py::test_run_success_posts_multipart_csv_and_probes_locked_artifact tests/test_smoke_content_ops_deflection_portfolio_result_page.py::test_preflight_only_writes_missing_inputs_without_network -q`
  - Passed: 4 passed.
- `bash scripts/run_extracted_pipeline_checks.sh`
  - Local caveat: reached 2879 passed, 10 skipped, then failed on 6
    deflection smoke/contract tests because the repo-root `.env` is loaded by
    smoke helpers during tests. The code-caused contract-doc failure was fixed;
    the env-sensitive deflection subset passes when deflection env defaults are
    blanked. I did not move or edit the user's `.env` secret file to sanitize
    the remaining subprocess preflight test.
- `bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file /tmp/pr-deflection-question-evidence-scoping-body.md`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | ~65 |
| `tests/test_extracted_ticket_faq_markdown.py` | ~127 |
| Plan doc | ~92 |
| **Total** | **~284** |

Under the 400 LOC soft cap.
