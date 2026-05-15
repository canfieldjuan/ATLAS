# Content Ops Survey Source Adapter

## Why This Slice Exists

AI Content Ops can ingest review, transcript, complaint, document, support
ticket, and ticket-thread source rows. Hosts also commonly have NPS, CSAT, or
survey exports where the useful source text lives in feedback/comment fields
and the score should stay attached as prompt-visible context.

## Scope

- Extend `extracted_content_pipeline/campaign_source_adapters.py` to recognize
  survey/NPS/CSAT source-row collections and ids.
- Preserve survey score fields on the normalized opportunity while using
  feedback/comment text as evidence.
- Add focused source-adapter tests.
- Refresh host-facing source-row docs and coordination state.

## Mechanism

- Add collection keys such as `survey_responses`, `surveys`, `nps_responses`,
  and `csat_responses`.
- Add id keys such as `survey_id`, `response_id`, and `feedback_id`.
- Treat `response_id` as a survey response identifier when it appears in a
  source row.
- Add text keys such as `feedback`, `feedback_text`, `response_text`,
  `comment_text`, and `open_ended_response`.
- Infer `source_type` as `nps_response`, `csat_response`, or
  `survey_response` when score/id fields indicate a survey bundle.

## Intentional

- No scoring logic or quality classification.
- No generated-asset, API, or database changes.
- No new CLI flags or source file formats.
- Score fields remain ordinary preserved source metadata.

## Deferred

- Survey-specific source quality and sentiment scoring.
- Longitudinal aggregation across repeated surveys.
- Dedicated survey reasoning pack inputs.

## Verification

- Focused source-adapter tests.
- Compile check for touched Python files.
- Local PR review gate.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `plans/PR-Content-Ops-Survey-Source-Adapter.md`
- `tests/test_extracted_campaign_source_adapters.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Source adapter | ~30 |
| Tests | ~45 |
| Docs and coordination | ~30 |
| Plan doc | ~55 |
| **Total** | ~160 |
