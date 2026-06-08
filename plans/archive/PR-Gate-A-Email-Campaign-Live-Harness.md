# PR-Gate-A-Email-Campaign-Live-Harness

## Why this slice exists

Issue #1376 calls out that `email_campaign` generates through
`content_ops_execution.py`, but the Gate A live-quality harness cannot review
or export its saved campaign drafts. Campaign output is sequence-shaped: one
run can save multiple campaign rows, and the existing helpers are
`review_campaign_drafts(...)` / `list_campaign_drafts(...)`, not the generic
repository shape used by landing pages, blogs, and sales briefs.

The final diff exceeds the 400 LOC soft cap because review feedback added two
must-fix safety points to the same surface: empty email-campaign batch updates
must not 500, and the live harness needs an `--outputs` selector so the next
convergence proof can isolate the already-converged generators without editing
the script.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Production hardening

1. Add `email_campaign` to generated-asset review support by delegating to
   `review_campaign_drafts(...)`.
2. Add `email_campaign` to the Gate A harness outputs and export block, using
   `list_campaign_drafts(...)` filtered to this run's saved ids.
3. Make persistence checks sequence-aware: email campaigns must export every
   saved unique draft id and fail on duplicate/collapsed ids, while sibling
   outputs keep the variant-count rule.
4. Add focused tests for review delegation, unsupported status rejection,
   review handoff, export filtering, and duplicate sequence detection.
5. Address review feedback: empty email-campaign batch updates return no
   updates instead of delegating into the campaign normalizer, and the harness
   accepts `--outputs` to select the proof set for a run.

### Review Contract

- Acceptance criteria:
  - [ ] `review_saved_ids(...)` can approve saved `email_campaign` draft ids.
  - [ ] The generated-asset branch calls `review_campaign_drafts(...)` with
        `campaign_ids`, tenant scope, and the requested supported status.
  - [ ] `export_saved_drafts(...)` exports only saved email campaign ids from
        the current Gate A run.
  - [ ] Sequence persistence fails closed on duplicate saved ids or missing
        exported rows.
  - [ ] Malformed-only email campaign review batches keep the existing
        missing-id response path instead of raising from an empty campaign id
        list.
  - [ ] `--outputs` can run a selected subset of Gate A outputs, and summary /
        review / persistence checks only require that selected set.
  - [ ] This PR does not run or self-certify live campaign output quality.
- Affected surfaces: generated asset review internals, Gate A live-quality
  harness plumbing, and focused tests.
- Risk areas: sequence rows mistaken for one-draft variants, campaign review
  status vocabulary, and drifting into #1377's review-contract PR.
- Reviewer rules triggered: R1, R2, R6, R10.

### Files touched

- `extracted_content_pipeline/api/generated_assets.py`
- `plans/PR-Gate-A-Email-Campaign-Live-Harness.md`
- `scripts/smoke_content_ops_gate_a_live_quality.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_smoke_content_ops_gate_a_live_quality.py`

## Mechanism

`_update_asset_statuses("email_campaign", ...)` calls
`review_campaign_drafts(...)` and returns updated row ids, preserving the
existing `review_saved_ids(...)` missing-id check. The harness imports
`list_campaign_drafts(...)`, exports reviewed `approved` campaign rows for the
current `target_mode`, and filters the result to the run's saved ids.

`SEQUENCE_OUTPUTS = {"email_campaign"}` skips the sibling multiple-variant
requirement for campaign rows and routes persistence through
`_sequence_persistence_errors(...)`, which compares saved id entries, unique
ids, and exported row count.

`--outputs` defaults to the full Gate A set, but resolves to a validated tuple
that is threaded into payload construction, configured-output checks,
saved-id extraction, summary, review errors, and persistence checks. That keeps
the default all-output coverage while allowing a live convergence run to choose
`landing_page,blog_post,sales_brief`.

The email-campaign batch updater returns `()` before delegating when the route
has no valid UUID ids to review. It still maps the requested campaign status
first, so unsupported campaign review statuses remain a 400.

## Intentional

- Reuse campaign review/export helpers; do not create `export_campaign_drafts`
  or a fake generic campaign repository.
- Do not run the live Gate A proof or judge campaign copy quality. The live run
  remains reviewer-owned per issue #1376.
- Issue #1376 mentions `report`, but the current harness and archived #1360
  plan cover `landing_page`, `blog_post`, and `sales_brief`. This slice adds
  the issue-title gap, `email_campaign`, without expanding report coverage too.
- `email_campaign` accepts only statuses supported by `review_campaign_drafts`:
  `approved`, `queued`, `cancelled`, and `expired`.
- Cross-layer caller hints are same-name collisions; no non-diff caller invokes these private harness helpers.
- Do not touch #1377's `atlas_brain/_content_ops_review_workflow.py` lane.

## Deferred

- Reviewer-owned live Gate A run with the updated harness and real exported
  email campaign sequence.
- Follow-up if the support-ticket Gate A payload produces empty or degenerate
  campaign opportunities in an `--outputs email_campaign` live run.
- Report coverage if the operator wants this script expanded beyond #1360.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_smoke_content_ops_gate_a_live_quality.py -q` - PASS (`12 passed`).
- `python -m pytest tests/test_atlas_content_ops_generated_assets_api.py -q` - PASS (`19 passed`, one `pynvml` warning).
- `scripts/validate_extracted_content_pipeline.sh` - PASS (run with `bash`).
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - PASS.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - PASS (`Atlas runtime import findings: 0`).
- `scripts/check_ascii_python.sh` - PASS (run with `bash`).
- `scripts/run_extracted_pipeline_checks.sh` - PASS (run with `bash`; `extracted_reasoning_core`: `295 passed`; `extracted_content_pipeline`: `3354 passed, 10 skipped`; one `pynvml` warning).
- `scripts/local_pr_review.sh` - PASS (with current PR body file).

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/generated_assets.py` | 36 |
| `plans/PR-Gate-A-Email-Campaign-Live-Harness.md` | 133 |
| `scripts/smoke_content_ops_gate_a_live_quality.py` | 127 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 111 |
| `tests/test_smoke_content_ops_gate_a_live_quality.py` | 171 |
| **Total** | **578** |
