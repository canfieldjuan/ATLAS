# PR-FAQ-Macro-Writeback-Scheduled-Telemetry
## Why this slice exists
PR-FAQ-Macro-Writeback-Scheduled-Publish intentionally deferred
operator-facing telemetry beyond the task result summary. The scheduled
publisher now returns a `_skip_synthesis` result so the runner avoids an LLM
summary, but the runner only notifies synthesized results. That leaves a
successful recurring Zendesk publish run silent even when autonomous result
notifications are enabled.
## Scope (this PR)
Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening
1. Add a runner metadata opt-in that sends `_skip_synthesis` builtin results
   through the existing best-effort `_notify_result(...)` path.
2. Register the scheduled FAQ macro publisher with that opt-in and notification
   tags.
3. Make the scheduled publisher's skipped-synthesis message include the
   aggregate tenant/draft counts it already computes.
4. Add focused tests for the runner opt-in, scheduler metadata, and scheduled
   summary content.
### Files touched
- `plans/PR-FAQ-Macro-Writeback-Scheduled-Telemetry.md`
- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `atlas_brain/autonomous/runner.py`
- `atlas_brain/autonomous/scheduler.py`
- `atlas_brain/autonomous/tasks/faq_macro_writeback_scheduled_publish.py`
- `tests/test_synthesis.py`
- `tests/test_scheduler.py`
- `tests/test_autonomous_faq_macro_writeback_scheduled_publish.py`
## Mechanism
The runner already centralizes notification delivery in `_notify_result(...)`,
which catches per-channel failures and respects the global autonomous notify
settings plus per-task `notify` metadata. This slice adds one small hook after
a builtin handler returns: if the result has a non-empty `_skip_synthesis`
string and the task metadata sets `notify_skipped_result`, the runner sends
that string through `_notify_result(...)` and returns the normal skipped
metadata. The scheduled macro task opts into that hook and replaces its static
skip string with a compact count summary after aggregation.
## Intentional
- No dashboard or history UI is added; this is only the autonomous notification
  path and compact run summary.
- No LLM synthesis is introduced for macro publish results; `_skip_synthesis`
  remains the deterministic path.
- No publish selection, Zendesk transport, credential lookup, or idempotency
  logic changes. Those were shipped in #1215 and #1216.
- Notification delivery remains best-effort and uses the existing
  `_notify_result(...)` channel handling rather than a task-specific sender.
## Deferred
- Future PR: operator-facing dashboard/history surfaces if the compact
  autonomous notification is not enough for day-to-day operations.
Parked hardening: none.
## Verification
- Python compile check for runner, scheduler, scheduled task, and tests -
  passed.
- Focused CI-enrolled macro-writeback workflow command with scheduled task,
  scheduler metadata, and runner skipped-notify tests - 12 passed.
- Broader synthesis + scheduled publish focused pytest - 25 passed.
- ASCII check for extracted package Python files - passed.
- Diff whitespace check - passed.
## Estimated diff size
| Area | LOC |
|---|---:|
| Plan | 65 |
| Workflow | 10 |
| Runner/scheduler/task | 45 |
| Tests | 80 |
| **Total** | **200** |
