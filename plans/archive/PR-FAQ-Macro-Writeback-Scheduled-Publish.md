# PR-FAQ-Macro-Writeback-Scheduled-Publish
## Why this slice exists
FAQ macro writeback is manual today. This slice adds the recurring Zendesk
publisher so approved, verified FAQ drafts can write back after a report sale.
Intercom remains parked because #1214 proved its macro API is read-only.
## Scope (this PR)
Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice
1. Add and register an interval-based autonomous task gated by typed config.
2. Select stored-credential Zendesk tenants by least-recent publish coverage and approved+verified FAQ drafts without completed mappings.
3. Call existing `FAQMacroWritebackPublishService.publish_faq_draft(...)` per draft and aggregate status.
4. Add focused main-suite tests with fakes; no live Zendesk calls.
### Files touched
- `plans/PR-FAQ-Macro-Writeback-Scheduled-Publish.md`
- `atlas_brain/config.py`
- `atlas_brain/autonomous/scheduler.py`
- `atlas_brain/autonomous/tasks/__init__.py`
- `atlas_brain/autonomous/tasks/faq_macro_writeback_scheduled_publish.py`
- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `tests/test_autonomous_faq_macro_writeback_scheduled_publish.py`
- `tests/test_scheduler.py`
## Mechanism
The task uses normal autonomous interval registration. At runtime it opens the
DB pool, builds the existing FAQ repository, mapping/attempt repositories, and
Zendesk provider with a stored-tenant-only credential source. Candidate
selection starts from active `content_ops_zendesk_credentials`, orders enrolled
tenants by never/least-recent Zendesk publish coverage, logs/reports any tenants
deferred by the per-run cap, filters approved drafts through the shared macro
preview helper, skips drafts whose Zendesk mappings are already published, then
calls `publish_faq_draft` per candidate. Per-draft exceptions are caught and
reported so the batch continues.
## Intentional
- No Intercom code is added or revived; #1214 stays parked because Intercom
  macros are GET-only.
- No host/global Zendesk config fallback; missing tenant credentials skip.
- No reimplemented Zendesk transport, payloads, history, or final double gate.
- No live Zendesk smoke in CI; tests use fakes around the orchestrator.
- The per-run tenant cap remains 25, but capped runs are observable and the
  tenant order rotates toward never/least-recently published tenants.
## Deferred
- Future PR: content-hash "needs update" detection for body-only FAQ edits.
- Future PR: operator-facing telemetry beyond the task result summary.
Parked hardening: none.
## Verification
- Python compile check for the new task, tests, config, and scheduler - passed.
- Focused pytest for scheduled publish orchestration + scheduler opt-in regression - 6 passed.
- ASCII check for extracted package Python files - passed.
- Atlas-brain CI enrollment audit - passed, 140 matching tests enrolled.
- Evidence-claim wiring regression sweep for scheduler registration - 4 passed.
- Local PR review bundle with the prepared PR body file - passed.
## Estimated diff size
| Area | LOC |
|---|---:|
| Plan | 63 |
| Config/scheduler/workflow | 83 |
| Task | 213 |
| Tests | 214 |
| **Total** | **573** |

Over the 400 LOC soft cap because review feedback required both scheduler
seeding coverage and a dedicated macro-writeback workflow, not just the task
orchestrator.
