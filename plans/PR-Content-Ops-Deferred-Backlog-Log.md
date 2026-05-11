# PR-Content-Ops-Deferred-Backlog-Log

## Why this slice exists

The Content Ops extraction has accumulated many `Deferred` sections across
older PR plans. Most of the high-impact items have since landed, but the old
plan docs still make it hard to tell what is active backlog versus closed
history.

This docs-only slice records the remaining AI Content Ops deferrals in priority
order so the next implementation slices can be picked deliberately.

## Scope (this PR)

1. Add a current AI Content Ops deferred backlog doc.
2. Link the backlog from `extracted_content_pipeline/STATUS.md`.
3. Update the coordination in-flight table for this docs slice.

### Files touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Deferred-Backlog-Log.md`

## Mechanism

The backlog separates retired historical deferrals from active follow-up work.
Active items are ordered by production value: operational proof first, then
data-feed readiness, then advanced reasoning provider work, then operator UX.

## Intentional

- Documentation only. No runtime behavior changes.
- The backlog is intentionally ordered, not exhaustive grep output.
- Old plan docs are left intact as historical records.

## Deferred

- Implementing the backlog items themselves.

## Verification

- `rg -n "Active Backlog|Retired Historical Deferrals|Deferred Backlog" docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md extracted_content_pipeline/STATUS.md docs/extraction/coordination/inflight.md`
- `git diff --check`

## Estimated diff size

4 files, under 160 LOC.
