# PR-Archive-Resolution-Audit-Zendesk-Plans

## Why this slice exists

Teardown housekeeping for the content-ops/resolution-audit-zendesk-writeback
lane: both PRs merged (#1979 foundation, #1955 route), so their plan docs are
merged history and belong under `plans/archive/` per the plan lifecycle
(`scripts/archive_plans.py`). This moves those two and regenerates the archive
index. Scoped to this lane's two plans rather than a full sweep because the
full sweep is blocked by a pre-existing reused-slice-name collision
(`PR-Deflection-Question-Label-Merge.md` exists in both root and archive),
which is a repo-wide data-hygiene issue outside this lane.

## Scope (this PR)

Ownership lane: content-ops/resolution-audit-zendesk-writeback
Slice phase: Workflow/process

1. Move the two merged lane plan docs from `plans/` root into `plans/archive/`.
2. Regenerate `plans/INDEX.md` from the archive (index-only; no sweep).

### Files touched

- `plans/INDEX.md`
- `plans/PR-Archive-Resolution-Audit-Zendesk-Plans.md`
- `plans/archive/PR-FAQ-Macro-Writeback-Approve-On-Publish-CAS.md`
- `plans/archive/PR-Resolution-Audit-Zendesk-Writeback.md`

## Mechanism

`git mv` the two plan docs into `plans/archive/` (git records them as 100%
renames, preserving history), then `python scripts/archive_plans.py index`
regenerates `plans/INDEX.md`. No content changes; pure relocation.

## Intentional

- Lane-scoped to the two just-merged plans, not the full 92-plan sweep. The
  sweep is blocked by a reused-slice-name collision and is a repo-wide
  maintenance task for the plan-hygiene owner, not this lane.

## Deferred

- Full `scripts/archive_plans.py archive` sweep (92 root plans) after the
  `PR-Deflection-Question-Label-Merge.md` name collision is resolved.

Parked hardening: none.

## Verification

- `python scripts/archive_plans.py index` -- regenerated INDEX (1282 plans).
- `git diff --cached --name-status` -- two R100 renames + INDEX modification.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 4 |
| `plans/PR-Archive-Resolution-Audit-Zendesk-Plans.md` | 54 |
| `plans/archive/PR-FAQ-Macro-Writeback-Approve-On-Publish-CAS.md` | 0 |
| `plans/archive/PR-Resolution-Audit-Zendesk-Writeback.md` | 0 |
| **Total** | **58** |
