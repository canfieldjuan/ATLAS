# PR-Archive-On-Merge-Ritual

## Why this slice exists

#1319 item 1 asked for archive-on-merge to be a **mechanical** step in the
teardown ritual, not a one-time sweep. Slices 1-3 shipped the tooling
(`scripts/archive_plans.py`), swept the 878-plan backlog, and added a
non-blocking advisory -- but the actual ritual wiring was never added, so newly
merged plans still land in the `plans/` root and accumulate until someone runs the
tool. This slice closes that gap: it adds the archive step to `AGENTS.md` §1g so the
retirement of a plan is tied to the same merge event that retires its branch and
worktree.

## Scope (this PR)

Ownership lane: governance/plans-archive
Slice phase: Workflow/process

1. Add a plan-archival step to the "Teardown on merge" ritual in `AGENTS.md` §1g.

### Files touched

- `AGENTS.md`
- `plans/PR-Archive-On-Merge-Ritual.md`

## Mechanism

The step is a targeted `git mv plans/PR-<Slice>.md plans/archive/` followed by
`python scripts/archive_plans.py index` to rebuild `plans/INDEX.md`, landed on
`origin/main` as a housekeeping commit (or folded into the next branch off `main`).
No tool change is needed: `git mv` is inherently scoped to the named plan, and the
existing `index` mode regenerates the navigation aid.

## Intentional

- **Targeted move, never bulk during teardown.** The ritual moves only the merged
  plan by name and explicitly forbids `archive_plans.py archive` (bulk) here, because
  bulk would sweep concurrent sessions' still-in-flight plans out of the root. Bulk
  `archive` stays a deliberate solo catch-up tool, not a per-merge step.
- **Nothing deprecated.** The bulk `archive` mode (catch-up/cleanup) and the
  non-blocking `check` advisory (backstop that nudges when the ritual is missed) both
  remain useful and complementary to the now-mechanical ritual step; neither is
  superseded.
- **Docs-only.** The capability already exists; this slice only ties it to the merge
  event, so the change is contained to the `AGENTS.md` contract.

## Deferred

- **Item 2 refinement:** a per-plan *merged-PR-status* detector (the spec's original
  item 2) rather than the current count-threshold advisory -- now lower priority since
  the ritual prevents accumulation at the source.
- **Item 4 refinement:** `drain_hardening.py` currently archives to a markdown file
  (`docs/technical-debt/hardening-archive.md`); the issue named the CSV debt register
  (`docs/technical-debt/debt-register-latest.csv`) as the correct archive tier. A
  retarget is a separate follow-up.
- **#1319 item 5:** segregating one-shot scripts into `scripts/oneshot/`.
- Parked hardening: none.

## Verification

```bash
python scripts/archive_plans.py index   # regenerates plans/INDEX.md (existing mode)
bash scripts/local_pr_review.sh          # gates pass; docs-only change
```

Verified locally: the referenced commands exist and run; the change is confined to the
`AGENTS.md` teardown section.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 17 |
| `plans/PR-Archive-On-Merge-Ritual.md` | 74 |
| **Total** | **91** |
