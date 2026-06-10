# PR-Archive-Remaining-Merged-Plans

## Why this slice exists

PRs #1324, #1325, #1326, #1329, #1331, #1333, and #1335 are already
squash-merged to `origin/main`, but their plan docs still live in the root
`plans/` directory. PR #1337 archived the S4 reviewer-workflow plan and
regenerated `plans/INDEX.md`, leaving an orientation mismatch: the index says
the root holds only in-flight slices, while seven merged workflow/operating
model plans are still there. This slice closes that housekeeping gap before the
next workflow slice adds live GitHub AI-thread reconciliation.

## Scope (this PR)

Ownership lane: dev-workflow/plan-archive-housekeeping
Slice phase: Workflow/process

1. Move the seven known merged root plan docs into `plans/archive/`.
2. Regenerate `plans/INDEX.md` so those moved plans are listed and the archive
   count matches the tracked archive.
3. Leave this in-flight plan in the root; the broad `archive_plans.py archive`
   helper is intentionally not used because it would archive the active plan
   too.

### Review Contract

- Acceptance criteria:
  - [ ] `plans/` root contains only this in-flight plan after the move.
  - [ ] The seven merged plan docs named in this slice are present under
        `plans/archive/` and absent from the root.
  - [ ] `plans/INDEX.md` is regenerated from the archive and includes the seven
        newly archived entries.
  - [ ] `python scripts/archive_plans.py check` reports one root plan doc
        (this slice) and stays under the threshold.
- Affected surfaces: developer workflow docs only (`plans/` lifecycle and
  archive index). No product, API, DB, auth, frontend, config, or runtime
  surface.
- Risk areas: accidentally archiving the active in-flight plan, losing a merged
  plan doc by overwrite, or leaving `plans/INDEX.md` stale.
- Reviewer rules triggered: R1 (requirements match), R10 (maintainability).

### Files touched

- `plans/INDEX.md`
- `plans/PR-Archive-Remaining-Merged-Plans.md`
- `plans/archive/PR-Archive-On-Merge-Ritual.md`
- `plans/archive/PR-Content-Ops-Claims-Map.md`
- `plans/archive/PR-Content-Ops-Content-PR.md`
- `plans/archive/PR-Content-Ops-Review-Vocabulary.md`
- `plans/archive/PR-Content-Ops-Triage-Experiment.md`
- `plans/archive/PR-Hardening-Drain.md`
- `plans/archive/PR-Plans-Check-Wiring.md`

## Mechanism

The implementation is mechanical documentation lifecycle work:

1. `git mv` each already-merged root plan doc into `plans/archive/`.
2. Run `python scripts/archive_plans.py index` to rebuild `plans/INDEX.md` from
   the tracked archive directory.
3. Verify with `find plans -maxdepth 1 -type f -name 'PR-*.md'` that the only
   root plan is this in-flight slice.

## Intentional

- Do not run `python scripts/archive_plans.py archive` in this branch. That
  command is appropriate on a clean main/housekeeping branch with no active
  plan, but here it would also move this slice's plan before the PR opens.
- No product tests are needed: this moves markdown plan docs and regenerates a
  markdown index only.

## Deferred

- PR-Reviewer-Live-AI-Thread-Reconciliation: add the CI-side GitHub-thread
  reconciliation gate that fetches live Codex/Copilot comments and compares
  them to the PR body's recorded AI reconciliation.

Parked hardening: none.

## Verification

- Ran the `scripts/archive_plans.py` index command -- regenerated
  `plans/INDEX.md` with 889 archived plans.
- Listed root plan files matching the PR-star markdown pattern -- only
  `plans/PR-Archive-Remaining-Merged-Plans.md` remains in root.
- Root/archive presence loop for the seven moved plans -- all seven reported
  `OK`, present under `plans/archive/` and absent from root.
- Ran the `scripts/archive_plans.py` check command -- reports `OK: 1 plan doc(s) in
  root (threshold 50).`
- Ran `scripts/sync_pr_plan.py` for
  `plans/PR-Archive-Remaining-Merged-Plans.md` -- updated plan from git diff.
- Ran `scripts/local_pr_review.sh` with the local PR body file in
  `tmp/pr-body-archive-remaining-merged-plans.md` -- passed advisory dirty
  review before commit. Clean committed-diff review runs again through
  `scripts/push_pr.sh` before push.
- Ran clean `scripts/local_pr_review.sh` with the local PR body file after
  commit -- passed; branch diff included 9 changed files, plan file claims and
  diff-size both matched exactly, cross-session drift checked 3 open PRs.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 9 |
| `plans/PR-Archive-Remaining-Merged-Plans.md` | 113 |
| `plans/archive/PR-Archive-On-Merge-Ritual.md` | 0 |
| `plans/archive/PR-Content-Ops-Claims-Map.md` | 0 |
| `plans/archive/PR-Content-Ops-Content-PR.md` | 0 |
| `plans/archive/PR-Content-Ops-Review-Vocabulary.md` | 0 |
| `plans/archive/PR-Content-Ops-Triage-Experiment.md` | 0 |
| `plans/archive/PR-Hardening-Drain.md` | 0 |
| `plans/archive/PR-Plans-Check-Wiring.md` | 0 |
| **Total** | **122** |
