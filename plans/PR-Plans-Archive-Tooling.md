# PR-Plans-Archive-Tooling

## Why this slice exists

`plans/` holds 877 `PR-*.md` docs and grows by one per PR, forever — plan docs ship
*with* their PR, so everything on `main` is from a merged PR and the directory is an
unstructured, ever-growing archive. The cost is orientation: a fresh or compacted
session that lists `plans/` faces the whole history with no navigation aid. This is
the first slice of #1319 (bound governance-file growth): the tooling to give the
directory a lifecycle. It deliberately does **not** perform the 877-file bulk move
(see Deferred) so this PR stays small and reviewable.

At 442 LOC it runs ~10% over the 400 soft cap: a review-driven collision-safety fix
to `archive_plans` (refuse to overwrite an already-archived slice name) added code
plus two fixtures. The overage is a data-loss guard, not new scope.

## Scope (this PR)

Ownership lane: governance/plans-archive
Slice phase: Workflow/process

1. Add `scripts/archive_plans.py` with three modes: `archive` (move root
   `PR-*.md` into `plans/archive/` + regenerate plans/INDEX.md), `index`
   (regenerate the index only), and `check` (non-blocking warning when the root
   exceeds a threshold; always exits 0).
2. Ship `tests/test_archive_plans.py` with §3h coverage and enroll it in CI beside
   the other CI-tooling fixtures.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Plans-Archive-Tooling.md`
- `scripts/archive_plans.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_archive_plans.py`

## Mechanism

Pure functions (`root_plan_files`, `archived_plan_files`, `plan_metadata`,
`build_index`, `over_threshold`) make the behavior testable on a temp tree.
`archive` moves each root `PR-*.md` into `plans/archive/` via a filesystem rename
(committed with `git add -A`, which git records as a rename so history is
preserved) and writes INDEX.md — one line per archived plan with its ownership
lane and slice phase parsed from the doc. `check` is the forcing function: it
prints a warning over threshold and returns 0, so it never fails a PR.

The archive is flat (all under `plans/archive/`) rather than bucketed by date: the
visible git history is compressed (every plan reads as added within one week), so
date-based quarter buckets would put everything in one bucket anyway. The generated
INDEX.md is the navigation aid instead.

## Intentional

- **`check` is non-blocking (exits 0).** A hard CI gate would fire on every PR
  (each adds its plan to the root), so the threshold is a nudge, not a wall —
  matching the chosen "script + threshold warn" forcing function.
- **The 877-plan bulk sweep is not in this PR.** Running `archive` over the backlog
  is an ~880-file rename better reviewed on its own; this slice ships the verified
  tool first.
- **`archive` is meant to run on a branch off `main`**, where every plan is merged.
  It is not run on an in-flight feature branch whose unmerged plan still sits in the
  root (that plan is active, not archivable).
- **`archive` fails safe on slice-name collisions.** Plan filenames are
  human-chosen, not immutable PR numbers, so a reused name could collide with an
  already-archived plan; `Path.rename` would silently overwrite it. `archive_plans`
  detects collisions up front and raises (moving nothing) rather than destroying the
  older plan.
- **The fixture is enrolled in `run_extracted_pipeline_checks.sh`**, beside the
  existing CI-tooling audit fixtures — shipping a fixture CI never runs would repeat
  the #1318 enrollment gap.

## Deferred

- **The bulk sweep** of the existing 877 plans (a separate, eyeballable run/PR).
- **Wiring `check` into `local_pr_review.sh`** — deferred until after the sweep, so
  the warning is meaningful rather than firing against the current full backlog on
  every PR.
- **#1319 items 4-5:** mechanical `HARDENING.md` drain (size/age threshold) and
  segregating one-shot scripts into `scripts/oneshot/` — independent follow-ups.
- Parked hardening: none.

## Verification

```bash
python -m pytest tests/test_archive_plans.py -q          # 8 passed
python scripts/archive_plans.py check                    # WARNING (877 > 50), exit 0
python scripts/archive_plans.py index                    # regenerates INDEX from archive
```

Verified locally: 8/8 fixtures pass (incl. collision-refusal + non-zero CLI exit);
`check` warns and exits 0; `archive`/`index` move + index round-trip on a temp tree.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Plans-Archive-Tooling.md` | 102 |
| `scripts/archive_plans.py` | 188 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_archive_plans.py` | 151 |
| **Total** | **446** |
