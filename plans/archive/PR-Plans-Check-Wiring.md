# PR-Plans-Check-Wiring

## Why this slice exists

Slice 3 of #1319. Slices 1-2 shipped the `archive_plans.py` tool and swept the
878-plan backlog into `plans/archive/`, leaving the `plans/` root empty. The tool's
`check` mode (a non-blocking threshold warning) was deliberately *not* wired into any
runner until the sweep landed, because pre-sweep it would have warned on every PR
against the full 878-plan backlog. The backlog is now cleared, so the threshold is
finally meaningful: this slice activates the forcing function by surfacing `check` as
an advisory in `local_pr_review.sh`.

## Scope (this PR)

Ownership lane: governance/plans-archive
Slice phase: Workflow/process

1. Add a non-blocking "Plans archive backlog" advisory step to
   `scripts/local_pr_review.sh` that runs `archive_plans.py check` and prints its
   output without affecting the pass/fail count.
2. Cover it with two fixtures in `tests/test_local_pr_review.py` (advisory runs when
   the script is present and stays non-blocking; SKIPs cleanly when absent).
3. Enroll `tests/test_local_pr_review.py` in `.github/workflows/pre_push_audit.yml` so
   the fixtures run in CI (the workflow already runs `local_pr_review.sh`, so its tests
   belong there).

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Plans-Check-Wiring.md`
- `scripts/local_pr_review.sh`
- `tests/test_local_pr_review.py`

## Mechanism

The advisory runs after the last real check (`git diff --check`) and before the
summary. It is intentionally *not* a `run_check` step, so it never touches the
`failures` tally: it prints `==> Plans archive backlog (advisory, non-blocking)`,
runs `python scripts/archive_plans.py check || true`, and echoes a one-line nudge to
run `archive_plans.py archive`. The `|| true` keeps `set -e` happy, and the
`[ -f scripts/archive_plans.py ]` guard makes it SKIP cleanly where the tool is
absent (e.g. the test fixture repos). The default threshold (50) is unchanged.

## Intentional

- **Advisory, not a gate.** Wiring `check` as a `run_check` would either always PASS
  (it exits 0) or, if hardened to fail, add per-PR friction every time the root
  creeps past the threshold. A pure advisory surfaces the nudge without ever
  blocking a PR -- the "script + threshold warn" forcing function chosen in slice 1.
- **`local_pr_review.sh`, not `pre_push_audit.sh`.** The local review bundle is the
  right altitude for a human-facing nudge; the CI pre-push audit is for hard gates.
- **Guarded + `|| true`.** Keeps the existing `set -euo pipefail` contract intact and
  the script portable to checkouts (or fixtures) that lack the tool.

## Deferred

- **Enrolling the sibling PR-tooling meta-tests** (`test_pre_push_audit.py`,
  `test_new_pr_plan.py`, `test_sync_pr_plan.py`, `test_install_local_pr_hook.py`,
  `test_push_pr_wrapper.py`) in CI -- a pre-existing gap; this slice only enrolls the
  file it touches rather than batch-enrolling tests it has not validated here.
- **#1319 items 4-5:** mechanical `HARDENING.md` drain (size/age threshold) and
  segregating one-shot scripts into `scripts/oneshot/` -- independent follow-ups.
- Parked hardening: none.

## Verification

```bash
python -m pytest tests/test_local_pr_review.py -q     # 9 passed (7 existing + 2 new)
bash scripts/local_pr_review.sh --current-pr-body-file <body>   # advisory prints, review passes
```

Verified locally: 9/9 fixtures pass; the advisory prints `OK: 0 plan doc(s) in root`
on the cleared backlog and never changes the review's exit status.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 8 |
| `plans/PR-Plans-Check-Wiring.md` | 83 |
| `scripts/local_pr_review.sh` | 12 |
| `tests/test_local_pr_review.py` | 30 |
| **Total** | **133** |
