# PR-Current-Head-Check-Readiness

## Why this slice exists

Recent PRs repeatedly showed stale red `live-reconciliation` / AI
reconciliation rows beside newer green current-head rows, which made the
watcher handoff report ambiguous readiness and forced patch-by-patch triage
instead of root-cause convergence.

### Problem-derived contract

- Root cause: PR watcher readiness already fetches check-runs for the final PR
  head, but non-managed `check_failures` / `check_pending` are still computed
  from broad `gh pr checks` rows. When GitHub reports duplicate rows for the
  same check name, an older failed row can keep the watcher red after the
  latest final-head run is green.
- Correct fix must touch/change: `scripts/pr_watcher.py` must derive active
  non-managed check blockers from the latest GitHub Actions check-run on the
  final PR head for each reported check name, while keeping shape/transport
  errors fail-closed. `tests/test_pr_watcher.py` must prove older failed
  duplicate rows do not mask the newer current-head green run and that a latest
  optional failure still blocks readiness.
- Must not change: Do not alter branch protection, required-check registry
  semantics, review-thread reconciliation, Codex review policy, merge
  authority, product behavior, or unrelated watcher/reporting lanes.

## Scope (this PR)

Ownership lane: dev-workflow/current-head-check-readiness
Slice phase: Workflow/process

1. Change PR watcher all-check readiness to collapse non-managed check state to
   final-head latest check-runs by check name.
2. Add regression coverage for stale duplicate check rows and active optional
   failures.

### Review Contract

- Acceptance criteria:
  - `scripts/pr_watcher.py` validates `gh pr checks` rows for shape but derives
    active non-managed failures/pending from latest final-head check-runs.
  - A duplicate optional check with an older failed run and newer successful
    final-head run produces `state == "ready_for_human_merge"` and empty
    `check_failures`.
  - A duplicate optional check whose latest final-head run failed still
    produces `state == "attention"` and the failed check name in
    `check_failures`.
  - Required/registry readiness remains governed by expected required/blocking
    contexts and latest final-head GitHub Actions check-runs.
- Reachability proof: `scripts/pr_watcher.py produce(...)` is exercised through
  `tests/test_pr_watcher.py`; observable effect is the emitted watcher status
  JSON state and `check_failures` / `readiness.required_*` fields.
- Affected surfaces: `scripts/pr_watcher.py` watcher readiness producer and
  `tests/test_pr_watcher.py` watcher fixtures.
- Risk areas: stale duplicate rows, optional check regressions, malformed
  check-row fail-closed behavior, required-check registry behavior.
- Reviewer rules triggered: R1, R2, R6, R8, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: PR watcher check-state normalizer in
  `scripts/pr_watcher.py`.
- Replaced-path behaviors: Non-managed `gh pr checks` failure/pending rows are
  no longer treated as active blockers when a newer final-head check-run for
  the same name supersedes them.
- Guard-relevant fields: `name`, `bucket`, `status`, `conclusion`,
  `started_at`, and GitHub Actions `app.id`.
- Caller x input shape: watcher config for one owned PR; GitHub CLI JSON for
  PR checks and commit check-runs.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no config value change.
- Explicit value probe: N/A - no config value change.
- Absent value probe: N/A - no config value change.
- Default-session/default-context probe: watcher default fake config is
  exercised by `tests/test_pr_watcher.py`.
- Side-effect ordering: watcher still reads PR/check/review state and writes a
  status snapshot only; it does not merge or mutate PRs.

### Files touched

- `plans/PR-Current-Head-Check-Readiness.md`
- `scripts/pr_watcher.py`
- `tests/test_pr_watcher.py`

## Mechanism

The watcher keeps `gh pr checks` as the reported check-name inventory and shape
validation source. After the final PR head is known, it fetches commit
check-runs for that exact SHA, collapses GitHub Actions runs by check name using
the latest `started_at`, and computes both required/blocking readiness and
non-managed `check_failures` / `check_pending` from that collapsed current-head
map.

## Intentional

- No branch protection or required-check registry changes; this is a readiness
  observation fix only.
- Non-GitHub Actions check-runs remain outside the provenance-backed readiness
  source because the existing required-check guard already trusts GitHub Actions
  app provenance for CI gates.

## Deferred

None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_pr_watcher.py` - 75 passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-current-head-check-readiness.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-body-current-head-check-readiness.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Current-Head-Check-Readiness.md` | 129 |
| `scripts/pr_watcher.py` | 28 |
| `tests/test_pr_watcher.py` | 91 |
| **Total** | **248** |
