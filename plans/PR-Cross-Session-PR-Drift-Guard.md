# PR-Cross-Session-PR-Drift-Guard

## Why this slice exists

Multiple sessions are now working around AI Content Ops at the same time. The
existing local PR review catches plan shape, diff size, manifest sync, and
whitespace issues, but it does not warn when a branch overlaps another open PR
or when `main` has landed changes to the same files since the branch forked.

That gap let the landing-page repair-attempt contract work drift across
sessions until a later PR superseded an earlier one. This slice adds a cheap
mechanical guard before PR open/update and a lightweight lane contract so
parallel sessions can stay scoped.

## Scope (this PR)

Ownership lane: workflow/pr-drift-guard

1. Add a local audit that compares the current branch's changed files with
   changes already landed on the base branch since the merge-base.
2. Add an optional GitHub-backed open-PR overlap check when `gh` is available.
3. Require newly added PR plan docs to declare an ownership lane.
4. Compare the current branch's ownership lane against open PR bodies when
   GitHub metadata is available.
5. Keep open-PR file overlap advisory while blocking stale-base file overlap and
   same-lane overlap.
6. Wire the audit into `scripts/local_pr_review.sh`.
7. Cover the audit behavior with fixture tests.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Cross-Session-PR-Drift-Guard.md` | Plan doc for this guard slice. |
| `scripts/audit_pr_session_drift.py` | Adds the cross-session branch/open-PR overlap audit. |
| `scripts/local_pr_review.sh` | Runs the new audit as part of local PR review. |
| `tests/test_audit_pr_session_drift.py` | Fixture tests for the audit CLI contract. |
| `tests/test_local_pr_review.py` | Confirms local review invokes the audit when present. |

## Mechanism

The audit resolves `merge-base(HEAD, base-ref)` and compares:

- current branch changed files: `git diff --name-only <base>...HEAD`
- base changes since the branch forked: `git diff --name-only <base>..<base-ref>`
- open PR files from `gh pr list` / `gh pr view` when GitHub metadata is
  available
- ownership lanes declared as `Ownership lane: <lane>` in changed PR plan docs
  and mirrored PR bodies

Exact file-path overlap with base changes exits nonzero and tells the builder to
rebase. File overlap with another open PR is advisory because parallel slices can
legitimately touch different regions of the same file. Any open PR with the same
ownership lane exits nonzero, even when the files differ.

If `gh` is unavailable, unauthenticated, or cannot read one open PR's metadata,
that GitHub metadata is explicitly skipped while local base-overlap and local
ownership-lane checks still run. Self-PR detection uses branch name first and
falls back to head SHA for detached checkouts.

## Intentional

- This is stale-base file detection plus explicit lane detection, not semantic
  duplicate-constant analysis. Open-PR file overlap is printed as a warning
  because same-file edits can still be legitimate.
- The GitHub PR sweep is fail-open when `gh` is unavailable so CI and offline
  local work do not block on network/auth state.
- The guard is repo-wide instead of Content-Ops-only because exact changed-file
  overlap is a general conflict signal and cheap to compute.

## Deferred

- A later semantic-contract slice can add targeted checks for duplicated
  business constants such as repair attempt caps.
- A later workflow slice can add a separate command that comments the overlap
  report on PRs if reviewers want GitHub-visible drift diagnostics.

## Verification

- Focused pytest: tests/test_audit_pr_session_drift.py and
  tests/test_local_pr_review.py.
- Local PR wrapper: `bash scripts/local_pr_review.sh origin/main`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| Audit script | ~335 |
| Local review hook | ~10 |
| Tests | ~390 |
| Total | ~800 |

This is over the soft 400 LOC target because new audit scripts require fixture
tests for the CLI contract, and the GitHub/lane fixture paths are the bulk of
the size.
