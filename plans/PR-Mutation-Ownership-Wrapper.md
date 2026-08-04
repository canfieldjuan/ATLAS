# PR-Mutation-Ownership-Wrapper

## Why this slice exists

The mechanical-enforcement audit found that Atlas has a session PR ownership
helper, but the prescribed PR mutation wrapper does not call it before editing
an existing PR body. That leaves the "do not touch another session's PR" rule as
a manual memory step at the exact point where the wrapper already has the PR
number, branch, and head SHA needed to enforce it.

Audit finding:

- `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` classified
  "PR ownership before mutation is guarded" as `MANUAL_HELPER` because
  `scripts/check_session_pr_ownership.py` existed but no invocation was found in
  `scripts/open_pr.sh`, `scripts/push_pr.sh`, or `scripts/local_pr_review.sh`.

### Problem-derived contract

- Root cause: existing-PR body edits happen through `scripts/open_pr.sh` after
  the wrapper discovers the target PR, but the discovered PR identity is not
  checked against the session-scoped ownership file before `gh pr edit` mutates
  GitHub.
- Correct fix must touch/change: `scripts/open_pr.sh` must invoke
  `scripts/check_session_pr_ownership.py` on the existing-PR edit path after
  discovering the PR number/head and before final review or `gh pr edit`;
  `tests/test_open_pr_wrapper.py` must prove an owned PR can proceed and an
  unowned PR fails before GitHub mutation.
- Must not change: do not change product code, branch protection, reviewer
  policy, new-PR creation semantics, draft consent semantics, local review
  contents, `push_pr.sh` behavior where no PR number is known, or any EOM /
  Dependabot / non-owned PR lane.

## Scope (this PR)

Ownership lane: dev-workflow/pr-mutation-ownership
Slice phase: Workflow/process

1. Add a wrapper-level ownership guard for existing PR body edits in
   `scripts/open_pr.sh`.
2. Add fixture tests proving the guard runs before mutation and blocks `gh pr
   edit` when session ownership fails.

### Review Contract

- Acceptance criteria:
  1. Existing-PR body updates through `scripts/open_pr.sh` call
     `scripts/check_session_pr_ownership.py --pr <number> --branch <branch>
     --head-sha <reviewed-head>` before `gh pr edit`.
  2. A failing ownership guard stops the wrapper before final local review,
     `gh pr edit`, or body stdin capture; settled by
     `tests/test_open_pr_wrapper.py::test_open_pr_existing_pr_ownership_guard_failure_blocks_before_edit`.
  3. An owned existing PR still edits via stdin and preserves the existing body
     publication contract; settled by
     `tests/test_open_pr_wrapper.py::test_open_pr_edit_passes_body_via_stdin_not_path`.
  4. New-PR creation remains on the existing create path because there is no PR
     number to check before creation; the wrapper's existing branch/repo/base,
     final-review, and post-create identity checks are unchanged.
- Reachability proof: `bash scripts/open_pr.sh BODY_FILE` is exercised by
  `tests/test_open_pr_wrapper.py`; the observable effect is the fake `gh pr
  edit` log or the absence of that log when the guard fails.
- Affected surfaces: `scripts/open_pr.sh`, `tests/test_open_pr_wrapper.py`, and
  this plan.
- Risk areas: shell argument ordering, PR edit admission boundary, session-state
  path resolution, existing create/update behavior.
- Reviewer rules triggered: R1, R2, R6, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: existing PR body mutation via `scripts/open_pr.sh`.
- Replaced-path behaviors: existing-PR edits previously checked branch/repo/head
  identity but did not check session ownership before `gh pr edit`.
- Guard-relevant fields: PR number, current branch, reviewed head SHA, and
  `ATLAS_SESSION_STATE_FILE` or the helper's default state-file fallback.
- Caller x input shape: `bash scripts/open_pr.sh BODY_FILE` with an already-open
  PR for the current branch and no create args.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: no deployed config; local
  `ATLAS_SESSION_STATE_FILE` may point at a session-specific state file and the
  helper defaults to `SESSION_STATE.local.md`.
- Explicit value probe: fixture sets `ATLAS_SESSION_STATE_FILE` to an owned PR
  state file and verifies the edit proceeds.
- Absent value probe: existing helper behavior remains unchanged; absence of a
  readable state file fails the guard.
- Default-session/default-context probe: wrapper invokes the existing helper so
  default fallback semantics stay owned by `scripts/check_session_pr_ownership.py`.
- Side-effect ordering: guard failure is tested to happen before final local
  review, `gh pr edit`, or PR-body stdin capture.

### Files touched

- `plans/PR-Mutation-Ownership-Wrapper.md`
- `scripts/open_pr.sh`
- `tests/test_open_pr_wrapper.py`

## Mechanism

`scripts/open_pr.sh` already resolves the current branch, verifies the pushed
head, discovers the existing PR snapshot, and confirms the PR head matches the
reviewed head. This slice adds one helper call at that existing-PR boundary:

```bash
python scripts/check_session_pr_ownership.py \
  --pr "$existing_pr_number" \
  --branch "$branch" \
  --head-sha "$reviewed_head"
```

The helper reads the current session-state file, rejects must-not-touch PRs,
and rejects PRs absent from the owned/may-touch set. Because the call happens
before the final local review and before `gh pr edit`, a wrong-lane PR cannot
burn review or mutate GitHub through the wrapper.

## Intentional

- No `push_pr.sh` guard in this slice because that wrapper normally has branch
  metadata but not a PR number/head snapshot; forcing PR lookup into push would
  widen the mutation surface and duplicate `open_pr.sh` identity logic.
- No new-PR creation guard in this slice because there is no PR number before
  creation. Existing create-path checks for branch publication, repo/base
  targeting, final local review, draft consent, and post-create identity remain
  unchanged.

## Deferred

- New-PR pre-create ownership admission, if desired, should be a separate
  helper mode with its own contract for "planned branch may open one PR."

Parked hardening: none.

## Verification

- `python -m pytest tests/test_open_pr_wrapper.py tests/test_check_session_pr_ownership.py`
  - 63 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Mutation-Ownership-Wrapper.md` | 152 |
| `scripts/open_pr.sh` | 10 |
| `tests/test_open_pr_wrapper.py` | 62 |
| **Total** | **224** |
