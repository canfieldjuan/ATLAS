# PR-Dev-Workflow-Open-PR-Stdin-Wrapper

## Why this slice exists

Issue #1306 tracks PR-opening friction. The first pass added plan/push helpers,
but the GitHub-side open/update step is still unwrapped, so builders keep
hand-rolling `gh pr create --body-file <path>` after the push. In this sandbox
shape, direct file-path body reads can fail inside `gh`; the stable form is
`--body-file - < file`, where the shell opens the file and `gh` reads stdin.

This slice closes that remaining workflow gap with a tiny wrapper and makes the
builder contract explicit so the mistake is not reintroduced after compaction.

## Scope (this PR)

Ownership lane: dev-workflow/pr-friction
Slice phase: Workflow/process

1. Add a PR open/update helper that creates the current-branch PR when missing
   or updates the existing PR body when present.
2. Ensure the helper always passes the PR body through stdin, never as a file
   path argument to `gh`.
3. Update builder-facing docs to use the wrapper and forbid raw
   `gh pr create/edit --body-file <path>`.
4. Add focused tests that fake `gh`, assert argv shape, and prove the body
   content arrives on stdin for both create and edit flows.

### Review Contract

- Acceptance criteria:
  - [ ] `scripts/open_pr.sh` accepts a body file and defaults to the current
        branch.
  - [ ] Missing body files and direct body args fail with clear messages.
  - [ ] Create flow invokes `gh pr create ... --body-file -` and feeds body
        content on stdin.
  - [ ] Existing-PR flow invokes `gh pr edit <branch> --body-file -` and feeds
        body content on stdin.
  - [ ] AGENTS/bootstrap docs tell builders to use the wrapper and avoid raw
        direct-path `--body-file`.
- Affected surfaces: builder workflow scripts, builder session docs, workflow
  helper tests.
- Risk areas: accidentally bypassing body env/local review semantics, or
  making existing PR body updates accept create-only args silently.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Dev-Workflow-Open-PR-Stdin-Wrapper.md`
- `scripts/open_pr.sh`
- `tests/test_open_pr_wrapper.py`

## Mechanism

`scripts/open_pr.sh BODY_FILE [gh-pr-create-args...]` validates the body file,
rejects direct body args, resolves the current branch, and checks whether a PR
already exists for that branch:

- Existing PR: `gh pr edit "$branch" --body-file - < "$body_file"`.
- Missing PR: `gh pr create "$@" --body-file - < "$body_file"`.

That means the shell opens the local body file and `gh` only reads fd 0. Tests
copy the wrapper into a temporary git repo and place a fake `gh` first on
`PATH`; the fake records argv and stdin so regressions to `--body-file <path>`
fail without making network calls.

## Intentional

- This is a sibling `open_pr.sh` rather than folding PR creation into
  `push_pr.sh`. Keeping push/local-review and GitHub PR creation separate
  preserves the current single-local-review push semantics while still
  removing the brittle hand-written `gh pr create/edit` step.
- Existing-PR updates reject create-only args instead of guessing how to map
  them to `gh pr edit`; body update is the safe shared path, and AGENTS now
  names manual `gh pr edit` for title/base/label changes.

## Deferred

- None.

Parked hardening: none.

## Verification

- Pending before push: targeted pytest wrapper suite covering
  tests/test_open_pr_wrapper.py and tests/test_push_pr_wrapper.py.
- Pending before push: push through scripts/push_pr.sh with the PR body file.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 10 |
| `docs/SESSION_BOOTSTRAP.md` | 5 |
| `plans/PR-Dev-Workflow-Open-PR-Stdin-Wrapper.md` | 99 |
| `scripts/open_pr.sh` | 74 |
| `tests/test_open_pr_wrapper.py` | 168 |
| **Total** | **356** |
