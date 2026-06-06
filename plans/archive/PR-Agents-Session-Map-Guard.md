# PR-Agents-Session-Map-Guard

## Why this slice exists

PR-Agents-Session-PR-Map made ownership mapping mandatory, but the deferred
tooling gap remains: before a builder inspects, updates, force-pushes, or
merges a PR, there should be a repeatable local command that checks the target
PR is actually listed as owned in the session map.

This slice adds that local guard so ownership checks are mechanical instead of
purely memory-based after compaction.

## Scope (this PR)

Ownership lane: repo-workflow/session-discipline
Slice phase: Workflow/process

1. Add a stdlib script that validates a PR number, branch, and optional head
   SHA against `SESSION_STATE.local.md`.
2. Fail closed when the state file is missing, the PR is listed as "must not
   touch", the PR is not listed as owned, the branch mismatches, the expected
   head SHA is omitted, or the expected head SHA mismatches.
3. Add focused tests for the positive path and each failure branch.
4. Document the guard in `AGENTS.md` as the command to run before PR mutation.

### Files touched

- `plans/PR-Agents-Session-Map-Guard.md`
- `scripts/check_session_pr_ownership.py`
- `tests/test_check_session_pr_ownership.py`
- `AGENTS.md`

## Mechanism

The script accepts explicit PR metadata:

```bash
python scripts/check_session_pr_ownership.py \
  --pr 1189 \
  --branch claude/pr-agents-session-pr-map \
  --head-sha <sha>
```

It parses the local session map sections, treats "must not touch" as a hard
block, accepts ownership only from "Owned Active PR" or "PRs This Session May
Touch", and compares branch/head values when present. If the map records an
expected head SHA, the caller must supply `--head-sha`; otherwise the command
fails closed. It prints deterministic errors and exits non-zero on ambiguity.

## Intentional

- The script does not call GitHub. Builders should pass metadata from
  `gh pr view` so the guard remains easy to test and can run without network.
- This is a local operator/builder guard, not a CI gate. CI does not have the
  ignored session map.
- The guard is not wired into every script yet. It gives the builder a concrete
  command to run before PR mutation.

## Deferred

- Parked hardening: none.
- A future shell integration can wrap merge/update commands and call this guard
  automatically.

## Verification

- `python -m py_compile scripts/check_session_pr_ownership.py tests/test_check_session_pr_ownership.py`
- `python -m pytest tests/test_check_session_pr_ownership.py -q`
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/agents-session-map-guard.md`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Guard script | 158 |
| Tests | 173 |
| AGENTS docs | 10 |
| **Total** | **424** |

The slice is slightly over the soft cap because this is a checker surface:
every failure branch needs a focused negative fixture so the guard cannot
silently false-green ambiguous ownership.
