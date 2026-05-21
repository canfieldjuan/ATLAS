# PR-Ready-Review-PR-Docs

## Why this slice exists

The repo workflow docs still told builders to open draft PRs until reviewer
LGTM. That conflicts with the current review flow because automated review
tools do not review draft PRs, which delays feedback and burns operator time.

This docs-only slice updates the source-of-truth workflow to open PRs ready for
review by default.

## Scope (this PR)

1. Update `AGENTS.md` to replace draft-first PR guidance with ready-for-review
   guidance.
2. Update `CLAUDE.md` to mirror the same workflow summary.
3. Keep draft PRs allowed only when the operator explicitly asks for draft
   mode.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Ready-Review-PR-Docs.md` | Plan doc for this docs-only workflow correction. |
| `AGENTS.md` | Change builder PR workflow from draft-first to ready-for-review by default. |
| `CLAUDE.md` | Mirror the ready-for-review default in the workflow highlights. |

## Mechanism

This is a documentation-only correction. It changes the PR workflow language
from “open as draft” to “open ready for review by default,” and explicitly says
draft PRs require an operator request.

## Intentional

- No code changes.
- No script changes.
- No GitHub settings changes.
- No changes to the reviewer verdict contract.

## Deferred

- Updating any external agent/plugin documentation outside this repository.
- Automating a CI guard that rejects draft PRs.

## Verification

- Draft-instruction search over `AGENTS.md` and `CLAUDE.md` for "Open as
  draft", "open as draft", "draft until", and "--draft" -> passed with 0
  matches.
- `git diff --check` -> passed with 0 whitespace errors.
- `bash scripts/local_pr_review.sh origin/main` -> passed 3/3 top-level
  checks: pre-push audit wrapper, plan/code consistency, and `git diff
  --check`.
- The pre-push audit wrapper inside local review reported all 8 internal checks
  passed: MCP tool counts, MCP port assignments, MCP tool-name inventories,
  extracted manifest sync, plan shape, plan files touched, plan diff size, and
  ASCII Python policy.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| Workflow docs | ~10 |
| Total | ~65 |
