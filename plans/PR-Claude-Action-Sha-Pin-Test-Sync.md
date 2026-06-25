# PR-Claude-Action-Sha-Pin-Test-Sync

## Summary
Sync the Claude workflow security test with the SHA-pinned Claude action already present on `main` after the Dependabot security update.

## Intentional
- Test-only update; no workflow behavior or action version change.
- Keeps the security assertion strict by matching the exact SHA currently pinned in `.github/workflows/claude.yml`.
- Unblocks Dependabot PR pre-push checks that currently fail on the stale expected Claude action SHA.

## Deferred
- None.

## Parked hardening
- None.

## Verification
- CI should run the pre-push audit and `tests/test_claude_workflow_security.py` against this branch.
- Local verification was not run from this projectless workspace because the repo is not checked out locally.

## Diff size
2 files: one test expectation update and one plan document.
