# PR-Claude-Workflow-SHA-Pin-Test-Robust

## Why this slice exists

`tests/test_claude_workflow_security.py::test_claude_actions_are_sha_pinned`
hard-codes the exact commit SHA each pinned action must carry. When Dependabot
PR #1818 bumped `anthropics/claude-code-action` in
`.github/workflows/claude.yml` to a new SHA, the workflow was updated but the
test fixture was not, so the assertion now fails on `main` and in
`pre-push-audit` on every open PR (the PR merge-ref includes main's bumped
workflow). The test is a tripwire that breaks on its own success criterion --
every legitimate, already-reviewed action bump turns it red until someone hand-
edits the fixture.

Root cause: the test encodes the wrong invariant. Its purpose (per its name) is
"actions are SHA-pinned" -- i.e. pinned to an immutable 40-char commit SHA
rather than a mutable tag/branch -- not "pinned to this one specific SHA".
This fixes the root by asserting the security property (each required action is
present and pinned to a 40-char hex SHA) instead of a rotating literal, so
routine Dependabot SHA bumps pass while a tag/branch pin (`@v1`, `@main`) still
fails.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Rework `test_claude_actions_are_sha_pinned` in
   `tests/test_claude_workflow_security.py` to assert each required action
   (`actions/checkout`, `anthropics/claude-code-action`) is present in
   `.github/workflows/claude.yml` and pinned to a 40-char hex commit SHA, via
   regex, instead of two hard-coded SHA string matches.
2. Add this plan doc.

No workflow change: `.github/workflows/claude.yml` is already correctly
SHA-pinned (Dependabot #1818); only the stale test fixture is wrong.

### Review Contract

Acceptance criteria:

- The test passes against the current `.github/workflows/claude.yml`
  (claude-code-action pinned to its post-#1818 SHA).
- A mutable ref (`@v1`, `@main`) or a missing required action still fails the
  test.
- A pinned SHA left only in a comment/example, or a look-alike fork such as
  `some-fork/actions/checkout`, does not satisfy the check (it is anchored to
  real `uses:` steps and matches the exact `owner/repo`).
- `test_claude_oidc_job_is_owner_gated` is unchanged.

Affected surfaces:

- CI fixture for the claude.yml security posture test (test-only change).

Risk areas:

- Loosening a security tripwire: mitigated by still requiring a 40-char hex SHA
  (not any tag) and still requiring each named action to be present, so the
  "no mutable action ref" property is preserved, only the specific-SHA coupling
  is dropped.

Triggered reviewer rules:

- R2 Test evidence
- R3 Security/auth
- R8 CI/workflow safety
- R14 Codebase verification

### Files touched

- `plans/PR-Claude-Workflow-SHA-Pin-Test-Robust.md`
- `tests/test_claude_workflow_security.py`

## Mechanism

The rewritten test extracts the action ref from each real `uses:` step
(`_USES_RE`, anchored to `uses:` line starts so a pinned SHA left in a comment
or example cannot satisfy the check), then for each required action matches the
exact `owner/repo` (so a look-alike such as `some-fork/actions/checkout` cannot
stand in for `actions/checkout`) and asserts every matching `uses:` ref is
pinned to a 40-char hex SHA (`_SHA_RE`). A 40-char SHA (old or new) passes; a
tag (`@v1`), branch (`@main`), look-alike fork, or absent action fails. This
preserves the original dual intent (the actions are used AND immutably pinned)
without coupling the test to a value Dependabot rotates.

## Intentional

- Test-only change: no product code, no workflow edit. The workflow is already
  pinned correctly; only the fixture encoded the wrong invariant.
- Keeps the named-action presence checks so removing `claude-code-action` or
  `checkout` still fails the test, matching the prior behavior.
- No `plans/INDEX.md` entry: in-flight plans live in the `plans/` root; INDEX is
  the archive index, updated when the slice is archived after merge.

## Deferred

- None. (Other stale hard-coded-SHA fixtures, if any surface later, would be
  separate slices.)

Parked hardening: none.

## Verification

- Direct execution (local; pytest not installed here): importing
  `tests/test_claude_workflow_security.py` and calling both test functions
  passes against the current `.github/workflows/claude.yml`.
- Negative check (local): `re.fullmatch(r"[0-9a-f]{40}", ref)` returns a match
  for both the old and new 40-char SHAs and `None` for `v1` / `main`.
- Format gates (local): `scripts/sync_pr_plan.py --check` on this plan doc,
  `scripts/check_ascii_python.sh`, and `git diff --check` all pass.
- CI: `pre-push-audit` / the unit suite runs the test and it passes, clearing
  the failure now seen on `main` and on the other open PRs.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Claude-Workflow-SHA-Pin-Test-Robust.md` | 120 |
| `tests/test_claude_workflow_security.py` | 34 |
| **Total** | **154** |
