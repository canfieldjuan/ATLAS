# PR-Security-Gitleaks-Precommit

## Why this slice exists

#1656's credential-rotation notes identify a prevention gap: Atlas has PR-time
and scheduled Gitleaks coverage, but no repo-provided local pre-commit path that
blocks staged secrets before they enter git history. The historical `.env`
exposure remains provider-rotation work, but the same class can recur if the
earliest guard is only CI-after-push.

Root cause: secret scanning is enforced after commits are already created, and
the repository does not provide an installable local hook contract. This slice
fixes the first no-cost prevention layer by adding a pre-commit configuration
that runs `gitleaks protect --staged` plus docs/tests that lock that behavior.
It does not rotate external provider credentials or change GitHub branch
protection; those remain separate operational follow-ups.

## Scope (this PR)

Ownership lane: security/gitleaks-precommit
Slice phase: Workflow/process

1. Add a repository `.pre-commit-config.yaml` with a local system hook that runs
   `gitleaks protect --staged --redact --verbose` and declares the required
   `pre-commit` version for hook-name stage support.
2. Document the local install/usage path in `docs/SECURITY_GUARDRAILS.md`.
3. Add focused workflow-policy tests that prove the hook scans staged content,
   redacts findings, does not receive filename subsets, and is CI-enrolled
   through the existing pre-push audit workflow.
4. Archive the merged #1824 structured logging plan doc as required teardown
   housekeeping.

### Review Contract

- Acceptance criteria:
  - [ ] The pre-commit hook uses the local/system `gitleaks` binary, avoiding a
        new remote hook dependency.
  - [ ] The hook runs `gitleaks protect --staged` so only staged changes gate a
        commit.
  - [ ] Findings are redacted in hook output.
  - [ ] The hook sets `pass_filenames: false` so pre-commit cannot narrow the
        scan to only the filenames it passes.
  - [ ] The config declares the minimum compatible `pre-commit` version for
        `stages: [pre-commit]`.
  - [ ] Docs explain the install command, the required local `gitleaks` binary,
        the required `pre-commit` version, and how this complements the
        existing CI guard.
  - [ ] The contract tests are already enrolled in pre-push audit CI.
- Affected surfaces: local developer hooks, security guardrail docs, pre-push
  audit tests, and plan archive housekeeping.
- Risk areas: making local development depend on an unavailable binary without
  documenting it, weakening existing CI expectations, or overstating provider
  credential rotation as complete.
- Reviewer rules triggered: R1, R2, R3, R11, R14.

### Files touched

- `.pre-commit-config.yaml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Security-Gitleaks-Precommit.md`
- `plans/archive/PR-Security-Structured-JSON-Logging.md`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

The new pre-commit config uses a `repo: local` hook with `language: system`.
That means Atlas does not fetch a remote hook repository or pin another
third-party package in this slice; operators install `pre-commit` and `gitleaks`
locally, then run `pre-commit install`.

The config declares `minimum_pre_commit_version: "3.2.0"` because
`stages: [pre-commit]` uses the hook-name stage spelling introduced in
pre-commit 3.2. The hook entry runs
`gitleaks protect --staged --redact --verbose`. `--staged` targets the commit
candidate, `--redact` keeps local hook output from echoing secret values, and
`pass_filenames: false` prevents pre-commit from passing a path subset that
could narrow the secret scan below the staged diff.

## Intentional

- This does not install hooks automatically; Git hooks live in `.git/` and must
  remain an explicit local developer setup step.
- This uses the local `gitleaks` binary instead of a remote pre-commit hook so
  the repo does not introduce another downloaded hook supply-chain edge.
- This does not claim the exposed historical credentials are rotated. It only
  prevents the same class from being committed again locally.

## Deferred

- #1656 follow-up: rotate/revoke the credentials exposed in historical commit
  `d63a9b77b9727766e14e523626c22dd6c1c80da8`.
- #1656 follow-up: make the GitHub Gitleaks PR scan a required branch
  protection status check; that repository setting is outside this code diff.

Parked hardening: none.

## Verification

- Focused security workflow tests: `15 passed in 0.09s`.
- Python compile check for touched tests: passed.
- Whitespace diff check: passed.
- Local review bundle: passed via `scripts/push_pr.sh` pre-push hook.

## Estimated diff size

| File | LOC |
|---|---:|
| `.pre-commit-config.yaml` | 11 |
| `docs/SECURITY_GUARDRAILS.md` | 13 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Security-Gitleaks-Precommit.md` | 115 |
| `plans/archive/PR-Security-Structured-JSON-Logging.md` | 0 |
| `tests/test_security_guardrails_workflow.py` | 24 |
| **Total** | **166** |
