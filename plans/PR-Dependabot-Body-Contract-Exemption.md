# PR-Dependabot-Body-Contract-Exemption

## Why this slice exists

Dependabot PRs in the operator-assigned security/update ranges #1776-#1790 and
#1911-#1919 are blocked by the `pr-body-contract` workflow because generated
Dependabot bodies do not include Atlas plan documents. That check is still the
right contract for human and agent-authored PRs, but it turns routine
dependency/security bumps into metadata work instead of letting their real CI
decide mergeability.

Root cause: `scripts/audit_pr_body.py` only understands the Atlas human PR body
shape and receives no PR author context, so bot-authored dependency PRs fail the
contract even when their diffs and package checks are otherwise valid.

## Scope (this PR)

Ownership lane: dependency-maintenance/pr-body-contract
Slice phase: Workflow/process

1. Teach the PR body audit to accept known Dependabot author logins as an
   explicit exemption.
2. Pass `github.event.pull_request.user.login` from the workflow into the audit.
3. Add focused regression coverage proving Dependabot is exempt while malformed
   non-Dependabot PR bodies still fail.

### Review Contract

Acceptance criteria:

- Dependabot-authored PRs can pass `pr-body-contract` without a plan-shaped body.
- Human and agent-authored PRs still require the existing plan-line,
  slice-phase, why paragraph, and required-section contract.
- The exemption is based on the GitHub PR author login, not on body text or
  changed files.
- Existing `audit_pr_body(body, root=...)` callers keep the same default
  behavior.

Affected surfaces:

- PR body contract CI workflow.
- PR body audit script and its regression tests.

Risk areas:

- Over-broad bot exemption that would weaken the review contract for normal PRs.
- Required check churn on queued Dependabot PRs.

Reviewer rules triggered: R2, R10, R14.

### Files touched

- `.github/workflows/pr_body_contract.yml`
- `plans/PR-Dependabot-Body-Contract-Exemption.md`
- `scripts/audit_pr_body.py`
- `tests/test_audit_pr_body.py`

## Mechanism

The PR body workflow continues writing the PR body to a temporary Markdown
file, then passes the GitHub PR author login through the new PR-author CLI
option. The audit has a small allowlist for Dependabot identities
(`app/dependabot`, `dependabot`, and `dependabot[bot]`). If the author is
Dependabot, the CLI returns success before body-shape validation; otherwise it
runs the existing validation unchanged.

The pure `audit_pr_body(body, root=...)` function remains strict. Tests cover the
pure strict path, the author detector, the CLI exemption, and the CLI failure
path for a normal author with the same invalid body.

## Intentional

- The exemption does not inspect dependency diffs. Mergeability still belongs to
  each Dependabot PR's package checks, security scans, and review state.
- This does not edit Dependabot PR bodies to point at the old shared maintenance
  plan; author-aware workflow behavior is clearer and avoids metadata-only churn
  on bot PRs.
- This does not merge the Tailwind major PRs or any PR with red package checks.

## Deferred

- Re-run or refresh the queued Dependabot PRs after this workflow change lands.
- Real red dependency PRs stay separate: Tailwind 3 to 4 UI package PRs and any
  other package-check failures need their own compatibility slices or should
  remain blocked.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_audit_pr_body.py -q - passed: 13 tests.
- Command: python scripts/audit_pr_body.py --pr-author app/dependabot plans/PR-Dependabot-Body-Contract-Exemption.md
  - passed with `pr body audit: PASS (Dependabot PR body exempt)`.
- Command: python scripts/audit_pr_body.py plans/PR-Dependabot-Body-Contract-Exemption.md
  - failed as expected with normal AGENTS.md section 1b body-contract errors,
    proving a non-Dependabot invalid body remains invalid.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/atlas_dependabot_body_contract_pr_body.md - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pr_body_contract.yml` | 4 |
| `plans/PR-Dependabot-Body-Contract-Exemption.md` | 108 |
| `scripts/audit_pr_body.py` | 24 |
| `tests/test_audit_pr_body.py` | 61 |
| **Total** | **197** |
