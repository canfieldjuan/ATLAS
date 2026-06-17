# Security Guardrails

Atlas has repo-wide CI guardrails for static security checks plus one
target-specific DAST workflow.

## What Runs Repo-Wide

`Security Guardrails` runs on pushes to `main`, manual dispatch, and a weekly
schedule. It intentionally does not run the full scanner suite on every pull
request because Atlas uses small high-frequency slices and the scanner set can
add several minutes of wall-clock time.

Most first-run scanners are advisory while the adoption backlog is burned down:
they upload SARIF or print findings without making `main` permanently red.
Full-history Gitleaks is the exception: it runs on every PR, push to `main`,
manual run, and scheduled run, and it fails on unbaselined leaks. On PRs after
the initial adoption merge, the scan reads
`docs/security/gitleaks-baseline.json` from the trusted base branch instead of
the checked-out PR head. A lightweight PR guard also rejects changes to that
baseline, so a future PR cannot add a secret and hide it by growing the
baseline.

Legitimate baseline rotations are allowed only through a narrow controlled
path: rotate or revoke the exposed provider credential first, add the
`security-rotation` PR label, and keep the diff limited to
`docs/security/gitleaks-baseline.json`, `docs/SECURITY_GUARDRAILS.md`,
`HARDENING.md`, and the slice plan under `plans/PR-*.md`. The label alone is
not enough; product-code or workflow changes in the same PR still fail the
baseline guard. The baseline guard runs from trusted base-branch workflow code
on `pull_request_target`, fetches the PR head only as git data, parses labels
from GitHub's JSON event payload, and rejects candidate baselines that drop
trusted-base fingerprints.

Current blocking posture: new unbaselined secrets block PRs; Semgrep, Trivy,
Checkov, pip-audit, and OSV are advisory/report-only until their adoption
backlogs are triaged and ratcheted.

- Full-history secret scan: Gitleaks checks out the complete branch history
  (`fetch-depth: 0`) so leaked keys in old commits are in scope. The workflow
  uses `docs/security/gitleaks-baseline.json` to suppress the known historical
  findings from the first trusted-main adoption scan while still failing on new
  leaks.
- Python SCA: pip-audit runs in advisory mode against deterministic tracked
  requirements files: `requirements.txt`, `atlas_edge/requirements.txt`, both
  `atlas_video-processing/**/requirements.txt` service files, and
  `graphiti-wrapper/requirements.txt`. `requirements.asr.txt` is parked until
  its floating `NVIDIA/NeMo@main` dependency is pinned.
- Ecosystem SCA: OSV Scanner runs recursively across the repository and reports
  dependency vulnerabilities to GitHub code scanning.
- SAST: Semgrep runs `p/default`, `p/owasp-top-ten`, and `p/python` across the
  repository and uploads SARIF in advisory mode.
- IaC/container config: Trivy config mode and Checkov scan Dockerfiles,
  Docker Compose, GitHub Actions, and Terraform if Terraform is added later.
  These are advisory until the initial HIGH/CRITICAL backlog is triaged.

These checks are repository-level guardrails. They are not tied to one Atlas
product unless the scanner finding points at product-specific files.

Findings from the nightly/main sweep should be triaged into an immediate fix PR
for exposed secrets or exploitable production risk, or into `HARDENING.md` for
non-blocking dependency/config/SAST debt.

The security workflows introduced here pin third-party GitHub Actions by commit
SHA and pin the Gitleaks container image by digest. The rest of the repository's
older workflow actions are intentionally left to a follow-up fleet-wide pinning
and OIDC review so this slice stays focused on the new guardrails.

Workflow supply-chain posture is checked by
`scripts/audit_workflow_security_posture.py`. It fails unapproved
`pull_request_target` jobs and unapproved `id-token: write` / `write-all`
usage, while reporting existing mutable action, reusable workflow, and
container/service image refs as warnings until the fleet-wide pin drain is
complete. Allowances are per expected job shape rather than whole-file
exceptions. `claude.yml` keeps `id-token: write` for the Claude Code action,
but the job is owner-gated, uses read-only GitHub token permissions, and pins
its third-party actions by commit SHA.

The adoption pass also removed current-tree secret-shaped fallback literals from
the GraphRAG Supabase API routes and the archived IndexNow script. Those paths
now require their environment variables instead of relying on hardcoded
placeholders.

## What Is Target-Specific

`Security DAST ZAP` runs OWASP ZAP baseline against a live URL. It is not a
whole-repo source scan; it tests the runtime surface exposed by one deployed
staging instance.

Configure repository variable `ATLAS_DAST_TARGET_URL` for scheduled and manual
scans.

If no target URL is configured, the workflow exits green with a message instead
of making every branch red before staging exists.

## Continuous Updates

Dependabot is enabled for:

- GitHub Actions.
- Active npm lockfile projects.
- Python requirements directories.
- Dockerfiles.
- Docker Compose files at the repo root and under `atlas_video-processing`.

The archived `_ARCHIVED_atlas-intel-next/package-lock.json` is intentionally
not enrolled for routine Dependabot churn. If that project is reactivated,
either move it out of `_ARCHIVED_` or add a separate archived-dependency policy.

## Initial Secret Scan Result

The initial trusted-main Gitleaks baseline contains 22 redacted findings across
3,862 commits. The highest-risk group is a committed `.env` in commit
`d63a9b77b9727766e14e523626c22dd6c1c80da8` with provider credentials,
including Stripe, Anthropic, OpenRouter, Reddit, Firecrawl, Stack Overflow,
Product Hunt, CAPTCHA, Apollo, SignalWire, Google Calendar, Resend, and Google
API-style keys. No raw secret values are stored in this document or in the
baseline file.

Treat those credentials as exposed. Rotate or revoke them at the provider, then
decide whether to rewrite history or keep the redacted baseline as the permanent
"known old leaks" boundary.

## Current Deferred Hardening

- Rotate/revoke the provider credentials exposed in historical commit
  `d63a9b77b9727766e14e523626c22dd6c1c80da8`.
- Rotate the archived IndexNow key that was removed from the branch tip but
  remains in git history.
- Pin remaining mutable action, reusable workflow, and container/service image
  refs across non-security product/check workflows.
- Burn down advisory Semgrep, Trivy, Checkov, pip-audit, and OSV findings, then
  ratchet the relevant scans from advisory to blocking.
- Pin or retire the floating `NVIDIA/NeMo@main` requirement in
  `requirements.asr.txt` before adding that file back to pip-audit.
- Add per-image Trivy image scans to image publish workflows when those
  production image build/push paths are named.
- Tune ZAP baseline rules after the first configured staging run produces a
  real report.
- Decide whether archived dependency manifests should be deleted, reactivated,
  or governed by a separate policy.
