# PR-Security-Full-Sweep

## Why this slice exists

The repo's current security automation is all baseline/diff-oriented or
dependency-bump-oriented, so it never re-examines the base:

- `dependabot.yml` opens PRs for new advisories on declared dependencies
  (SCA-lite). It does not scan the tree, history, or secrets.
- `security_guardrails.yml` runs a gitleaks baseline-growth guard -- by design
  it ignores anything already in the baseline, so secrets already committed are
  not resurfaced.
- `security_dast_zap.yml` is dynamic (running-app) scanning, not code/secrets/
  dependency analysis.
- There is no semgrep, no CodeQL, no full-history gitleaks, and no full SCA
  pass.

So buried/pre-existing issues -- e.g. a secret committed long ago (a committed
`.env` at `d63a9b77` was flagged this session) -- are never surfaced by CI,
because every existing scan deliberately skips the base. This slice adds the
missing no-baseline, whole-tree, full-history sweep that flushes buried
problems out, on a schedule, separate from the PR-time baseline gates.

## Scope (this PR)

Ownership lane: ci/security
Slice phase: Production hardening

Add `.github/workflows/security_full_sweep.yml`:

- Triggers: `schedule` (single, clearly-commented cron line, default
  `0 22 * * *` = 5:00pm CDT nightly) + `workflow_dispatch` for on-demand runs.
  The cron line carries a UTC<->Central conversion block so retiming is a
  one-line edit (GitHub cannot read a variable in `schedule`).
- `permissions: contents: read`; a `concurrency` group; per-job
  `timeout-minutes: 20`.
- Three independent jobs:
  1. semgrep -- whole tree, registry rules, no `--baseline-commit`, `--error`
     so findings exit non-zero.
  2. gitleaks -- full history (`fetch-depth: 0`, `--log-opts=--all`),
     `--redact`, no baseline file.
  3. SCA -- pip-audit over `requirements.txt` and `requirements.asr.txt`.
- Actions SHA-pinned to the repo convention (`actions/checkout`,
  `actions/setup-python`); scanners run via CLI to avoid additional unpinned
  `uses` entries.

### Files touched

- `.github/workflows/security_full_sweep.yml`
- `plans/PR-Security-Full-Sweep.md`

## Mechanism

Each scan step runs under `set -euo pipefail` and lets the scanner's non-zero
exit (semgrep `--error`, gitleaks on leaks, pip-audit on vulns) fail the job.
A separate summary step gated `if: always()` appends counts to
`$GITHUB_STEP_SUMMARY` from the report file -- it reports regardless of scan
outcome but does not mask the scan step's failure (the job is already red). It
runs under `set -u` only, so a reporting hiccup cannot abort. gitleaks runs
with `--redact` so secret values never reach logs.

## Intentional

- Separate from the PR-time baseline gates. Baseline scans gate new issues on a
  diff; this sweep finds buried/existing ones. Both are wanted.
- Fails red by design. This is a scheduled tripwire, expected to be red until
  buried debt is triaged; that is the opposite of the advisory
  `continue-on-error` pattern (which can never fail and so hides findings).
- Not brittle. No `continue-on-error`, no swallowed exit codes, no error
  bypass. Scanner findings and scanner-tool errors both turn the job red. The
  `if: always()` guard appears only on summary steps, never on a scan step.
- Configurable cadence. One commented cron line + `workflow_dispatch`.
- gitleaks `--redact`: secret locations are already public in history, so the
  redacted report is a tripwire, not a new disclosure. The real fix for a hit
  is key ROTATION + history scrub, noted in the workflow and summary.

## Deferred

- `osv-scanner` (broader lockfile SCA) and CodeQL as additional jobs -- the
  repo already references a pinned osv-scanner reusable workflow that can be
  wired in a follow-up.
- SARIF upload to the GitHub Security tab (needs `security-events: write`);
  kept out of v1 to keep permissions minimal.
- Auto-triage / issue-filing of findings.

Parked hardening: none.

## Verification

- `.github/workflows/security_full_sweep.yml` parses as valid YAML (python yaml
  safe-load, no error).
- `scripts/audit_workflow_security_posture.py` passes -- "workflow security
  posture audit passed" (no ERROR; pinned actions, `contents: read`, no
  `pull_request_target`, no OIDC).
- `tests/test_audit_workflow_security_posture.py` and
  `tests/test_claude_workflow_security.py` pass (15 passed).
- Self-audit: grepping the new workflow for `continue-on-error`, the OR-true
  bypass, and `set +e` returns nothing; scan steps propagate scanner exit
  codes.
- ASCII-only check on the new files is clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/security_full_sweep.yml` | ~150 |
| `plans/PR-Security-Full-Sweep.md` | ~95 |
| **Total** | **~245** |
