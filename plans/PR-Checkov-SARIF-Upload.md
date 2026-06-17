# PR-Checkov-SARIF-Upload

## Why this slice exists

The security guardrail backlog includes advisory Checkov IaC findings, but the
current Checkov job only leaves results in the Actions log. That makes backlog
triage harder than the other scanners: Gitleaks, Semgrep, and Trivy all upload
SARIF into GitHub code scanning, while Checkov does not.

Root cause: the Checkov job is advisory but has no SARIF output/upload path.
This fixes that root by configuring Checkov to emit a SARIF file and uploading
that file to GitHub code scanning with the already pinned CodeQL SARIF upload
action pattern used elsewhere in the same workflow.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. TODO: Name the narrow behavior this PR changes.
1. Configure the Checkov IaC scan to produce both CLI output and
   `checkov.sarif`.
2. Upload `checkov.sarif` to GitHub code scanning in advisory mode.
3. Update security guardrail docs so Checkov is described consistently with
   the other SARIF-producing scanners.

### Review Contract

Acceptance criteria:

- The Checkov job grants only the additional permissions needed for SARIF
  upload: `security-events: write` plus `actions: read`.
- The Checkov action still scans the same directory/frameworks and keeps
  `soft_fail: true`.
- Checkov writes `output_format: cli,sarif` and
  `output_file_path: console,checkov.sarif`.
- The workflow uploads `checkov.sarif` with
  `github/codeql-action/upload-sarif` pinned to a commit SHA.
- Docs say Checkov now uploads SARIF in advisory mode.

Affected surfaces:

- Scheduled/main security guardrail Checkov reporting.
- GitHub code scanning visibility for IaC findings.

Risk areas:

- Checkov must stay advisory; adding SARIF upload must not make existing
  backlog findings block main.
- The new upload step needs scoped permissions at the job level because job
  permissions override workflow-level defaults.

Triggered reviewer rules:

- R1 Requirements match
- R2 Test evidence
- R3 Security/auth
- R8 CI/workflow safety
- R14 Codebase verification

### Files touched

- `.github/workflows/security_guardrails.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/INDEX.md`
- `plans/PR-Checkov-SARIF-Upload.md`
- `plans/archive/PR-ASR-Requirements-Audit-Pin.md`

## Mechanism

Checkov's GitHub Action supports dual output by setting:

```yaml
output_format: cli,sarif
output_file_path: console,checkov.sarif
```

The first output preserves readable job logs; the second writes a SARIF file.
The existing CodeQL upload action pattern then publishes that SARIF file with
`if: always()` so findings still surface even when Checkov reports policy
violations. `soft_fail: true` remains in place, so the job is still advisory.

## Intentional

- This does not change Checkov policy scope, frameworks, skipped paths, or
  blocking behavior. It only changes result visibility.
- This adds one more pinned `github/codeql-action/upload-sarif` use with the
  same SHA currently used by the workflow. Dependabot #1630 is already open to
  update that action family; this slice stays on the repo's current pin pattern.
- This does not attempt to triage or fix the Checkov findings; it makes the
  backlog easier to inspect in code scanning.

## Deferred

- Burn down the actual advisory Checkov findings and ratchet the relevant
  scanner once the known backlog is fixed or explicitly waived.

Parked hardening: none.

## Verification

- YAML parse smoke for `.github/workflows/security_guardrails.yml` -- passed.
- `rg -n "checkov\.sarif|output_format|output_file_path|security-events|actions: read|soft_fail|Checkov.*SARIF|Trivy and Checkov" .github/workflows/security_guardrails.yml docs/SECURITY_GUARDRAILS.md`
  -- passed; Checkov SARIF output, upload permissions, advisory soft-fail, and
  docs language are present.
- `python scripts/sync_pr_plan.py --check plans/PR-Checkov-SARIF-Upload.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/security_guardrails.yml` | 10 |
| `docs/SECURITY_GUARDRAILS.md` | 3 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Checkov-SARIF-Upload.md` | 118 |
| `plans/archive/PR-ASR-Requirements-Audit-Pin.md` | 0 |
| **Total** | **134** |
