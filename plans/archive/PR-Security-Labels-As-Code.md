# PR-Security-Labels-As-Code

## Why this slice exists

Issue #1656 now has a CVE remediation SLA path, but the labels that make the
path observable are still mutable GitHub metadata. PR #1814 created
`cve-remediation-sla` manually, while `.github/dependabot.yml`, `SECURITY.md`,
and the Gitleaks baseline rotation guard also rely on `dependencies`,
`security`, and `security-rotation`. Live repo state proves the drift: a new
Dependabot PR only received `cve-remediation-sla` because the other configured
labels do not exist.

Root cause: security-critical label names were referenced independently from
docs, Dependabot config, and guard scripts without a committed repo-label
contract or any automation that restores those labels when GitHub metadata
drifts.

This change fixes the root for the security label set by making those labels
code-owned in a manifest, adding a main/manual sync workflow, and testing that
every security policy label reference is covered by the manifest.

Diff budget note: this is over the 400 LOC soft cap because the root fix is
indivisible across manifest, trusted sync workflow, stdlib sync helper, and
negative tests. Splitting those pieces would recreate the original failure
mode: labels could be documented without being reconciled, or reconciled
without a tested contract that catches drift. The post-review Codex P2 fixes
kept the same PR over the soft cap because they close the same label-as-code
failure class: source-reference drift, live-metadata drift, case-only drift,
and invalid label metadata reaching the trusted sync job.

## Scope (this PR)

Ownership lane: security/labels-as-code
Slice phase: Workflow/process
Max files: 7

Post-review fix budget: the local fix-mode baton limits the
workflow-dispatch guard change to the label workflow, the security policy docs
test, and this plan. The plan-level budget covers the whole PR diff.

1. Add a committed repo-label manifest for the security/Dependabot labels
   used by the current #1656 security guardrails.
2. Add a stdlib label-sync helper and a main/manual GitHub Actions workflow
   that reconciles the manifest into repository labels and repairs live drift
   on a weekly schedule.
3. Extend the security policy docs contract tests so Dependabot labels,
   labels documented in `SECURITY.md`, the Gitleaks rotation label, the
   manifest, and the label-sync workflow stay aligned.

### Review Contract

Acceptance criteria:
- `.github/labels.json` defines every label referenced by
  `.github/dependabot.yml`, `SECURITY.md`, and the Gitleaks rotation guard.
- The sync helper plans create/update operations for missing or stale live
  labels, handles case-only live label drift as an update/rename, and refuses
  malformed manifests.
- The label workflow only syncs from trusted `main`/manual execution, not from
  untrusted PR code.
- The security policy docs workflow runs when the manifest, sync workflow, or
  sync helper changes.

Affected surfaces:
- Repository label metadata contract for Dependabot/CVE SLA and controlled
  Gitleaks baseline rotations.
- Security policy docs checks.

Risk areas:
- Over-broad workflow write permissions.
- Manifest drift from the actual label names consumed by Dependabot or guard
  scripts.
- A sync script that silently accepts duplicate or malformed labels.

Triggered reviewer rules:
- R1 Requirements match.
- R2 Test evidence.
- R3 Security/auth.
- R8 CI/workflow enrollment.
- R13 Fix class, not example.
- R14 Codebase verification.

### Files touched

- `.github/labels.json`
- `.github/workflows/atlas_security_policy_docs_checks.yml`
- `.github/workflows/repo_labels.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/PR-Security-Labels-As-Code.md`
- `scripts/sync_github_labels.py`
- `tests/test_security_policy_docs.py`

## Mechanism

`.github/labels.json` is the code-owned label manifest. The sync helper reads
that manifest, reads live labels from `gh label list --json
name,description,color`, computes missing/stale labels, and either reports
drift (`--check`) or applies it (`--apply`) with `gh label create/edit`.

`.github/workflows/repo_labels.yml` runs the helper on pushes to `main` that
change the manifest/helper/workflow, on manual dispatch, and on a weekly
schedule. The workflow grants `issues: write` only to the sync job because
GitHub labels are issue/PR metadata; it does not run on `pull_request`. Manual
dispatches are guarded to the `main` ref and the checkout pins to the
repository default branch before running the token-backed repo script.

`tests/test_security_policy_docs.py` stays the CI-facing contract for the
security policy documents. It now also imports the sync helper directly and
proves malformed manifests, missing referenced labels, missing live labels, and
stale live labels fail the relevant checks. It derives the documented
Dependabot labels from `SECURITY.md`, imports the Gitleaks rotation label from
the guard script, and ensures the security docs workflow runs when those source
files change.

The sync helper keys live labels case-insensitively and emits `gh label edit
<current-name> --name <manifest-name>` for case-only drift. Manifest loading
also enforces GitHub's 100-character label description limit so invalid label
metadata fails before merge instead of in the trusted main sync job.

## Intentional

- The manifest is scoped to security-critical labels, not every default repo
  label. This slice closes the #1656 label drift; a full repo taxonomy cleanup
  can happen separately if needed.
- The sync workflow runs only from trusted `main` or manual dispatch. PRs prove
  the contract through tests instead of letting untrusted branch code mutate
  repository metadata; the manual path is branch-guarded and checks out the
  default branch before running the sync script.
- The helper shells out to `gh` rather than adding a GitHub REST dependency.
  The workflow image already includes `gh`, and the helper remains stdlib-only
  for local and CI use.

## Deferred

- Full repository label taxonomy cleanup is deferred; this PR only owns labels
  that current security policy and guardrail code consume.

Parked hardening: none.

## Verification

- `python -m unittest tests.test_security_policy_docs` -- 18 tests passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` --
  passed; existing mutable-action warnings are unrelated parked debt.
- `python -m py_compile` against `scripts/sync_github_labels.py` -- passed.
- `scripts/check_ascii_python.sh` through `bash` -- passed.
- `python scripts/sync_github_labels.py --manifest .github/labels.json --apply`
  -- created `dependencies`, `security`, and `security-rotation` live labels
  from the manifest.
- `python scripts/sync_github_labels.py --manifest .github/labels.json --check`
  -- passed after sync.
- `scripts/push_pr.sh` with the prepared PR body and `-u origin HEAD` --
  local review passed and branch pushed.
- Review-fix verification: `python -m unittest tests.test_security_policy_docs`
  -- 18 tests passed with the workflow-dispatch guard assertions.
- Review-fix verification:
  `python scripts/audit_workflow_security_posture.py .github/workflows` --
  passed; existing mutable-action warnings are unrelated parked debt.
- Review-fix verification:
  `python scripts/sync_github_labels.py --manifest .github/labels.json --check`
  -- passed.
- Codex P2 review-fix verification:
  `python -m unittest tests.test_security_policy_docs` -- 20 tests passed with
  path-filter, `SECURITY.md` label extraction, schedule, case-only drift, and
  overlong-description coverage.
- Codex P2 review-fix verification:
  `python scripts/audit_workflow_security_posture.py .github/workflows` --
  passed; existing mutable-action warnings are unrelated parked debt.
- Codex P2 review-fix verification:
  `python -m py_compile` against `scripts/sync_github_labels.py` -- passed.
- Codex P2 review-fix verification:
  `python scripts/sync_github_labels.py --manifest .github/labels.json --check`
  -- passed.
- Pending review-fix push: `scripts/push_pr.sh` with the updated PR body and
  `--force-with-lease origin HEAD`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/labels.json` | 22 |
| `.github/workflows/atlas_security_policy_docs_checks.yml` | 10 |
| `.github/workflows/repo_labels.yml` | 40 |
| `docs/SECURITY_GUARDRAILS.md` | 7 |
| `plans/PR-Security-Labels-As-Code.md` | 187 |
| `scripts/sync_github_labels.py` | 234 |
| `tests/test_security_policy_docs.py` | 206 |
| **Total** | **706** |
