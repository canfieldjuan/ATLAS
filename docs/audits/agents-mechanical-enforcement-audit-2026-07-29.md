# AGENTS Mechanical Enforcement Audit - 2026-07-29

## Summary

Current `origin/main` is not a Claude two-reviewer gate. AGENTS.md defines one
builder plus the GitHub Codex connector, whose threads are enforced by
`live-reconciliation` (`AGENTS.md:3`, `AGENTS.md:7`, `AGENTS.md:8`). The live
branch-protection payload for `main` currently requires only these contexts:

- `live-reconciliation`
- `Gitleaks PR secret scan`
- `Gitleaks baseline growth guard`

Command evidence:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks
```

The local required-status checker confirms the mismatch between intended and
live protection:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks > /tmp/atlas-required-status-checks.json
python scripts/check_required_status_checks.py --payload-file /tmp/atlas-required-status-checks.json
# required status check audit: FAIL
# - diff-budget: missing required check
```

## Status Labels

- `ENFORCED_REQUIRED`: live branch protection requires the check.
- `ENFORCED_CI_NOT_REQUIRED`: a CI job exists, but live branch protection does
  not require it.
- `LOCAL_ONLY`: enforced by local wrappers or helpers before push.
- `ADVISORY_ONLY`: reports warnings or intentionally does not fail.
- `PROSE_ONLY`: policy exists in docs, but no enforcing artifact was found.
- `CONTRADICTED`: policy or checker expectation is stronger than live config.

## Enforcement Matrix

| Promise | Status | Evidence |
|---|---|---|
| Codex connector threads gate merge readiness. | `ENFORCED_REQUIRED` | AGENTS names Codex review threads as the reviewer gate (`AGENTS.md:7`, `AGENTS.md:8`). `ai_reconciliation_live.yml` defines required context `live-reconciliation` (`.github/workflows/ai_reconciliation_live.yml:42`) and runs `scripts/check_ai_reconciliation_live.py` (`.github/workflows/ai_reconciliation_live.yml:57`, `.github/workflows/ai_reconciliation_live.yml:61`). Live branch protection requires `live-reconciliation`. |
| `live-reconciliation` requires current-head Codex review attestation, except docs-only bypass. | `ENFORCED_REQUIRED` | The live checker fails missing current-head attestation (`scripts/check_ai_reconciliation_live.py:474`, `scripts/check_ai_reconciliation_live.py:484`) and allows docs-only PRs with no open scoped threads (`scripts/check_ai_reconciliation_live.py:469`, `scripts/check_ai_reconciliation_live.py:471`). Live branch protection requires `live-reconciliation`. |
| Gitleaks PR secret scan and baseline growth guard block merge. | `ENFORCED_REQUIRED` | `security_guardrails.yml` job name is `Gitleaks PR secret scan` (`.github/workflows/security_guardrails.yml:19`, `.github/workflows/security_guardrails.yml:20`). `gitleaks_baseline_growth_guard.yml` job name is `Gitleaks baseline growth guard` (`.github/workflows/gitleaks_baseline_growth_guard.yml:12`, `.github/workflows/gitleaks_baseline_growth_guard.yml:13`). Live branch protection requires both contexts. |
| Human non-Markdown PRs add exactly one plan; Markdown-only PRs may use `Docs-only: true`. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires this and says outcomes must surface locally and in CI (`AGENTS.md:96`, `AGENTS.md:101`). Local and CI plan admission run `audit_pr_plan_presence.py` (`scripts/pre_push_audit.sh:116`, `.github/workflows/plan_admission.yml:45`, `.github/workflows/plan_admission.yml:52`). Live branch protection does not require `Plan Admission`. |
| Full PR body shape or valid docs-only body is checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS defines full and docs-only body shapes (`AGENTS.md:191`, `AGENTS.md:201`, `AGENTS.md:204`). `audit_pr_body.py` validates the shape (`scripts/audit_pr_body.py:4`, `scripts/audit_pr_body.py:9`). `pr_body_contract.yml` runs it in CI (`.github/workflows/pr_body_contract.yml:43`, `.github/workflows/pr_body_contract.yml:49`). Live branch protection does not require `PR Body Contract`. |
| The local review bundle catches plan, body, drift, reconciliation, plan/code, and whitespace before PR review. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS describes the local bundle (`AGENTS.md:524`, `AGENTS.md:531`). `local_pr_review.sh` runs the PR body audit, pre-push wrapper, session drift, AI reconciliation, plan/code consistency, reviewer rules, and `git diff --check` (`scripts/local_pr_review.sh:141`, `scripts/local_pr_review.sh:158`, `scripts/local_pr_review.sh:173`, `scripts/local_pr_review.sh:193`, `scripts/local_pr_review.sh:214`, `scripts/local_pr_review.sh:220`, `scripts/local_pr_review.sh:231`). `pre_push_audit.yml` runs the bundle on PRs (`.github/workflows/pre_push_audit.yml:64`, `.github/workflows/pre_push_audit.yml:70`). Live branch protection does not require `Pre-push Audit`. |
| `diff-budget` should be a required status check. | `CONTRADICTED` | The workflow exists (`.github/workflows/diff_budget.yml:23`) and the required-status checker expects `diff-budget` (`scripts/check_required_status_checks.py:13`, `scripts/check_required_status_checks.py:15`). Live branch protection omits it, and the checker fails against the live payload. |
| Unit suite execution should happen on every PR. | `ENFORCED_CI_NOT_REQUIRED` | `unit_gate.yml` states the whole unit suite should run on every PR (`.github/workflows/unit_gate.yml:3`, `.github/workflows/unit_gate.yml:9`) but also states it is not yet required (`.github/workflows/unit_gate.yml:11`). Live branch protection does not require `Unit Gate`. |
| Session lane drift is checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires session ownership/lane checks before PR mutation (`AGENTS.md:425`, `AGENTS.md:454`). `new_pr_plan.sh` refuses a lane mismatch (`scripts/new_pr_plan.sh:126`) and `audit_pr_session_drift.py` fails ownership or current-body phase errors (`scripts/audit_pr_session_drift.py:112`, `scripts/audit_pr_session_drift.py:119`). `session_lane.yml` runs the drift audit in CI (`.github/workflows/session_lane.yml:60`, `.github/workflows/session_lane.yml:67`). Live branch protection does not require `Session Lane`. |
| PR ownership before mutation is guarded. | `LOCAL_ONLY` | AGENTS says run `check_session_pr_ownership.py` before PR mutation when metadata is known (`AGENTS.md:436`, `AGENTS.md:440`). The script fails when the PR is not owned or is listed must-not-touch (`scripts/check_session_pr_ownership.py:92`, `scripts/check_session_pr_ownership.py:95`). No PR workflow was found that can validate a local session-state file. |
| Push wrapper prevents skipped local review. | `LOCAL_ONLY` | AGENTS says `push_pr.sh` wires the body file and rejects `--no-verify` (`AGENTS.md:485`, `AGENTS.md:490`). The wrapper rejects `--no-verify` (`scripts/push_pr.sh:55`, `scripts/push_pr.sh:57`) and runs local review when the managed hook will not (`scripts/push_pr.sh:86`, `scripts/push_pr.sh:89`). GitHub cannot know if a user pushed another way. |
| Plans archive backlog nudges teardown cleanup. | `ADVISORY_ONLY` | AGENTS requires archiving merged plan docs during teardown (`AGENTS.md:649`). `local_pr_review.sh` marks plans archive backlog as advisory and non-blocking (`scripts/local_pr_review.sh:233`, `scripts/local_pr_review.sh:240`). This is intentionally not a red check. |
| Cross-layer caller hints must be read for logic/shared-function PRs. | `ADVISORY_ONLY` | AGENTS says the hints are advisory because outside references can be valid (`AGENTS.md:555`, `AGENTS.md:558`). `local_pr_review.sh` runs `audit_cross_layer_callers.py` as a check label (`scripts/local_pr_review.sh:185`, `scripts/local_pr_review.sh:186`), but the policy text makes the builder's response judgment-based. |
| Guard class closure, seam convergence, boundary enumeration, and deployed config probing are structural tripwires. | `ADVISORY_ONLY` | These workflows self-identify as advisory or never-blocking: guard class closure (`.github/workflows/guard_class_closure.yml:3`, `.github/workflows/guard_class_closure.yml:6`), seam convergence (`.github/workflows/seam_convergence.yml:3`, `.github/workflows/seam_convergence.yml:8`), boundary enumeration (`.github/workflows/boundary_change_enumeration.yml:3`, `.github/workflows/boundary_change_enumeration.yml:4`), and deployed config probing (`.github/workflows/deployed_config_probing.yml:3`, `.github/workflows/deployed_config_probing.yml:4`). Live branch protection requires none of them. |
| Auditor/checker changes ship fixture tests. | `PROSE_ONLY` | AGENTS requires fixture tests for new `scripts/audit_*.py` files and negative coverage for checkers (`AGENTS.md:897`, `AGENTS.md:914`, `AGENTS.md:932`). Existing local review runs many tooling tests from `pre_push_audit.yml` (`.github/workflows/pre_push_audit.yml:80`, `.github/workflows/pre_push_audit.yml:83`), but no generic changed-file gate was found that fails every new or changed checker lacking fixture coverage. |

## Follow-Up Slices

1. `PR-Require-Diff-Budget-Branch-Protection`: make live branch protection match
   `scripts/check_required_status_checks.py` by requiring `diff-budget`, or
   intentionally remove `diff-budget` from the required-context expectation.
2. `PR-Required-Workflow-Enrollment-Audit`: decide which existing CI-only
   process checks must become branch-protection required, starting with
   `Pre-push Audit`, `PR Body Contract`, `Plan Admission`, `Review Contract`,
   `Session Lane`, and `Unit Gate`.
3. `PR-Audit-Checker-Fixture-Enforcement`: add a changed-file gate for new or
   materially changed `scripts/audit_*.py`, `scripts/check_*.py`, and
   `scripts/evaluate_*.py` files so fixture coverage is mechanical instead of
   prose-only.
4. `PR-Teardown-Archive-Enforcement-Decision`: decide whether plan archive
   backlog should remain advisory or become a required cleanup gate with a
   concurrent-session-safe exception.

## Bottom Line

The current hard merge gate is much narrower than the local/CI review bundle:
branch protection blocks Codex thread reconciliation and Gitleaks only. The
AGENTS rules that keep PR shape sane mostly exist and run, but they are not live
branch-protection blockers today. The biggest concrete contradiction is
`diff-budget`: repository code expects it to be required, while live protection
does not require it.
