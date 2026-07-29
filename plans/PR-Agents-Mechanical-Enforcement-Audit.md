# PR-Agents-Mechanical-Enforcement-Audit

## Why this slice exists

The operator asked to find the mechanical AGENTS.md checks that are not being
respected after repeated PRs reached GitHub with immediate red checks and review
loops. The repository has many local and CI workflow guards, but it is not
obvious which are branch-protection blockers, which are ordinary red checks, and
which are advisory warnings. This slice creates the code-grounded enforcement
map before changing any gate.

### Problem-derived contract

- Root cause: AGENTS.md mixes merge-blocking duties, CI jobs, local-only wrapper
  duties, and advisory review aids in one workflow contract, so a builder can
  see "mechanical" and assume "merge-blocking" even when the actual enforcement
  is softer.
- Correct fix must touch/change: add a documentation audit that maps AGENTS.md
  policy anchors to the scripts, workflows, and live branch-protection evidence
  that enforce or fail to enforce them.
- Must not change: no workflow behavior, no required checks, no scripts, no
  product code, no customer-visible shape, and no unrelated open PR lanes.

## Scope (this PR)

Ownership lane: workflow/agents-enforcement-audit
Slice phase: Workflow/process

1. Add a single audit report classifying AGENTS mechanical promises as required,
   CI-only, local-only, advisory, prose-only, contradicted, or unknown live
   config.
2. Keep this slice audit-only and name follow-up enforcement slices without
   implementing them.

### Review Contract

- Acceptance criteria:
  1. `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` records
     the live required-status payload command and names the current required
     contexts from that payload.
  2. The audit report cites current code or workflow file lines for each
     enforcement classification it assigns.
  3. The audit report explicitly separates branch-protection blockers from
     CI-only, local-only, and advisory checks.
  4. The audit report names at least one follow-up slice for every contradicted
     or prose-only enforcement gap it finds.
- Reachability proof: N/A - documentation/audit-only slice with no runtime
  surface.
- Affected surfaces: `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md`
  and this plan.
- Risk areas: stale live GitHub state, over-claiming required enforcement, and
  accidentally changing workflow behavior in an audit slice.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: N/A - no boundary change.
- Replaced-path behaviors: N/A.
- Guard-relevant fields: N/A.
- Caller x input shape: N/A.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no guard/config boundary change.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: N/A.
- Side-effect ordering: N/A.

### Files touched

- `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md`
- `plans/PR-Agents-Mechanical-Enforcement-Audit.md`

## Mechanism

The audit report reconstructs AGENTS enforcement from the code outward:
AGENTS.md policy lines, local wrappers, CI workflows, enforcement scripts, and
the live `main` branch-protection required-status payload. It assigns one status
per promise and records follow-up slices for any gap that is not enforced today.

## Intentional

- No checker or workflow changes in this slice. Changing gates before the map is
  written would mix diagnosis and remediation.
- Use the full plan/body contract even though the diff is Markdown-only because
  this is non-trivial workflow governance.

## Deferred

- Promote or repair specific mechanical gates in follow-up slices named by the
  audit report.
- Synthetic fixture/adapters test suite for AGENTS enforcement behavior remains
  deferred until the current enforcement map identifies which contracts need
  executable fixtures first.

Parked hardening: none.

## Verification

- `gh pr list --state open --json number,title,headRefName,headRefOid,isDraft --limit 60`
- `git log --oneline -15 origin/main`
- `gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks > /tmp/atlas-required-status-checks.json && python scripts/check_required_status_checks.py --payload-file /tmp/atlas-required-status-checks.json` - expected failure proving `diff-budget` is missing from live branch protection.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-body-agents-mechanical-enforcement-audit.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` | 84 |
| `plans/PR-Agents-Mechanical-Enforcement-Audit.md` | 119 |
| **Total** | **203** |
