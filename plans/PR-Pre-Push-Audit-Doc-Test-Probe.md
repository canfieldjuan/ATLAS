# PR-Pre-Push-Audit-Doc-Test-Probe

## Why this slice exists

PR #2294 promoted `unit-gate` but left `pre-push-audit` visible-only because
`docs/audits/required-workflow-enrollment-audit-2026-08-04.md` still named a
trusted-base PR-side docs/test consistency blocker. The operator asked to keep
moving through the CI lane and to fix mechanical checks upstream instead of
patching review symptoms.

This PR is over the 400 LOC target because it adds a new trusted-base safety
checker and the negative fixture matrix that proves the checker catches the
stale docs/test classes before merge. The implementation code is narrow; most
of the overage is test proof and plan/docs context.

### Problem-derived contract

- Root cause: `pre-push-audit` runs trusted base code on PR events and
  materializes the PR head as data, which is the right security model, but its
  trusted-base unit-test step cannot execute PR-authored tests. A PR can change
  `ci/gates.yml`, `docs/SECURITY_GUARDRAILS.md`,
  `.github/workflows/branch_protection_required_checks.yml`, or
  `tests/test_security_guardrails_workflow.py` in a way that only fails after
  merge when the push-to-main variant runs the new tests.
- Correct fix must touch/change: add a base-owned data-only auditor that reads
  the PR tree and checks the required-status registry, security docs,
  branch-protection workflow trigger paths, and security-guardrails test
  constants agree; run that auditor from `scripts/pre_push_audit.sh`; enroll
  its tests in `.github/workflows/pre_push_audit.yml`; document that the
  blocker now has a probe.
- Must not change: do not execute PR-authored code/tests in
  `pull_request_target`; do not promote `pre-push-audit` to `branch_required`;
  do not change live branch protection; do not change product/EOM/content lanes.

## Scope (this PR)

Ownership lane: dev-workflow/pre-push-audit-doc-test-probe
Slice phase: Workflow/process

1. Add `scripts/audit_pr_side_docs_test_consistency.py`, a data-only checker
   for the required-status docs/test contract.
2. Run the checker from `scripts/pre_push_audit.sh` and add trusted-base unit
   test enrollment in `.github/workflows/pre_push_audit.yml`.
3. Update the CI/security docs and audit notes to say the blocker has a probe
   but `pre-push-audit` is still not branch-required.

### Review Contract

- Acceptance criteria:
  - `scripts/audit_pr_side_docs_test_consistency.py` derives branch-required
    contexts/workflows from PR-side `ci/gates.yml` using the trusted
    `check_required_status_checks.py` parser.
  - The checker fails when `docs/SECURITY_GUARDRAILS.md` omits a
    branch-required context.
  - The checker fails when `.github/workflows/branch_protection_required_checks.yml`
    omits a required workflow/doc/script/test trigger path.
  - The checker rejects `paths:` lists that are nested below another
    `on.push` child instead of being the direct `on.push.paths` node.
  - The checker fails when a required workflow path is later excluded by a
    negative `on.push.paths` trigger.
  - The checker fails when `ci/gates.yml` points any registry gate at a
    workflow path that is missing from the PR tree.
  - The checker rejects absolute, traversal, or otherwise repo-escaping
    registry workflow paths.
  - The checker fails when `tests/test_security_guardrails_workflow.py`
    contains stale `REQUIRED_STATUS_CONTEXTS` or `REQUIRED_STATUS_WORKFLOW_PATHS`
    literals.
  - The checker rejects duplicate or mutating assignments to audited test
    constants instead of accepting an earlier matching value.
  - The checker rejects nested bindings and mutations of audited test constants
    so branch-time runtime effects cannot bypass the PR-side data probe.
  - The checker rejects runtime binding forms for audited test constants,
    including loop/import/function/class/delete/walrus/exception/with targets.
  - The checker requires audited test constants to stay literal tuples so the
    PR-side audit matches the push-to-main runtime equality check.
  - The checker rejects `push` mappings that are nested under another
    `on` child instead of being the direct `on.push` trigger.
  - The checker's CLI entrypoint is covered for synchronized OK output and a
    representative failure output/exit-code path.
  - `tests/test_open_pr_wrapper.py` real-local-review fixtures include the
    audited CI registry and security-guardrails test surfaces so wrapper tests
    exercise the intended plan-shape failure, not missing fixture files.
  - `scripts/pre_push_audit.sh` invokes the checker with `--repo-root
    "$repo_root"` so trusted-base CI reads PR files as data.
  - `.github/workflows/pre_push_audit.yml` runs
    `tests/test_audit_pr_side_docs_test_consistency.py` in both PR and
    push-to-main tooling-test steps.
- Reachability proof: `bash scripts/pre_push_audit.sh` reaches the new
  "PR-side docs/test consistency" check and reports pass/fail without executing
  PR-authored tests.
- Affected surfaces: pre-push local review bundle, trusted-base
  `.github/workflows/pre_push_audit.yml`, required-status docs/tests/audit docs.
- Risk areas: trusted-base execution safety, stale required-status docs/test
  mappings, false positives from exact-set comparisons, over-broad workflow
  promotion.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/pre_push_audit.sh` gains a new blocking
  pre-push audit check; `scripts/audit_pr_side_docs_test_consistency.py` is a
  new admission checker for CI/security docs/test drift.
- Replaced-path behaviors: none; existing checks still run in the same order
  except the new checker runs before PR watcher safety.
- Guard-relevant fields: `ci/gates.yml` branch-required `context` and
  `workflow`; backticked contexts in `docs/SECURITY_GUARDRAILS.md`; quoted
  push path triggers in `.github/workflows/branch_protection_required_checks.yml`;
  literal `REQUIRED_STATUS_CONTEXTS` and `REQUIRED_STATUS_WORKFLOW_PATHS`.
- Caller x input shape: local and trusted-base CI call the checker with a repo
  root; the checker reads text/AST literals only and never imports PR tests.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no environment or deployed config
  fallback changes.
- Explicit value probe: `python scripts/audit_pr_side_docs_test_consistency.py`
  passes on the current repo.
- Absent value probe: fixture tests remove docs/workflow/test entries and assert
  failures.
- Default-session/default-context probe: `bash scripts/pre_push_audit.sh`
  reaches the checker through the default local-review bundle.
- Side-effect ordering: the checker runs after existing docs/inventory checks
  and before PR watcher safety; it does not write files or mutate GitHub state.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md`
- `docs/audits/required-workflow-enrollment-audit-2026-08-04.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `plans/PR-Pre-Push-Audit-Doc-Test-Probe.md`
- `scripts/audit_pr_side_docs_test_consistency.py`
- `scripts/pre_push_audit.sh`
- `tests/test_audit_pr_side_docs_test_consistency.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_pre_push_audit.py`
- `tests/test_pre_push_audit_workflow.py`

## Mechanism

`scripts/audit_pr_side_docs_test_consistency.py` loads the trusted
`scripts/check_required_status_checks.py` parser from `ATLAS_AUDIT_SCRIPT_ROOT`
and parses PR-side `ci/gates.yml` from `ATLAS_AUDIT_REPO_ROOT`. It derives the
branch-required context set and workflow set, then reads three PR-side files as
data:

- `docs/SECURITY_GUARDRAILS.md` must mention every branch-required context as a
  backticked context.
- `.github/workflows/branch_protection_required_checks.yml` must trigger on
  every branch-required workflow plus the registry, docs, checker script,
  branch-protection workflow, and security-guardrails test file through the
  direct `on.push.paths` node, and must not contain negative path triggers that
  effectively remove audited paths.
- `tests/test_security_guardrails_workflow.py` must carry literal
  `REQUIRED_STATUS_CONTEXTS` and `REQUIRED_STATUS_WORKFLOW_PATHS` tuples as
  exactly one unconditional module-level assignment each, matching the derived
  registry tuple without any later or nested assignment, mutation, or runtime
  rebinding.
- Every workflow path named by `ci/gates.yml`, required or advisory, must stay
  inside the PR tree and exist as a regular file.

`scripts/pre_push_audit.sh` calls the checker with `--repo-root "$repo_root"`,
so `pull_request_target` CI executes trusted code while inspecting the
materialized PR worktree as data.

## Intentional

- No `pre-push-audit` branch-required promotion in this PR; the slice only
  closes the data-only consistency blocker.
- No PR-authored test execution in `pull_request_target`; AST parsing and text
  checks are the safety boundary.
- The checker enforces known required-status docs/test surfaces, not a generic
  natural-language docs consistency engine.

## Deferred

- Re-run the required-workflow enrollment decision for `pre-push-audit` after
  this probe burns in.

Parked hardening: none.

## Verification

- `python scripts/audit_pr_side_docs_test_consistency.py` - passed.
- `python -m pytest tests/test_open_pr_wrapper.py tests/test_audit_pr_side_docs_test_consistency.py tests/test_pre_push_audit_workflow.py tests/test_pre_push_audit.py tests/test_security_guardrails_workflow.py -q` - 135 passed.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `docs/SECURITY_GUARDRAILS.md` | 8 |
| `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` | 9 |
| `docs/audits/required-workflow-enrollment-audit-2026-08-04.md` | 13 |
| `docs/ci_cd_autonomous_coding_map.md` | 6 |
| `plans/PR-Pre-Push-Audit-Doc-Test-Probe.md` | 212 |
| `scripts/audit_pr_side_docs_test_consistency.py` | 433 |
| `scripts/pre_push_audit.sh` | 1 |
| `tests/test_audit_pr_side_docs_test_consistency.py` | 506 |
| `tests/test_open_pr_wrapper.py` | 6 |
| `tests/test_pre_push_audit.py` | 57 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| **Total** | **1261** |
