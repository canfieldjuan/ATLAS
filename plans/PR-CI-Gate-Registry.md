# PR-CI-Gate-Registry

## Why this slice exists

Issue #2260 tracks the CI/CD enforcement arc after the audit found that Atlas'
mechanical checks are spread across AGENTS prose, local wrappers, CI workflows,
and branch-protection expectations. Phase 1 creates the canonical gate registry
so later enforcement work can consume one source of truth instead of duplicating
required/advisory gate lists.

Diff-budget override: this slice is over the 400 LOC target because the
registry, dependency-free parser, watcher integration, tests, and stale-doc
corrections must land atomically to avoid replacing one split source of truth
with another.

### Problem-derived contract

- Root cause: Required/advisory/local gate identity is hardcoded in scripts and
  restated in docs, so live branch-protection drift and stale AGENTS claims are
  easy to miss.
- Correct fix must touch/change: add a machine-readable gate registry, load
  default required branch-protection contexts from it, add tests for registry
  failure modes, and update docs that currently describe gate truth.
- Must not change: live GitHub branch-protection settings, runtime behavior of
  existing workflows, wrapper mutation behavior, or product/package test
  coverage.

## Scope (this PR)

Ownership lane: ci-cd-enforcement
Slice phase: workflow/process

1. Add a canonical repo-owned registry for current CI/CD gates and their
   enforcement class.
2. Make the required-status checker derive its default expected contexts from
   the registry.
3. Correct docs so humans and agents treat the registry/code as truth.
4. Add unit coverage for registry parsing and required-context derivation.

### Review Contract

- Acceptance criteria:
  - [ ] `scripts/check_required_status_checks.py` derives its default required
    context list from the gate registry, not an inline tuple.
  - [ ] Registry tests prove the current eight expected required contexts are
    selected and sorted by registry order.
  - [ ] Registry tests fail closed for malformed gate entries and prove
    advisory gates are not treated as branch-required.
  - [ ] AGENTS/docs no longer claim Intel UI test enrollment is manual-only;
    they identify the existing audit path.
  - [ ] Live branch-protection mutation is not part of this PR and remains
    documented as follow-up.
- Reachability proof: `python scripts/check_required_status_checks.py
  --payload-file <payload>` still validates a GitHub required-status payload
  against registry-derived defaults.
- Affected surfaces: branch-protection audit script, gate registry docs,
  CI/CD map docs, AGENTS bootstrap/procedure docs, security workflow tests.
- Risk areas: CI contract drift, stale docs, parser failure mode, accidental
  promotion of advisory gates.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/check_required_status_checks.py` default
  required-context resolution.
- Replaced-path behaviors: inline default tuple replaced by registry-derived
  required context list.
- Guard-relevant fields: registry gate `enforcement`, `context`, and
  `trusted_base` fields.
- Caller x input shape: CLI default invocation with GitHub required-status JSON
  payload.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - repo-local registry, no deployed config.
- Explicit value probe: tests cover explicit `--required` CLI overrides.
- Absent value probe: tests cover default registry loading when `--required` is
  omitted.
- Default-session/default-context probe: N/A - no session/config fallback change.
- Side-effect ordering: N/A - read-only audit script.

### Files touched

- `.github/workflows/branch_protection_required_checks.yml`
- `.github/workflows/unit_gate.yml`
- `AGENTS.md`
- `ci/gates.yml`
- `docs/OVERNIGHT_ARC_WORKFLOW.md`
- `docs/SECURITY_GUARDRAILS.md`
- `docs/SESSION_BOOTSTRAP.md`
- `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/ci_cd_runtime_duplication_audit.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-CI-Gate-Registry.md`
- `scripts/check_required_status_checks.py`
- `scripts/watch_owned_pr.sh`
- `tests/test_security_guardrails_workflow.py`
- `tests/test_watch_owned_pr.py`

## Mechanism

The registry lists current gates and classifies which are intended
branch-required. The required-status checker loads that file using a dependency-
free parser for the registry's constrained YAML shape and derives its default
expected checks from entries marked `branch_required`. Existing CLI overrides
still bypass the default list for targeted tests or future manual audits.

## Intentional

- Do not add a PyYAML dependency to the branch-protection audit workflow.
- Do not mutate live branch protection in this slice; this PR only makes drift
  auditable from repo state.
- Do not promote advisory gates in this slice.

## Deferred

- Phase 2/operator action: align live branch protection with registry-derived
  expectations or replace the broad required list with the future
  `atlas-merge-contract` meta-gate.
- Future slices: local-review attestation artifacts, wrapper ownership/draft
  hardening, reviewer thread disposition, Python coding standards ratchet.

Parked hardening: none.

## Verification

- Passed:
  - `/tmp/atlas-ci-gate-registry-venv/bin/python -m pytest tests/test_security_guardrails_workflow.py tests/test_watch_owned_pr.py -q`
  - `python3 -m py_compile scripts/check_required_status_checks.py`
  - `python3 - <<'PY' | python3 scripts/check_required_status_checks.py ...`
  - `git diff --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/branch_protection_required_checks.yml` | 1 |
| `.github/workflows/unit_gate.yml` | 10 |
| `AGENTS.md` | 10 |
| `ci/gates.yml` | 131 |
| `docs/OVERNIGHT_ARC_WORKFLOW.md` | 5 |
| `docs/SECURITY_GUARDRAILS.md` | 19 |
| `docs/SESSION_BOOTSTRAP.md` | 2 |
| `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` | 55 |
| `docs/ci_cd_autonomous_coding_map.md` | 14 |
| `docs/ci_cd_runtime_duplication_audit.md` | 10 |
| `docs/long_running_session_watcher_handoff.md` | 5 |
| `plans/PR-CI-Gate-Registry.md` | 161 |
| `scripts/check_required_status_checks.py` | 209 |
| `scripts/watch_owned_pr.sh` | 25 |
| `tests/test_security_guardrails_workflow.py` | 105 |
| `tests/test_watch_owned_pr.py` | 21 |
| **Total** | **783** |
