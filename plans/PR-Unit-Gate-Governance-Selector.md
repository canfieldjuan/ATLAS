# PR-Unit-Gate-Governance-Selector

## Why this slice exists

Issue #2260 and the #2283 required-workflow enrollment audit found that
`unit-gate` is not ready for branch protection because governance docs can fall
through the selector as test-free Markdown. The concrete failure mode was a
required-status/security doc change that selected no reachable tests, passed
growth-only, and then failed `tests/test_security_guardrails_workflow.py` in a
later broader path.

### Problem-derived contract

- Root cause: `scripts/select_impacted_tests.py` only treats known code/workflow
  CI surfaces as explicit owners. Markdown docs default to test-free selection,
  so CI/security governance docs whose claims are validated by
  `tests/test_security_guardrails_workflow.py` can skip that owning suite.
- Correct fix must touch/change: add explicit selector ownership for the
  current required-status/security governance docs, prove each selects
  `tests/test_security_guardrails_workflow.py`, and keep the missing-owner
  fallback fail-closed.
- Must not change: no `ci/gates.yml` enforcement promotion, no live branch
  protection mutation, no `unit_gate.yml` runtime change, no product behavior,
  and no broader full-suite policy change.

## Scope (this PR)

Ownership lane: dev-workflow/unit-gate-selector
Slice phase: Workflow/process

1. Map CI/security governance docs that feed required-status/security claims to
   their owning selector tests instead of Markdown growth-only.
2. Add regression coverage proving the mapped docs select
   `tests/test_security_guardrails_workflow.py`.
3. Remove duplicate explicit-owner membership while editing the same decision
   set.

### Review Contract

- Acceptance criteria:
  - `scripts/select_impacted_tests.py` maps
    `docs/SECURITY_GUARDRAILS.md` to
    `tests/test_security_guardrails_workflow.py`.
  - `scripts/select_impacted_tests.py` maps the current CI/CD governance audit
    docs in scope to `tests/test_security_guardrails_workflow.py`.
  - `tests/test_select_impacted_tests.py::test_explicit_ci_surface_owners_are_selected`
    proves those mappings select the owning suite.
  - `scripts/select_impacted_tests.py` has only one explicit-owner entry for
    `scripts/check_ai_reconciliation_live.py`.
- Reachability proof: `python -m pytest tests/test_select_impacted_tests.py -q`
  exercises the real selector entrypoint and observes selected test paths.
- Affected surfaces: `scripts/select_impacted_tests.py`,
  `tests/test_select_impacted_tests.py`, and this plan.
- Risk areas: under-selection of CI governance tests, stale owner paths,
  over-broad selector escalation, and duplicate owner-map membership.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

Closure declaration for explicit governance-doc owners:

- Set status: CLOSED for this slice. The set is the current docs whose
  required-status/security governance claims are directly coupled to
  `tests/test_security_guardrails_workflow.py`: `docs/SECURITY_GUARDRAILS.md`,
  `docs/ci_cd_autonomous_coding_map.md`,
  `docs/ci_cd_runtime_duplication_audit.md`,
  `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md`, and
  `docs/audits/required-workflow-enrollment-audit-2026-08-04.md`.
- Membership source: ENUMERATED from issue #2260, the #2283 audit's named
  blocker, and the current repository docs that state required-status/security
  gate policy. A future governance doc that makes the same kind of claim must
  add an explicit owner in the PR that introduces it.
- Outside-set behavior: unlisted Markdown remains test-free by default, which is
  the cheap side for normal docs. Unlisted non-Markdown or unknown runtime
  assets retain the existing fail-closed/FULL behavior.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/select_impacted_tests.py` explicit owner map.
- Replaced-path behaviors: listed governance docs select
  `tests/test_security_guardrails_workflow.py`; ordinary Markdown docs remain
  test-free; missing owner files still escalate to FULL.
- Guard-relevant fields: changed path string, explicit owner tuple, owner file
  existence.
- Caller x input shape: unit-gate passes newline-separated changed paths from
  git diff; direct tests pass single changed-path fixtures.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - selector has no env/config fallback.
- Explicit value probe: `tests/test_select_impacted_tests.py` fixture paths for
  each mapped governance doc.
- Absent value probe: existing missing-owner fixture still proves FULL fallback.
- Default-session/default-context probe: N/A - no session context.
- Side-effect ordering: N/A - pure selector, no mutation.

### Files touched

- `plans/PR-Unit-Gate-Governance-Selector.md`
- `scripts/select_impacted_tests.py`
- `tests/test_select_impacted_tests.py`

## Mechanism

`scripts/select_impacted_tests.py` keeps its existing import-graph selector and
explicit-owner fallback. This slice adds the current CI/security governance docs
to `EXPLICIT_TEST_OWNERS`, pointing them at
`tests/test_security_guardrails_workflow.py`, and removes a duplicate
`scripts/check_ai_reconciliation_live.py` entry. The existing owner-file
existence check continues to escalate stale owners to FULL.

## Intentional

- This does not promote `unit-gate` to branch protection. It only closes the
  selector blocker identified before promotion.
- This does not derive doc ownership from prose. The governance-doc set is
  closed and enumerated for this slice because regular Markdown remains
  intentionally test-free.

## Deferred

- Re-run the #2283 required workflow enrollment decision after selector coverage
  is proven in CI.
- Decide branch-protection `strict` behavior separately from selector coverage.
- Decide whether `pre-push-audit` needs a safe PR-side docs/test consistency
  probe.

Parked hardening: none.

## Verification

- `/tmp/atlas-pr2259-venv/bin/python -m pytest tests/test_select_impacted_tests.py -q`
  -- passed locally, 61 passed.
- `/tmp/atlas-pr2259-venv/bin/python -m pytest tests/test_security_guardrails_workflow.py tests/test_select_impacted_tests.py -q`
  -- passed locally, 96 passed.
- `python3 -m py_compile scripts/select_impacted_tests.py` -- passed locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Unit-Gate-Governance-Selector.md` | 145 |
| `scripts/select_impacted_tests.py` | 18 |
| `tests/test_select_impacted_tests.py` | 35 |
| **Total** | **198** |
