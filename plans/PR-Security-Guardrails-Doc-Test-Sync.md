# PR-Security-Guardrails-Doc-Test-Sync

## Why this slice exists

Commit `a51e414` ("Record required status alignment", 2026-08-04, direct to
main) updated `docs/SECURITY_GUARDRAILS.md` to record the completed
branch-protection status alignment: live GitHub settings now contain every
registry-required context pinned to the GitHub Actions app source, so the
doc's old interim wording -- "live GitHub settings still require only ..."
plus the separate-REST-PATCH plan -- was deleted. The doc's pinning test,
`test_security_guardrails_docs_name_required_gitleaks_checks`, still asserts
the deleted sentences, so main is red on that test and every open PR's
`pre-push-audit` lane fails on inherited breakage. This slice re-pins the
test to the doc's current, deliberate content. The doc itself is not
touched: the doc change was the recorded operator action; the test is the
artifact that lagged.

### Problem-derived contract

- Root cause: a51e414 changed pinned documentation without updating the
  test that pins it, leaving main's unit suite red.
- Correct fix must touch/change: only the stale assertions inside
  `test_security_guardrails_docs_name_required_gitleaks_checks`, re-pinning
  the completed-alignment sentence and the verification command
  (`scripts/check_required_status_checks.py`) that replaced the removed
  interim wording. The REQUIRED_STATUS_CONTEXTS loop and the
  `Branch Protection Required Checks` workflow assertion stay untouched.
- Must not change: no doc content, no workflow, no guardrails behavior, no
  other test.

## Scope (this PR)

Ownership lane: security-guardrails-doc-test-sync
Slice phase: repair

1. Replace the two dead assertions in
   `test_security_guardrails_docs_name_required_gitleaks_checks` with pins
   to the doc's current completed-alignment wording and its verification
   command, with a dated comment naming the a51e414 alignment.

### Review Contract

- Acceptance criteria:
  - [ ] `python -m pytest tests/test_security_guardrails_workflow.py -q`
        passes on this branch (35 passed), settled by
        `tests/test_security_guardrails_workflow.py`.
  - [ ] The test still requires every `REQUIRED_STATUS_CONTEXTS` entry and
        the `Branch Protection Required Checks` workflow mention in the
        doc, settled by `tests/test_security_guardrails_workflow.py`.
- Reachability proof: the test file runs in the `pre-push-audit` unit
  sweep on every PR; this branch's run is the direct proof.
- Affected surfaces: one test function; no runtime code.
- Risk areas: pinning the wrong sentence would let future doc drift pass
  silently; mitigated by pinning the alignment claim verbatim plus the
  verification command path.
- Reviewer rules triggered: R5.

### Boundary-change enumeration

N/A - no boundary change.

### Deployed-config probing

N/A - no guard/config boundary change.

### Files touched

- `plans/PR-Security-Guardrails-Doc-Test-Sync.md`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

The test keeps its structural assertions (every registry-required context
named in the doc, the auditing workflow named) and swaps the two
assertions that pinned the pre-alignment interim state for pins on the
post-alignment state a51e414 recorded: the sentence stating live GitHub
settings contain every registry-required context pinned to the GitHub
Actions app source, and the `scripts/check_required_status_checks.py`
verification command the doc now documents in place of the old
REST-PATCH plan.

## Intentional

- The doc is treated as the source of truth: a51e414 was the operator's
  deliberate record of completed alignment, so the test moves to it, not
  the reverse.
- No broadening of the test into an exact-set audit; the doc itself states
  the checker proves registry coverage and source pinning, not an
  exact-set audit.

## Deferred

- Nothing; this is a complete repair of the single failing pin.

Parking predicate: anything beyond re-pinning the changed sentences is
parked.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_security_guardrails_workflow.py -q` -- 35 passed on this branch; 1 failed, 34 passed on unmodified main.
- `python -m py_compile tests/test_security_guardrails_workflow.py` -- passed.
- ASCII scan of the touched Python file -- no non-ASCII bytes.
- `python -m pytest -q tests/test_audit_plan_doc.py tests/test_audit_plan_code_consistency.py tests/test_audit_pr_plan_presence.py tests/test_check_diff_budget.py` -- 103 passed.
- `python scripts/check_boundary_change_enumeration.py --base origin/main --strict` -- OK.
- `python scripts/check_deployed_config_probing.py --base origin/main --strict` -- OK.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Security-Guardrails-Doc-Test-Sync.md` | 110 |
| `tests/test_security_guardrails_workflow.py` | 11 |
| **Total** | **121** |

## Cold diff reconstruction

Gaps first: no contract gaps found in the current diff.

- The single test function swaps its two dead pins for the doc's current
  completed-alignment sentence and verification command, with a dated
  comment tracing the change to the a51e414 status-alignment record; all
  other assertions in the file are untouched. Citation:
  `tests/test_security_guardrails_workflow.py:119`.

Scope check: the only change is the one test function plus this plan; no
doc, workflow, or runtime code moved.
