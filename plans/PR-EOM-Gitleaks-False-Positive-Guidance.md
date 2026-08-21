# PR-EOM-Gitleaks-False-Positive-Guidance

## Why this slice exists

H-20 in [Billing & Payments -- Hardening & Deferred #2363](https://github.com/canfieldjuan/ATLAS/issues/2363#issuecomment-5336686567) was discovered while resolving the Gitleaks false-positive repair in #2436. That repair deliberately documented two controlled **baseline**-growth paths: provider credential rotation/revocation and reviewed exact scanner false-positive evidence. The guard's initial shared unlabeled baseline/ignore response advertised the reviewed-false-positive route for ignore growth too, even though the canonical policy excludes ignore-list exceptions. The P1 Codex finding on #2438 requires the response to distinguish those trigger shapes without changing the guard's verdicts.

### Problem-derived contract

- Root cause: `evaluate_baseline_rotation()` has one shared missing-label response for two distinct triggers. It treats `.gitleaksignore` growth as if it could use the reviewed exact scanner false-positive baseline route, while `docs/SECURITY_GUARDRAILS.md` limits that route to a candidate baseline `Fingerprint` and expressly excludes ignore-list exceptions. The guard's admission decision is already correct; its human-facing recovery guidance is not.
- Correct fix must touch/change: split only the missing-label guidance in `scripts/check_gitleaks_baseline_rotation.py` using the existing `ignore_requires_rotation` predicate. A baseline-only change names rotation/revocation or the documented reviewed-false-positive route. Any added ignore fingerprint, including a combined baseline+ignore change, names credential rotation/revocation and states that reviewed false-positive evidence does not authorize ignore growth. Extend focused tests to pin all three rejected shapes.
- Must not change: label enforcement; baseline/ignore membership rules; allowed rotation paths; Gitleaks configuration, history, and workflow behavior; product/financial behavior; the legacy Monthly Invoice task; the already-merged #2436 baseline records; and any other active PR lane.

## Scope (this PR)

Ownership lane: eom/billing-payments-security-hardening
Slice phase: Production hardening

Max files: 3

1. Keep all unlabeled baseline and ignore-growth decisions rejected, but make their error states accurately follow the policy: the two routes for baseline-only growth and credential rotation/revocation only for ignore growth.
2. Add focused assertions for rejected baseline-only, ignore-only, and combined baseline+ignore shapes, and run the script against the historical #2436 baseline diff without the required label to prove the CLI prints the corrected baseline guidance and returns nonzero.

### Files touched

- `plans/PR-EOM-Gitleaks-False-Positive-Guidance.md`
- `scripts/check_gitleaks_baseline_rotation.py`
- `tests/test_check_gitleaks_baseline_rotation.py`

### Review Contract

- Acceptance criteria:
  1. `evaluate_baseline_rotation()` still returns `allowed=False` for an unlabeled trusted-base baseline change and its `reason` contains `security-rotation`, provider credential rotation/revocation, and reviewed false-positive evidence; settled by `tests/test_check_gitleaks_baseline_rotation.py::test_rejects_baseline_change_without_rotation_label`.
  2. An unlabeled added `.gitleaksignore` fingerprint, alone or combined with a baseline change, still returns `allowed=False`, requires provider credential rotation/revocation, and states that reviewed false-positive evidence does not authorize ignore growth; settled by `tests/test_check_gitleaks_baseline_rotation.py::test_rejects_ignore_growth_without_rotation_label` and `tests/test_check_gitleaks_baseline_rotation.py::test_rejects_combined_baseline_and_ignore_growth_with_ignore_recovery_guidance`.
  3. All existing labeled, initial-adoption, membership-preservation, and disallowed-path decisions remain unchanged; settled by the full focused test file.
  4. The real CLI prints the corrected message and exits 1 for the historic #2436 baseline change when labels are absent; settled by `python scripts/check_gitleaks_baseline_rotation.py --base-ref 35e9ac75926967091d92a797764dee8a6434dd80 --head-ref d5ce8bb2eceff8eb3b68ec2cf02fbb559cf86ca1 --labels-json '[]'`.
- Reachability proof: existing `scripts/check_gitleaks_baseline_rotation.py` is the same CLI invoked by the trusted `Gitleaks baseline growth guard`; criterion 4 executes its real `main()` path against a historical baseline-growth range without writing data.
- Affected surfaces: the baseline-growth guard's operator-facing missing-label rejection reasons and its direct unit tests only. GitHub workflow, labels, decision verdicts, and baseline data are intentionally unaffected.
- Risk areas: misleading recovery instructions, a policy-forbidden ignore exception, accidental relaxation of a security gate, source-policy drift, and false-positive test coverage.
- Reviewer rules triggered: R1, R2, R3, R10, R14.

### Boundary-change enumeration

The decision predicate is preserved; this is an operator-guidance change on its
rejected boundary response.

- Boundary path/seam: `evaluate_baseline_rotation()`'s unlabeled controlled-growth rejection at `scripts/check_gitleaks_baseline_rotation.py:83-91`.
- Replaced-path behaviors: the initial shared two-path instruction is replaced with a baseline-only two-path instruction and an ignore-growth rotation-only instruction; every result remains rejected until the required label is present.
- Guard-relevant fields: `changed_paths`, `labels`, `base_has_baseline`, and added ignore fingerprints remain read-only inputs and retain their current verdict behavior.
- Caller x input shape: trusted baseline change without added ignore x absent label -> rejected with two-path guidance; ignore growth alone or with baseline growth x absent label -> rejected with rotation-only guidance; unchanged/initial-adoption/labeled and disallowed-path shapes are preserved by focused tests.

### Closure declaration

N/A -- this slice does not add or edit a decision-driving member set. It reads
the existing label/path/fingerprint inputs but changes no membership or verdict.
The two English recovery paths name the already-documented alternatives; they
do not form an admission set. The plan's finite probe inventory is derived from
the existing single unlabeled-rejection seam, and every other input continues
through unchanged code covered by the focused suite.

### Deployed-config probing

N/A -- this change does not read, add, or alter deployment configuration. The
existing no-label test is the absent-label admission shape; no side effect is
possible because the guard reads Git refs and exits before any write.

## Mechanism

The existing unlabeled rejection branch keeps the same `Decision(False, ...)`
and all fields. It chooses its message from the existing `ignore_requires_rotation`
predicate: baseline-only growth is a concise two-path instruction (rotate/revoke
or documented reviewed false-positive evidence), while any added ignore fingerprint
requires rotation/revocation and explicitly says the reviewed-false-positive route
does not authorize ignore growth. Focused tests cover baseline-only, ignore-only,
and combined shapes while the rest of the existing test file holds the unchanged
verdict/membership/path behavior. A no-label CLI run over the historic #2436 diff
proves the baseline-only user-facing `main()` output, not only a direct helper call.

## Intentional

- Do not make reviewed false positives a new label, a weaker admission rule, or a configurable bypass. Both routes still require the existing `security-rotation` label and all existing trusted-base checks.
- Do not describe reviewed false-positive evidence as an ignore-growth alternative. The canonical policy expressly limits it to exact scanner-generated baseline fingerprints.
- Do not repeat the full evidence checklist in the error string. `docs/SECURITY_GUARDRAILS.md` remains the canonical policy; the message names the path and points operators to documented evidence rather than duplicating a drift-prone checklist.
- Do not fold the discovered legacy monthly-invoicing risk into this security message PR. It is a separate financial behavior and must be assessed as its own Billing & Payments safety slice.

## Deferred

- The legacy monthly-invoice task's possible conflict with the approved billing-run workflow is tracked in [#2363](https://github.com/canfieldjuan/ATLAS/issues/2363#issuecomment-5335394858). It requires a separate financial-safety decision and is not changed here.
- The repository-wide OSV dependency backlog remains in the existing security-scanner hardening register; it is unrelated to this message-only guard repair.

Parking predicate: any change that relaxes admission, changes Gitleaks
membership/configuration/workflows, or alters financial behavior is parked as a
separate slice. Parked hardening: none within this message/test-only predicate.

## Verification

- `python -m pytest tests/test_check_gitleaks_baseline_rotation.py -q` -- 17 passed.
- `python -m pytest tests/test_check_gitleaks_baseline_rotation.py tests/test_security_guardrails_workflow.py tests/test_security_policy_docs.py -q` -- 75 passed, 43 subtests passed.
- The no-label historical CLI probe in the Review Contract -- exit 1 with the baseline-only two-path guidance.
- `python -m ruff check scripts/check_gitleaks_baseline_rotation.py tests/test_check_gitleaks_baseline_rotation.py` -- passed.
- `git diff --check` -- passed.
- `python -m black --check scripts/check_gitleaks_baseline_rotation.py tests/test_check_gitleaks_baseline_rotation.py` -- known pre-existing failure: both files require reformatting on clean `main` as well; formatting is intentionally not widened into this message-only slice.
- `python scripts/sync_pr_plan.py plans/PR-EOM-Gitleaks-False-Positive-Guidance.md --check` -- passed.
- Managed `scripts/push_pr.sh` local PR review bundle is required before publication of the P1 repair; its final status is recorded in the PR body after publication.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Gitleaks-False-Positive-Guidance.md` | 112 |
| `scripts/check_gitleaks_baseline_rotation.py` | 21 |
| `tests/test_check_gitleaks_baseline_rotation.py` | 20 |
| **Total** | **153** |
