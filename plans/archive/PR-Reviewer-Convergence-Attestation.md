# PR Reviewer Convergence Attestation

## Why this slice exists

Review rounds can drift because the PR body can claim reconciliation with vague
prose while local review only checks for a globally resolved-looking AI section.
The local wrapper also assumes a `python` executable exists, which makes the
mechanical gate host-dependent.

This slice is intentionally over the 400-line soft cap because the enforcement
change is only useful if the PR-body contract, reconciliation parser, live
thread-history check, local/open/push wrapper paths, docs, and regression tests
land together. Splitting those pieces would leave at least one mandatory PR path
accepting or publishing a body shape that another mandatory path rejects.

### Problem-derived contract

The fix must make full PR bodies carry structured review disposition evidence
and explicit mechanical verification evidence before push. It must reuse the
existing PR body, local review, and live reconciliation path; it must not add a
second reviewer gate or mutate live branch protection.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

- Workflow/process slice.
- Tighten `scripts/audit_ai_reconciliation.py` around structured dispositions.
- Tighten `scripts/audit_pr_body.py` around required PR body sections and
  mechanical verification evidence.
- Tighten `scripts/check_ai_reconciliation_live.py` so `no-findings` cannot
  survive after Codex review-thread history exists and a single unrelated
  disposition-only ledger cannot stand in for resolved review-thread history.
- Keep `scripts/local_pr_review.sh` as the single local review funnel and make
  `scripts/open_pr.sh`, `scripts/push_pr.sh`, `scripts/pre_push_audit.sh`, and
  the shared ASCII checker use the same Python interpreter fallback.
- Honor every canonical Dependabot author identity in the local review
  reconciliation funnel.
- Map the shared ASCII shell checker to its owning regression test so the unit
  gate stays scoped to this slice instead of escalating to unrelated full-suite
  failures.

### Files touched

- `AGENTS.md`
- `docs/REVIEWER_RULES.md`
- `extracted/_shared/scripts/check_ascii_python.sh`
- `plans/PR-Reviewer-Convergence-Attestation.md`
- `scripts/audit_ai_reconciliation.py`
- `scripts/audit_pr_body.py`
- `scripts/check_ai_reconciliation_live.py`
- `scripts/local_pr_review.sh`
- `scripts/open_pr.sh`
- `scripts/pre_push_audit.sh`
- `scripts/push_pr.sh`
- `scripts/select_impacted_tests.py`
- `tests/test_audit_ai_reconciliation.py`
- `tests/test_audit_pr_body.py`
- `tests/test_check_ai_reconciliation_live.py`
- `tests/test_local_pr_review.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_pre_push_audit.py`
- `tests/test_push_pr_wrapper.py`
- `tests/test_select_impacted_tests.py`

### Review Contract

- Reviewer rules triggered: R2, R10.
- A full PR body without `## AI reconciliation` fails local/body audit.
- A full PR body without `## Mechanical verification` fails local/body audit.
- A reconciliation section with vague global claims fails unless it records
  `no-findings` or allowed finding dispositions.
- A reconciliation finding bullet fails unless it names a root decision and
  carries exactly one allowed disposition.
- Live reconciliation fails when `no-findings` is present after scoped Codex
  review-thread history exists.
- Live reconciliation fails when a full PR-body ledger claims clear but does
  not name every scoped Codex review-thread root decision.
- Live reconciliation also correlates disposition-only structured ledgers
  against resolved Codex review-thread history.
- Live reconciliation requires exact root-decision correlation or a sufficiently
  long normalized-title match, so tiny substrings cannot satisfy a resolved
  thread disposition.
- Live reconciliation correlates resolved-thread dispositions against the same
  canonical, top-level, unfenced `## AI reconciliation` section used by the
  PR-body audit.
- A mechanical verification section fails unless at least one `Command:` line
  includes `Result:` and `Environment:`.
- Mechanical verification rejects placeholder commands/results and environments
  outside `local`, `Office PC`, or `CI`.
- Mechanical verification parses `Command`, `Result`, and `Environment` as
  explicit delimited fields and rejects missing or duplicate fields.
- Unit-gate test ownership for `scripts/audit_ai_reconciliation.py` includes
  the direct caller suites that import or load its parsing contract.
- Unit-gate test ownership for `scripts/audit_pr_body.py` includes the live
  reconciliation and mandatory local/open/push wrapper caller suites.
- `Docs-only: true` and Dependabot body exemptions keep their existing behavior.
- Every Dependabot identity accepted by the shared PR change policy keeps that
  exemption through local review's AI reconciliation audit.
- The wrappers run on hosts with `python3` but no `python`.

## Mechanism

`scripts/audit_ai_reconciliation.py` now treats the reconciliation body as a
small structured ledger: `no-findings` or finding bullets with `fixed-in`,
`waived-duplicate`, `waived-out-of-scope`, `waived-speculative`, `waived-nit`,
or `not-applicable`.

`scripts/audit_pr_body.py` adds `AI reconciliation` and `Mechanical
verification` to the ordered required section list, then validates mechanical
verification bullets for command/result/environment evidence, including an
actual non-placeholder command value.

`scripts/local_pr_review.sh` requires the AI reconciliation audit whenever a PR
body file is supplied, except for the existing Dependabot body exemption.
`scripts/check_ai_reconciliation_live.py` rejects stale `no-findings` when
resolved Codex review-thread history proves findings existed, and it rejects a
clear full-body ledger, including a structured disposition-only ledger, that
omits any scoped Codex review-thread title.
The correlator requires an exact normalized root-decision match, or a long
multi-token containment match, so a one-character disposition cannot stand in
for a real review-thread title.
The live correlator now derives disposition roots from the canonical PR-body
section parser so indented or fenced heading-like examples cannot satisfy a
different ledger than the PR-body audit validates.

`scripts/audit_pr_body.py` parses mechanical verification command bullets as
delimited fields, so command text cannot satisfy a missing `Result:` field and
duplicate labels fail closed.

`scripts/local_pr_review.sh`, `scripts/open_pr.sh`, `scripts/push_pr.sh`,
`scripts/pre_push_audit.sh`, and `extracted/_shared/scripts/check_ascii_python.sh`
use `${PYTHON:-python3}` so the mechanical path is not blocked on a missing
`python` shim. `scripts/select_impacted_tests.py` keeps the shared ASCII checker
owned by `tests/test_pre_push_audit.py`, avoiding an unrelated full-suite
escalation for this local-gate wrapper change.
The unit selector also keeps reconciliation parser changes enrolled with
`tests/test_audit_pr_body.py`, `tests/test_check_ai_reconciliation_live.py`,
and the mandatory local/open/push wrapper suites, not only the parser's direct
test file.

## Intentional

- No live branch-protection settings change.
- No new GitHub Actions workflow; existing local review and PR body contract
  wiring is the enforcement point.
- The live GitHub thread reality check stays in `live-reconciliation`.

## Deferred

- Branch protection alignment and any future meta-gate remain separate slices.
- Automatically syncing live thread IDs into the PR body remains out of scope.
- Parked hardening: none.

## Verification

- `python3 -m py_compile scripts/audit_ai_reconciliation.py scripts/audit_pr_body.py scripts/check_ai_reconciliation_live.py scripts/select_impacted_tests.py` -- passed locally.
- `python -m pytest tests/test_check_ai_reconciliation_live.py -q` -- passed locally, 70 passed.
- `/tmp/atlas-pr2259-venv/bin/python -m pytest tests/test_check_ai_reconciliation_live.py tests/test_audit_pr_body.py tests/test_select_impacted_tests.py -q` -- passed locally, 190 passed.
- `python -m pytest tests/test_check_ai_reconciliation_live.py tests/test_select_impacted_tests.py tests/test_local_pr_review.py -q` -- passed locally, 145 passed.
- `python -m pytest tests/test_audit_ai_reconciliation.py tests/test_audit_pr_body.py tests/test_local_pr_review.py tests/test_open_pr_wrapper.py tests/test_push_pr_wrapper.py tests/test_pre_push_audit.py tests/test_check_ai_reconciliation_live.py tests/test_select_impacted_tests.py -q` -- passed locally, 283 passed.
- `python3 scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` -- passed locally after reducing `scripts/audit_ai_reconciliation.py` back under its ratchet.
- `bash scripts/local_pr_review.sh --current-pr-body-file <body>` -- passed locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 31 |
| `docs/REVIEWER_RULES.md` | 6 |
| `extracted/_shared/scripts/check_ascii_python.sh` | 18 |
| `plans/PR-Reviewer-Convergence-Attestation.md` | 190 |
| `scripts/audit_ai_reconciliation.py` | 132 |
| `scripts/audit_pr_body.py` | 163 |
| `scripts/check_ai_reconciliation_live.py` | 192 |
| `scripts/local_pr_review.sh` | 37 |
| `scripts/open_pr.sh` | 7 |
| `scripts/pre_push_audit.sh` | 21 |
| `scripts/push_pr.sh` | 3 |
| `scripts/select_impacted_tests.py` | 30 |
| `tests/test_audit_ai_reconciliation.py` | 123 |
| `tests/test_audit_pr_body.py` | 177 |
| `tests/test_check_ai_reconciliation_live.py` | 150 |
| `tests/test_local_pr_review.py` | 111 |
| `tests/test_open_pr_wrapper.py` | 8 |
| `tests/test_pre_push_audit.py` | 19 |
| `tests/test_push_pr_wrapper.py` | 8 |
| `tests/test_select_impacted_tests.py` | 30 |
| **Total** | **1456** |
