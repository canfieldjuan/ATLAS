# PR-Reviewer-Edge-Stop-Boundary

## Why this slice exists

The operator reported that Codex connector reviews keep expanding PRs with
adjacent edge-case, polish, and hardening threads after the current review round
has been addressed. The code-grounded audit found the boundary conflict:
`AGENTS.md` and `docs/REVIEWER_RULES.md` already tell reviewers to waive
out-of-scope hardening, but R13 / boundary-probe language can still promote
newly imagined adjacent probes into merge blockers. `AGENTS.md` also names
`scripts/codex_review_scope_policy.py` and
`tests/test_codex_review_scope_policy.py` as the deterministic fixture oracle,
so the rule change needs executable policy coverage.

### Problem-derived contract

- Root cause: The reviewer contract lacks a first-round novelty stop for
  adjacent edge-case/hardening/polish probes. R13 and boundary-probe proof are
  correct for the named defect class, but current wording and fixtures do not
  make adjacent classes non-blocking once the Review Contract is met.
- Correct fix must touch/change: Tighten the connector-facing scope language in
  `AGENTS.md`, tighten the canonical rule pack in `docs/REVIEWER_RULES.md`, and
  add deterministic fixture coverage in `scripts/codex_review_scope_policy.py`
  plus `tests/test_codex_review_scope_policy.py`.
- Must not change: Do not change product behavior, CI workflow enrollment,
  `live-reconciliation`, GitHub API behavior, or the ability to block real
  Review Contract, CI, security/privacy/money, data-correctness, existing
  behavior, claimed-mechanism, or material reachable correctness/back-
  compatibility/performance failures introduced by the diff.

## Scope (this PR)

Ownership lane: reviewer-boundary/workflow-process
Slice phase: Workflow/process

1. Add a hard reviewer boundary: adjacent edge-case/hardening/polish concerns
   are parked unless they falsify the current Review Contract or a material
   blocking surface.
2. Encode that boundary in the Codex scope-policy fixture oracle, while keeping
   concrete security/data/CI/material failures blocking.

### Review Contract

- Acceptance criteria:
  1. `AGENTS.md` tells Codex to stop at the Review Contract boundary and park
     adjacent hardening/polish/edge probes unless they invalidate this PR's
     contract, CI, existing behavior, safety, data correctness, material
     reachable correctness/back-compat/performance, or claimed mechanism.
  2. `docs/REVIEWER_RULES.md` applies the same boundary before missing-evidence,
     R13, and boundary-probe language can escalate adjacent probes.
  3. `scripts/codex_review_scope_policy.py` classifies an adjacent-but-non-
     falsifying edge probe as `WAIVE_OUT_OF_SCOPE`.
  4. The same policy keeps adjacent concrete security and material performance
     failures as `BLOCKER`.
  5. `tests/test_codex_review_scope_policy.py` locks the fixture names and CLI
     pass count for the added scenarios.
- Reachability proof: `python scripts/codex_review_scope_policy.py --self-test`
  and the focused pytest file exercise the executable policy oracle. No runtime
  product surface is introduced.
- Affected surfaces: `AGENTS.md`, `docs/REVIEWER_RULES.md`,
  `scripts/codex_review_scope_policy.py`,
  `tests/test_codex_review_scope_policy.py`.
- Risk areas: reviewer severity boundary, fixture drift, accidentally weakening
  real material blockers.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: Codex review disposition classifier in
  `scripts/codex_review_scope_policy.py`.
- Replaced-path behaviors: adjacent edge probes without a concrete falsifying
  path move from default `MAJOR` fallback to `WAIVE_OUT_OF_SCOPE`.
- Guard-relevant fields: `adjacent_to_scope`, `invalidates_review_contract`,
  `breaks_existing_behavior`, `red_ci`, `explicitly_claimed_mechanism_false`,
  `impact`, `concrete_failure_path`.
- Caller x input shape: synthetic fixture dictionaries consumed by
  `classify_finding`.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no deployed config.
- Explicit value probe: N/A - no deployed config.
- Absent value probe: N/A - no deployed config.
- Default-session/default-context probe: N/A - no deployed config.
- Side-effect ordering: N/A - pure classifier/docs change.

### Files touched

- `AGENTS.md`
- `docs/REVIEWER_RULES.md`
- `plans/PR-Reviewer-Edge-Stop-Boundary.md`
- `scripts/codex_review_scope_policy.py`
- `tests/test_codex_review_scope_policy.py`

## Mechanism

The prose contract now states that the Review Contract boundary wins over
adjacent edge exploration. The executable policy adds an early
`adjacent_to_scope` classification: if the finding does not invalidate the
current Review Contract, existing behavior, CI, safety/data surfaces, or a
claimed mechanism, it is waived out of scope. A paired fixture proves a concrete
adjacent security failure still blocks.

## Intentional

- No `live-reconciliation` change: current code already allows a PR body that
  honestly acknowledges open findings to pass the live contradiction check.
- No broad reviewer rewrite: this slice only changes the adjacent-probe stop
  boundary and fixture oracle.

## Deferred

None.

Parked hardening: none.

## Verification

- `python scripts/codex_review_scope_policy.py --self-test` - PASS (13
  fixtures).
- `pytest tests/test_codex_review_scope_policy.py` - PASS (5 tests).
- `python scripts/sync_pr_plan.py plans/PR-Reviewer-Edge-Stop-Boundary.md --check` - PASS.
- Pending before push: `bash scripts/local_pr_review.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 22 |
| `docs/REVIEWER_RULES.md` | 27 |
| `plans/PR-Reviewer-Edge-Stop-Boundary.md` | 141 |
| `scripts/codex_review_scope_policy.py` | 62 |
| `tests/test_codex_review_scope_policy.py` | 7 |
| **Total** | **259** |
