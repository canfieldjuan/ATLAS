# PR-Reddit-Fit-Guard

## Why this slice exists

Third slice of the approved Reddit Listening v2 arc (#1931 comment
4872154794), after S1 (harness, db4a10d3d) and S2 (contract + prompt,
6e8f78a2f). S3 ships the enforcement half of the ruler: the runtime guard
that scans parsed fit output against the SAME catalogue the harness grades
and the prompt teaches, before anything model-written can be persisted
(S4) or rendered (S6). The guard adds only POLICY -- which codes block --
and the slice's centerpiece parity test pins that the policy partitions
the catalogue exactly, so a rule added in any later slice cannot silently
ship unclassified. Zero model calls, zero network, zero store/digest
changes. Under the diff cap.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Vertical slice

1. `atlas_reddit/fit_guard.py`: `GuardDecision(ok, codes)`;
   `BLOCKING_CODES` / `ADVISORY_CODES` frozensets (v1 policy: every
   catalogue family blocks; advisory is deliberately EMPTY but exists so
   the parity test forces an explicit classification decision for every
   future rule); `guard_fit_decision(FitDecision, *, pii_allowlist)` --
   reason and angle scanned SEPARATELY (the S1 anchor lesson), codes
   sorted/deduped/text-free.
2. `tests/test_atlas_reddit_fit_guard.py`: the parity test; both sides
   (clean yes/no decisions pass; each rule family blocks; reason guarded
   not just angle; angle-start greeting fires; max-length clean parse
   passes); allowlist parity with the harness; privacy of codes.
3. Housekeeping (separate first commit): archive the merged S2 plan and
   regenerate `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] Parity: `BLOCKING_CODES | ADVISORY_CODES == ALL_RULE_CODES` and
        the sets are disjoint -- proven by a test that fails when a rule
        lands upstream without a policy decision.
  - [ ] Both error directions probed: clean advisory output (yes and no
        verdicts, including maximal lengths) passes; every rule family
        blocks; the reason field is guarded, not only the angle.
  - [ ] Fields scanned separately (angle-start greeting probe).
  - [ ] Codes are stable, sorted, deduped, and carry no matched text.
  - [ ] The guard consumes parsed FitDecisions only -- contract
        enforcement stays in the parser; the guard adds policy.
- Reachability proof (#1952): the guard is exercised through the REAL
  parser (`parse_fit_decision` output feeds `guard_fit_decision` in every
  test); its persistence/rendering callers land in S4/S6 as approved --
  wiring deferral named per the arc. The harness CLI reachability pair
  from S1/S2 still runs unchanged (Verification).
- Affected surfaces: one new pure module + its test file + plans
  housekeeping. No store, digest, CLI, config, or credential surface.
- Risk areas: policy drift (the parity test is the counter); guard/
  harness scan divergence (impossible -- both call the same
  `scan_fit_text`).
- Reviewer rules triggered: R1, R2 (guard-shaped: both sides probed),
  R11 (zero new dependencies), R12 (test auto-enrolls via the workflow
  glob), R14 (parity + reachability named above).
- Test-adapter posture: nothing is faked; real parser -> real guard over
  the real catalogue.

### Files touched

- `atlas_reddit/fit_guard.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Fit-Guard.md`
- `plans/archive/PR-Reddit-Fit-Contract.md`
- `tests/test_atlas_reddit_fit_guard.py`

## Mechanism

`guard_fit_decision` scans `decision.reason` and `decision.angle` (when
present) separately via the shared `scan_fit_text`, unions the finding
codes, and returns `ok = no blocking code fired` plus the sorted, deduped
code tuple. Policy lives in two frozensets over `ALL_RULE_CODES`; v1
blocks everything because a flagged-but-rendered advisory state would put
the unsafe text on the digest anyway -- the empty `ADVISORY_CODES` exists
precisely so the parity test forces future rules through an explicit
decision. The `pii_allowlist` parameter mirrors the harness mechanism for
parity; runtime callers pass nothing.

## Intentional

- **Guard behind the parser, never instead of it**: shape/length/verdict
  rules stay in `parse_fit_decision`; the guard assumes a contract-valid
  FitDecision and adds claim-safety policy only.
- **Everything blocks in v1** -- a deliberate policy choice, not an
  accident, recorded in code comments and pinned by
  `test_all_families_block_in_v1`.
- **Redaction is S4's CHECK constraint**, not guard behavior: the guard
  decides, the store enforces guard_ok=0 rows carry empty reason/angle.
  Named here so the S4 reviewer sees the pairing.

## Deferred

- S4 store v5 + manual import (persists GuardDecision codes;
  guard-rejected rows flagged + REDACTED) -> S5 judge client -> S6
  runner + digest (renders guard_ok rows only), per the approved arc.

Parked hardening: none.

## Verification

- `.venv/bin/python -m pytest tests/test_atlas_reddit_fit_guard.py -q`:
  15 passed (parity; both sides; separate-field scan; allowlist parity;
  code privacy).
- Full package suite `.venv/bin/python -m pytest
  tests/test_atlas_reddit_*.py -q`: 499 passed.
- Harness CLI reachability pair unchanged:
  `python scripts/evaluate_atlas_reddit_fit.py --cases
  tests/fixtures/atlas_reddit_fit_eval/cases.jsonl --predictions
  tests/fixtures/atlas_reddit_fit_eval/predictions_pass.jsonl
  --fail-on-eval-fail` exits 0; fail-file variant exits 1.
- ASCII byte-scan on new files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/fit_guard.py` | 60 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Fit-Guard.md` | 118 |
| `plans/archive/PR-Reddit-Fit-Contract.md` | 0 |
| `tests/test_atlas_reddit_fit_guard.py` | 157 |
| **Total** | **338** |
