# PR-Unit-Gate-Selector-Absent-Fallback

## Why this slice exists

#2207 merged the impacted-test selector (`72658931b`). For a `pull_request`
event GitHub takes the **workflow definition** from the merged ref but the
checkout step uses the **PR head** tree. Every open branch cut before #2207
landed therefore runs a workflow that calls
`scripts/select_impacted_tests.py` against a tree that does not contain it:

    python: can't open file '.../scripts/select_impacted_tests.py': [Errno 2]

Measured immediately after the merge: **20+ open PRs fail `unit-gate` on this**,
including #2195, #2199, #2200, #2201, #2208 and every Dependabot PR. None of
them did anything wrong; main advanced under them.

### Problem-derived contract

- Root cause: the workflow assumes the selector exists in the checked-out tree,
  which is only true for branches created after #2207.
- Correct fix must: treat an absent selector as one more input the gate cannot
  map, and escalate to FULL -- the same failure direction the selector already
  takes for every other unmappable input. It must not require unrelated
  sessions to rebase, which would mean editing other lanes (3a.1 forbids it).
- Must not change: selection behavior when the selector IS present, the ratchet
  semantics, or the committed baseline.
- **Does change, deliberately:** which baseline the growth guard resolves. It
  compared against the base branch's current baseline; it now compares against
  the **merge base**. Same root cause as the ENOENT -- a branch cut before main
  moved gets failed for something main did. If main shrinks the baseline after a
  branch forks, the branch's larger baseline reads as growth and
  `added_baseline_entries()` fails a PR that added nothing, and the
  selector-absent heads this slice unblocks are the branches most likely to hit
  it. The ratchet's *meaning* is unchanged -- "did THIS PR grow the baseline" --
  the merge base is simply the correct reference for that question.

## Scope (this PR)

Ownership lane: ci-gates
Slice phase: Production hardening
Max files: 3

1. `.github/workflows/unit_gate.yml`: if `scripts/select_impacted_tests.py` is
   absent from the checked-out tree, write `FULL` to the selection file instead
   of invoking it.

### Review Contract

- Acceptance criteria:
  1. On a PR head lacking `scripts/select_impacted_tests.py`, the Select step
     exits 0 and the selection file contains `FULL` -- settled by the
     `unit-gate` run on any of the 20+ affected PRs after this lands.
  2. On a PR head containing the selector, the step still invokes it and the
     selection is unchanged -- settled by the `unit-gate` run on this PR, whose
     own head does contain it.
  3. The FULL branch calls `check_unit_gate.py` with only `--baseline` and
     `--base-baseline`, both of which exist on pre-#2207 heads, so the fallback
     cannot fail on a missing flag.
  4. `tests/test_unit_gate_selector_fallback.py` executes the Select step's own
     `run:` body in a temp tree and asserts `FULL` when the selector is absent
     -- settled by running that file, and by reverting the guard, which fails 3
     of its tests.
  5. With the selector present the else-branch invokes it rather than escalating:
     the fixture installs a sentinel selector emitting a token the guard cannot
     produce, so "non-empty output" cannot pass for "the selector ran".
  6. The growth guard resolves the merge-base baseline, proven against a **real**
     git repository where main shrinks the baseline after the branch forks --
     reverting the resolution fails that test.
- Reachability proof: the `unit-gate` workflow on any affected PR; the observable
  output is the `--- selection ---` echo and a non-zero-length gate run.
- Affected surfaces: CI only.
- Risk areas: silently running FULL when selection was possible -> bounded by
  criterion 2, which proves the selector still runs when present.
- Reviewer rules triggered: R1, R12, R14.

### Files touched

- `.github/workflows/unit_gate.yml`
- `plans/PR-Unit-Gate-Selector-Absent-Fallback.md`
- `tests/test_unit_gate_selector_fallback.py`

## Mechanism

A `[ ! -f ... ]` guard before the invocation. Absent selector is treated exactly
as the selector treats an unmappable input: escalate to FULL rather than guess.

## Intentional

- **Fallback is FULL, not empty.** An absent selector means we know nothing
  about the change's blast radius, and the whole design principle is that
  uncertainty runs more tests, never fewer.
- **No rebase requested of anyone.** Requiring 20+ branches to rebase would push
  the cost onto sessions that did nothing wrong and would mean cross-lane edits.

## Deferred

- Nothing.

Parked hardening: none.

## Verification

Run `tests/test_unit_gate_selector_fallback.py` under pytest: 5 passed. Three 3i
probes, each reverting one behavior and confirming the matching test fails:

- guard reverted to the pre-#2212 form -> 3 fail
- guard made unconditional (swallowing the normal path) -> 2 fail
- merge-base resolution reverted to the base ref -> the end-to-end test fails

The merge-base test builds a real git repository rather than stubbing git,
because the defect it prevents lives entirely in history resolution and a stub
cannot express "main moved after the branch forked".

    python -c "import yaml; yaml.safe_load(open('.github/workflows/unit_gate.yml'))"

- Diagnosis confirmed against a real failing run (`30186458544`, #2210) and by
  checking the head trees of every open PR for the file.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/unit_gate.yml` | 33 |
| `plans/PR-Unit-Gate-Selector-Absent-Fallback.md` | 126 |
| `tests/test_unit_gate_selector_fallback.py` | 171 |
| **Total** | **330** |
