# PR-Unit-Gate-File-Baseline-Escalation

## Why this slice exists

The staged EOM Invoicing MCP payment-response compatibility slice cannot pass
the repository's local Unit Gate as currently implemented. Its selector reaches
four OAuth/MCP test modules that are recorded as bare file-level baseline
failures. Those modules pass when executed in isolation, so the scoped ratchet
demands that their baseline entries be removed. The same gate's required full
suite then proves that all four still fail during collection because legacy
tests install a fake `mcp` package in `sys.modules` before OAuth imports run.

The root cause is selector unsoundness: a scoped pass can establish a test-file
result, but it cannot establish that a bare file-level full-suite collection
failure is fixed. The selector must therefore choose the full suite whenever
its otherwise-scoped result includes an exact bare test-file baseline entry.
This hardening slice is justified because it blocks the current Billing &
Payments vertical prerequisite without weakening its financial contract.

### Problem-derived contract

- Root cause: a selected bare file-level baseline entry represents a known
  full-suite collection failure, yet the scoped gate treats its isolated pass
  as evidence that the entry is stale.
- Correct fix must touch/change: the impacted-test selector must inspect exact
  bare test-file entries in the committed baseline after it derives a scoped
  selection, then return `FULL` for an overlap.
- Must not change: the payment projection, financial behavior, known-failure
  baseline contents, unit-gate regression/growth predicate, test selection for
  exact node-level baseline entries, deployment configuration, or hosted test
  execution.

## Scope (this PR)

Ownership lane: developer-experience/ci
Slice phase: production hardening
Max files: 3

1. Escalate a scoped selector result to `FULL` when it includes a test path
   listed as a bare file-level known failure in `tests/unit_gate_baseline.txt`.
2. Prove the admission boundary with a fixture whose selected file has a bare
   baseline entry and a just-outside fixture whose baseline entry names a single
   test node in that same file.
3. Unblock the staged EOM MCP payment-response projection without accepting a
   weaker unit-gate baseline or changing financial code.

### Review Contract

- Acceptance criteria:
  - [ ] A selected fixture test file with a matching bare baseline line returns
    `FULL`; settled by
    `tests/test_select_impacted_tests.py::test_selected_file_level_baseline_escalates_to_full`.
  - [ ] A selected fixture test file with only a named test-node baseline entry
    stays scoped; settled by
    `tests/test_select_impacted_tests.py::test_node_level_baseline_does_not_escalate_to_full`.
  - [ ] The selector never changes the baseline and preserves its existing
    unresolvable-input `FULL` behavior; settled by the changed-file diff and
    the existing selector fixture suite.
  - [ ] The EOM staging branch's changed paths select `FULL` after this
    prerequisite merges; settled by the local selector command recorded in
    this plan's Verification section.
- Reachability proof: the production selector CLI and the local pre-push mirror
  both consume `scripts/select_impacted_tests.py`; the fixtures call its real
  `select` function and the staging branch later exercises the same CLI before
  its local full-suite gate.
- Affected surfaces: local/hosted unit-gate test selection and the Billing &
  Payments MCP-projection prerequisite's acceptance path.
- Risk areas: false-green scoped tests, unnecessary full-suite escalation, and
  inadvertent baseline or financial behavior changes.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R14.

### Boundary-change enumeration

- Admitted to `FULL`: a selected test path exactly equal to a non-comment,
  bare Python-file baseline entry.
- Not admitted to `FULL` by this rule: a baseline entry containing `::`, a
  baseline path outside the selected test set, blank/comment lines, and an
  absent baseline file.

### Guard class-closure declaration

- **OPEN:** baseline rows and selector-produced test paths are open text/path
  inputs; their membership cannot be exhaustively listed in this PR.
- **DERIVED:** each selector invocation derives the candidate set from the
  committed `tests/unit_gate_baseline.txt` with the existing bare-path grammar,
  then intersects it with the selector's derived impacted-test set.
- **DEFAULTED:** unrecognized, commented, blank, node-level, or unselected
  rows retain normal scoped selection because they are not a recorded
  full-suite collection failure. An unreadable baseline instead defaults to
  `FULL`, the safe direction because it prevents a false-green scope at the
  cost of only a local full-suite run.
- The property proof generates baseline line family × selected-path membership
  × whitespace wrapper and compares the result to a specification-derived
  oracle; it does not rely on a list of reported MCP/OAuth paths.

### Deployed-config probing

N/A - this is deterministic repository-local test selection; it does not read
credentials, feature flags, services, or deployed configuration.

### Files touched

- `plans/PR-Unit-Gate-File-Baseline-Escalation.md`
- `scripts/select_impacted_tests.py`
- `tests/test_select_impacted_tests.py`

## Mechanism

After the selector derives its ordinary set of impacted test files, it reads
only bare test-file entries from the committed unit-gate baseline. An exact
intersection escalates to `FULL` with an explanatory stderr message. The
existing full-run ratchet remains authoritative for baseline shrink proof; the
scoped path still catches new failures and remains fast for node-level debt.

## Intentional

- This does not repair every legacy module-level `sys.modules` fake. Those
  fakes are the underlying test-debt class, but changing them would broaden an
  infrastructure prerequisite into unrelated test rewrites.
- Escalating the affected selection costs one full local suite, which is the
  necessary proof because the baseline itself records a full-suite-only
  collection failure.
- A bare file-level baseline is intentionally treated differently from a
  node-level baseline: the former denotes a collection failure before a test
  node exists.

## Deferred

- The legacy process-global MCP fake cleanup is recorded in Billing & Payments
  Hardening & Deferred #2363, discovered by the EOM MCP-projection staging
  slice. This PR parks only that unrelated rewrite class; it fixes the
  selector's immediate false verdict.
- After this merges, rebase and rerun the staged EOM MCP payment-response
  projection before proceeding to its deployment/restart handoff (#2362).

Parked hardening: legacy MCP collection-fake cleanup (#2363).

## Verification

- `python -m py_compile scripts/select_impacted_tests.py
  tests/test_select_impacted_tests.py` — passed.
- `python -m pytest tests/test_select_impacted_tests.py -q` — 70 passed.
- `python scripts/select_impacted_tests.py --changed-file
  /tmp/eom-mcp-payment-projection-changed-files.txt --repo
  /home/juan-canfield/Desktop/Atlas-worktrees/eom-mcp-payment-projection` —
  `FULL`, naming exactly the four file-level MCP/OAuth baseline entries.
- `ruff check scripts/select_impacted_tests.py tests/test_select_impacted_tests.py
  --ignore F841` and `git diff --check` — passed.
- Managed `scripts/push_pr.sh` local gate remains the final acceptance evidence;
  no GitHub-hosted check is used as acceptance evidence.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Unit-Gate-File-Baseline-Escalation.md` | 159 |
| `scripts/select_impacted_tests.py` | 63 |
| `tests/test_select_impacted_tests.py` | 75 |
| **Total** | **297** |
