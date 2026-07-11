# PR-Pre-Push-Caller-Hints-Timeout

## Why this slice exists

PR #2061's privacy vertical proof is code-green and reconciled, but its
`pre-push-audit` check is red because GitHub cancels the local-review bundle at
the job's hard 10-minute limit while `audit_cross_layer_callers.py` is running.
The exact-head run
`https://github.com/canfieldjuan/ATLAS/actions/runs/29134149285` passed every
preceding audit and then emitted only `==> Cross-layer caller hints` before
cancellation. This workflow/process slice is justified by that failed product
run and implements the operator's explicit instruction to remove the blocker
without modifying #2061's privacy code.

The same #2061 tree reproduces the cost locally: 111 changed Python symbols,
2,893 searchable tracked code files, 813 reported references, and 202.19 seconds
for the caller-hint audit alone. Raising the workflow timeout would preserve the
underlying changed-symbol x repository scan and make the next large guard PR fail
later, so this slice fixes the algorithmic root.

### Problem-derived contract

- Root cause: `build_hints` calls `find_references` once per changed symbol, and
  each call rereads every non-diff tracked code file and retokenizes every Python
  file. Runtime therefore scales as O(changed symbols x repository bytes): #2061
  performs roughly 321,000 file visits before pattern filtering. The GitHub
  runner cannot finish that advisory scan inside ten minutes. This slice fixes
  the root; the timeout is a consequence, not the repair target.
- Correct fix must touch/change: `scripts/audit_cross_layer_callers.py` must
  index candidate symbol-name lines for all changed symbols in one repository
  pass, then apply each symbol's existing class/function/method patterns to only
  those candidates. It must preserve output ordering, reference counts, path
  safety, changed-symbol detection, Python token filtering, malformed-token
  fallback, non-Python comment filtering, Unicode-decode skips, and advisory
  exit behavior. `tests/test_audit_cross_layer_callers.py` must prove multiple
  symbols share one read/tokenization pass and that references/noise retain the
  existing semantics. The actual #2061 tree must reproduce 111 symbols and 813
  references with materially lower runtime.
- Must not change: `.github/workflows/pre_push_audit.yml` timeout or trusted-base
  security boundary, `scripts/local_pr_review.sh` check ordering/pass-fail
  semantics, branch protection, reviewer gates, privacy code/tests/PR #2061,
  watcher infrastructure, or any product surface.

## Scope (this PR)

Ownership lane: workflow/pre-push-audit-performance
Slice phase: Workflow/process

1. Replace per-symbol repository rescans with a single batched candidate index
   in the existing cross-layer caller-hint auditor.
2. Add fixture coverage for shared file reads, multiple function/method/class
   symbols, comment/noise rejection, and malformed-Python fallback.
3. Benchmark the real #2061 head against current `origin/main` and compare its
   symbol/reference output to the captured baseline.

Max files: 3

### Review Contract

- Acceptance criteria:
  - [ ] All searchable tracked files are read at most once per audit invocation,
        independent of the number of changed symbols.
  - [ ] Python files are tokenized once and candidate lines are indexed by the
        requested symbol names; malformed Python retains the existing all-line
        fallback rather than silently disappearing.
  - [ ] Non-Python code keeps the existing blank/comment-line exclusions and
        class/function/method reference patterns remain unchanged.
  - [ ] Multiple changed symbols with the same name each receive the same
        matching references without duplicate file reads.
  - [ ] Unicode-decode failures remain advisory skips and unsafe repository
        paths still fail closed.
  - [ ] The CLI output order and summary counts remain stable for equivalent
        input; the captured #2061 baseline remains 111 changed symbols and 813
        non-diff references.
  - [ ] The real #2061 audit completes locally in under 20 seconds, at least a
        10x improvement over the measured 202.19-second baseline.
  - [ ] The pre-push workflow timeout and trusted-base execution contract are
        untouched; no product or privacy behavior changes.
- Reachability proof: run the actual CLI entrypoint from a detached #2061
  worktree against `origin/main`, capture its emitted summary/reference hints,
  compare that output with the pre-change baseline, and record elapsed time.
- Affected surfaces: advisory cross-layer caller hints inside local and GitHub
  pre-push review bundles.
- Risk areas: false-negative caller hints, output-order drift, excessive memory
  from indexing, malformed-source handling, and trusted-base workflow behavior.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Files touched

- `plans/PR-Pre-Push-Caller-Hints-Timeout.md`
- `scripts/audit_cross_layer_callers.py`
- `tests/test_audit_cross_layer_callers.py`

## Mechanism

Collect the changed symbols exactly as today, group them by simple name, and
scan the sorted searchable file list once. For Python, tokenize each file once
and retain line numbers only for `NAME` tokens present in the requested-name
set. For non-Python code, retain only existing non-comment candidate lines and
extract requested identifier names with one combined escaped-name regex. If
Python tokenization fails, fall back to name extraction across all lines, which
preserves today's conservative behavior.

For each candidate name/line, apply the unchanged `reference_patterns` for all
changed symbols sharing that name and append a `Reference` to that symbol's
ordered bucket. Finally construct `CallerHint` values in original changed-symbol
order. This changes scan shape from one full repository traversal per symbol to
one traversal plus candidate pattern checks; it does not change what qualifies
as a reference.

## Intentional

- No workflow timeout increase: the measured root is repeated parsing, and a
  larger timeout would only postpone the same failure class.
- No external `rg` dependency or shell pipeline: the auditor keeps its portable
  Python/tokenize semantics and fixture-testable path/error behavior.
- A small in-memory candidate index is intentional. It stores only matching
  name/line pairs and rendered reference snippets, not whole repository text.
- Added/deleted Python files and decorator-only edits retain their documented
  advisory blind spots; this slice changes performance, not audit scope.

## Deferred

- Making advisory workflow checks required remains tracked in #2035 and is not
  decided by this performance repair.
- Any future semantic expansion of caller detection (imports, property reads,
  deleted symbols, decorators) requires its own accepted slice and precision
  fixtures.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_cross_layer_callers.py -q` (`15 passed`).
- The pre-push workflow's complete PR-review tooling test list plus
  `tests/test_audit_cross_layer_callers.py` (`449 passed`).
- Exact `phase-c4-scripts` maturity ratchet command from CI (passed; no new
  brittleness above `baseline_scripts.json`).
- Real #2061 CLI baseline from detached head `74d432fdb` against `origin/main`:
  `/usr/bin/time ... audit_cross_layer_callers.py origin/main` (`202.19s`, 111
  changed symbols, 813 non-diff references).
- Real #2061 CLI after the batched scan: the same command (`2.24s`, 111 changed
  symbols, 813 non-diff references; `cmp` confirmed byte-identical output).
- Plan sync check against `origin/main` (passed).
- `git diff --check` (passed).
- Cold reconstruction found no gap, forbidden touch, or untraced change.
- Pending before push: the managed local review bundle.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Pre-Push-Caller-Hints-Timeout.md` | 156 |
| `scripts/audit_cross_layer_callers.py` | 104 |
| `tests/test_audit_cross_layer_callers.py` | 138 |
| **Total** | **398** |
