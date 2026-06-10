# PR-Semantic-Diff-Advisor-CI

## Why this slice exists

Four recent BLOCKERs were one class: a recognition surface changed and the
boundary case slipped -- #1439 (over-broad auth fold), #1446 (generic
first_response key captured auto-acks as resolutions), #1453 (fail-open
default on a new contract field), #1466 (stemmer/term-set asymmetry). The
defect itself needs judgment, but the TRIGGER is mechanical: each bug arrived
via one of four detectable diff patterns. This slice mechanizes the trigger --
an advisory CI check that detects those patterns in a PR diff and prints the
standing adversarial question for each -- so the question fires pre-push at
the builder instead of costing a red reconciliation-gate round-trip.

Companion to the structural sweep (PR #1470) and the invariant-test pack
steered to the deflection lane (issue #1471). This is the judgment-trigger
layer; it raises questions, never verdicts.

## Scope (this PR)

Ownership lane: review-workflow
Slice phase: workflow/process

1. Add `scripts/semantic_diff_advisor.py` -- stdlib-only detector. Pure
   detection functions take old/new source pairs; a thin git layer feeds them
   from the PR diff (merge-base vs HEAD). Advisory: exit 0 unless a future
   --strict flag is passed.
2. Add `tests/test_semantic_diff_advisor.py` -- unit tests whose fixtures are
   minimized replays of the four real BLOCKERs, plus quiet-control cases.
3. Add `.github/workflows/semantic_diff_advisor.yml` -- runs the unit tests
   (blocking: tool correctness gates) then the advisor against the PR base
   (non-blocking: tool findings advise).

### Files touched

- `.github/workflows/semantic_diff_advisor.yml`
- `plans/PR-Semantic-Diff-Advisor-CI.md`
- `scripts/semantic_diff_advisor.py`
- `tests/test_semantic_diff_advisor.py`

### Review Contract
- Acceptance: the advisor fires on replays of all four real BLOCKER diffs and
  stays quiet on a copy-only control; unit tests pass and run in the
  workflow; the advisory step cannot fail the build; ASCII-clean; stdlib-only.
- Affected surfaces: CI only. No runtime or product code.
- Risk areas: noise (mitigated: high-precision AST membership diffing for the
  main detector, test paths excluded, new-set threshold of 3 members, quiet
  controls in tests); a quiet-run-equals-correct misread (mitigated: banner
  states questions-not-verdicts and quiet is not a stamp).
- Reviewer rules triggered: R8 (deterministic static analysis, no LLM). R13
  context: the tool exists to force class-level questions, not example fixes.

## Mechanism

Detection is pure (old source, new source) -> findings, so tests need no git.
Four detectors:

1. RECOGNITION_SET_WIDENED / RECOGNITION_SET_ADDED -- parse both sources,
   collect module-level constants that are collections of string literals
   (set, tuple, list, dict keys, or set/frozenset/tuple calls wrapping one),
   and diff membership per constant name. Added members are listed in the
   finding. New sets fire only at 3+ members to cut noise.
2. MATCHER_CHANGED -- module-level compiled-regex constants whose pattern
   source changed between the two parses.
3. NORMALIZER_TERMSET_COUPLING -- a module-level function whose name matches
   token, stem, normal, fold, or signal changed (or was added) in a module
   that holds string-collection constants.
4. DEFAULTED_CONTRACT_FIELD -- line-diff (difflib opcodes) finds added lines
   carrying a defaulting idiom (nullish-coalesce to zero or empty, get with a
   zero/empty default, or-zero) within 60 lines below a projection, parse,
   validate, snapshot, or payload function context. Works on Python and JS.

The git layer resolves merge-base vs HEAD, skips test paths and non-Python,
non-JS files, and feeds worktree content as the new side. Each finding prints
with the standing adversarial question for its pattern.

## Intentional
- Advisory, not gating: the questions need judgment; a hard gate would block
  legitimate widenings. Same posture as the structural sweep (#1470).
- The unit-test step IS blocking -- the tool's own correctness gates; only
  its findings advise.
- Test paths are excluded from detection: tests legitimately add string
  fixtures and would drown the signal.
- JS/TS files get only the line-based defaulting detector (no JS parser in
  stdlib); the AST detectors are Python-only. The #1453 class was a JS bug
  and the line-based detector catches it at the exact flagged line.
- PR-trigger only (no push trigger): the tool is a diff inspector; there is
  no meaningful base on a push to main.

## Deferred
- A --strict promotion path (fail when patterns are found without a matching
  negative-fixture diff) once noise is measured in practice.
- PR-comment output (posting the questions as a sticky comment) instead of
  job-log output.
- Extending matcher-change detection to f-string regex patterns (currently
  compared by unparsed expression source).
- Parked hardening: none.

## Verification
- Unit tests: 10 passed (replays of #1446, #1439, #1466, #1453 fire with the
  expected finding names and details; unrelated-change and unchanged-file
  controls stay quiet; test-path filter pinned; new-module set firing + threshold pinned).
- Real-diff replay against the four actual BLOCKER heads:
  #1446 head fires RECOGNITION_SET_WIDENED naming first_response on the exact
  set; #1439 head fires on the auth folds; #1466 head fires
  NORMALIZER_TERMSET_COUPLING on the stemmer; #1453 head fires
  DEFAULTED_CONTRACT_FIELD at the exact line the bot flagged (L102).
  Copy-only control (#1441 head) yields zero findings.
- Advisory by construction: exit 0 without --strict; workflow adds
  continue-on-error on the advisory step; the unit-test step is blocking.
- ASCII-clean; stdlib-only (workflow installs pytest only).

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/semantic_diff_advisor.yml` | 57 |
| `plans/PR-Semantic-Diff-Advisor-CI.md` | 121 |
| `scripts/semantic_diff_advisor.py` | 381 |
| `tests/test_semantic_diff_advisor.py` | 175 |
| **Total** | **734** |
