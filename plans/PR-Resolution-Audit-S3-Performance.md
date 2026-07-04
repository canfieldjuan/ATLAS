# PR-Resolution-Audit-S3-Performance

## Why this slice exists

Slice 3 of the Resolution Audit CSV audit arc (epic #1956, slice issue #1959) --
Phase 3 performance. Profiles the real pipeline on synthetic files to quantify the
O(n^2) risk, memory multiplier, and redundant recomputation that Phase 0 flagged and
Slice 2 partially measured. Appends a Performance section to the existing FINDINGS.md.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: workflow/process

1. Profile the full chain (input package -> report generate) at 2k/5k/10k tickets;
   fit the growth curve; cProfile the hotspots.
2. Measure memory (peak RSS / file->RAM multiplier) and redundant recomputation.
3. Append findings (P1-P5) to `docs/audits/resolution-audit-csv/FINDINGS.md`.
4. No product code changes; profiling scripts stay in gitignored `_audit_scratch/perf/`.

### Files touched

- `docs/audits/resolution-audit-csv/FINDINGS.md`
- `plans/PR-Resolution-Audit-S3-Performance.md`

### Review Contract

- Acceptance criteria:
  - The Performance section records each finding with severity and `file:line`, and
    measured numbers (time/memory at each size + growth exponent). The reviewer
    independently re-ran the scaling curve (2k/5k/10k) against `origin/main`.
  - No product code is modified; `_audit_scratch/` stays gitignored and out of the diff.
- Affected surfaces: documentation only; no runtime/API/auth/UI surface, no reachability proof applies.
- Rule mapping: R14 (verify against the codebase) is universal and satisfied -- the
  reviewer re-ran the scaling curve. No other rule classes are triggered by this doc-only diff.

## Mechanism

Synthetic fixtures (realistic ticket text) built under `_audit_scratch/perf/` and run
through the real functions; cProfile + `resource.getrusage` peak RSS. Numbers read from
runs, not hand-computed. Reviewer reproduced the 2k/5k/10k scaling.

## Intentional

- Findings only; refactor recommendations are Slice 4 (#1960).
- Corrects the Phase-0 O(n^2) framing: the token-set clustering IS O(n^2)-ish but is
  gated off above 2000 rows, so the real cost is a ~linear-with-large-constant path.

## Deferred

- `REFACTORS.md`, `PRESENTATION.md`, one-page `SUMMARY.md` -- Slice 4 (#1960).

## Verification

- Measured on real runs against `origin/main`; reviewer re-ran the scaling sweep
  (2k=4.2s, 5k=9.2s, 10k=18.6s; ~linear ~1.86ms/ticket) and confirmed the token-set
  skip above 2000 rows (build_package flat 1k-8k on short text).
- `git diff --check` clean; ASCII-only markdown.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/FINDINGS.md` | 70 |
| `plans/PR-Resolution-Audit-S3-Performance.md` | 66 |
| **Total** | **136** |
