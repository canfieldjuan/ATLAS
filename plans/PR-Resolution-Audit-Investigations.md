# PR-Resolution-Audit-Investigations

## Why this slice exists

Follow-up to the Resolution Audit CSV audit arc (#1956) and remediation issue #1993:
Juan asked for a thorough per-finding investigation (not just the highest-value fix).
This ships `INVESTIGATIONS.md` -- for each of the ~25 findings, the root cause traced
to the true upstream origin, the exact fix (code sketch), boundary-probed edge cases,
blast radius, tests-first, and effort -- plus shared-root maps that collapse the
findings into ~7 fixes, and corrected/verified designs for the load-bearing ones
(R1 clustering, R2 cache, P5-2 money). Over the LOC soft-target because it is one
indivisible reference doc closing the remediation analysis for a 25-finding audit.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: workflow/process

1. `docs/audits/resolution-audit-csv/INVESTIGATIONS.md` -- the per-finding deep-dive.
2. No product code changes; prototypes stay in gitignored `_audit_scratch/`.

### Files touched

- `docs/audits/resolution-audit-csv/INVESTIGATIONS.md`
- `plans/PR-Resolution-Audit-Investigations.md`

### Review Contract

- Acceptance criteria:
  - Every finding (C1-C3, F1/F3-F9/F11, H4/H5, M6-M9, P1/P2/P4/P5/P7, P5-1..P5-7,
    richness) has a root cause with `file:line`, an exact fix, and effort.
  - Load-bearing claims marked `[verified-by-reviewer]` were independently re-run:
    R2 speedup+parity (6-55x, byte-identical), P5-2 cents reconciliation, the guard
    second-side trap (repeat-basis), the R1 separation-gap (negative -> band overlap),
    and the P7 byte-alias cap.
  - No product code modified; `_audit_scratch/` gitignored and out of the diff.
- Affected surfaces: documentation only; no runtime/API/auth/UI surface.
- Rule mapping: R14 (verify against the codebase) is universal and satisfied. No other
  rule classes triggered (doc-only).

## Mechanism

Four area investigations (parsing, clustering, performance, presentation+richness) run
in parallel, each tracing findings to origin and prototyping the key fixes in scratch.
The reviewer re-ran the load-bearing prototypes (R2 parity/speedup, P5-2 reconciliation,
guard trap, R1 band-overlap, P7 byte-alias) before compiling.

## Intentional

- Recommendations + verified designs only; no product code touched.
- INVESTIGATIONS.md corrects the audit's R1 (a naive semantic pass reproduces F4 --
  the cosine bands overlap; complete/average linkage + category scope + calibration required).

## Deferred

- Implementing any fix is a separate non-audit lane (tracked in #1993).

## Parked hardening

- None. Doc-only, no code/test changes.

## Verification

- Reviewer re-ran: R2 (6.1x one-liner / 55x key-index, byte-identical parity), P5-2 cents
  (column==headline exactly), guard second-side trap (naive sum false-positives on tc==1),
  R1 separation-gap (-0.470, bands overlap), P7 (`_MAX_DEFLECTION_SUBMIT_ROWS = bytes`).
- `git diff --check` clean; ASCII-only; `_audit_scratch/` gitignored and out of the diff.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/INVESTIGATIONS.md` | 226 |
| `plans/PR-Resolution-Audit-Investigations.md` | 75 |
| **Total** | **301** |
