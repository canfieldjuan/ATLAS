# PR-Resolution-Audit-Self-Correction-Round2

## Why this slice exists

A code-grounded review (Codex on #2002 + reviewer) found that the previous self-correction
PR (#2002) *itself* introduced two errors and left process gaps. This follow-up fixes them,
so the audit's self-correction survives the review it invited. All five points were verified
against the code before this change. Doc-only; product fixes stay in #1993.

The five gaps (verdict -> fix):

| # | Gap (verified) | Fix |
|---|---|---|
| 1 | H-x12 magnitude wrong: "a week ~12x too low" -- the fallback/true ratio is `12T/365`, so a week is ~4.3x too low, a year ~12x too high | correct the math in FINDINGS H-x12 |
| 2 | H-x12 over-scoped as a "customer-facing" defect: the rendered markdown (`:4076-4083`) + PDF (`deflection_pdf_renderer.py:305-312`) already hedge ("if this uploaded batch is monthly pace") | rescope to the raw `report_model`/API field |
| 3 | C1 "proven" not reproducible: proof was in gitignored `_audit_scratch/` | commit the proof artifact `docs/audits/resolution-audit-csv/investigations/c1_linkage_closure.py` and reference it |
| 4 | #2002 merged before Codex reviewed; Codex thread on FINDINGS.md:170 open | this PR addresses the thread; hold #2002-follow-up for Codex before merge; resolve the #2002 thread |
| 5 | merged plans left in `plans/` root (AGENTS.md:111-119) | archive #1994 + #2002 plans to `plans/archive/`, rebuild INDEX |

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: workflow/process

1. FINDINGS H-x12: correct magnitude (`12T/365`) + rescope to the raw field.
2. Commit `docs/audits/resolution-audit-csv/investigations/c1_linkage_closure.py` (reproducible C1 proof); INVESTIGATIONS
   C1 references it; add a Round-2 line to the self-corrections log.
3. Archive the two merged audit plans + rebuild `plans/INDEX.md` (closes the AGENTS.md gap
   the reviewer flagged for this arc's plans -- in-scope hygiene, not a drive-by).

### Files touched

- `docs/audits/resolution-audit-csv/FINDINGS.md`
- `docs/audits/resolution-audit-csv/INVESTIGATIONS.md`
- `docs/audits/resolution-audit-csv/investigations/c1_linkage_closure.py`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-Self-Correction-Round2.md`
- `plans/archive/PR-Resolution-Audit-Investigations.md`
- `plans/archive/PR-Resolution-Audit-Self-Correction.md`

### Review Contract

- Acceptance criteria:
  - H-x12 states the `12T/365` ratio (year ~12x high, week ~4.3x low) and scopes the defect
    to the raw field, citing the hedged renderers `:4076-4083` / `:305-312`.
  - `docs/audits/resolution-audit-csv/investigations/c1_linkage_closure.py` is committed, ASCII, and re-runs to
    `merge=0.295 / connect=0.535 / no pure-2 band` (model revision pinned + expected output recorded so it stays reproducible); INVESTIGATIONS C1 links it.
  - The two merged plans are under `plans/archive/`; `plans/INDEX.md` rebuilt.
  - No product code modified.
- Affected surfaces: documentation only.
- Rule mapping: R14 (verify against the codebase) universal and satisfied -- the `12T/365`
  math, the renderer hedge, and the MST closure were re-run/verified.

## Mechanism

`12T/365` re-derived (T=7 -> 0.230x; T=365 -> 12.0x); renderer hedge confirmed at
`faq_deflection_report.py:4076-4083` and `deflection_pdf_renderer.py:305-312`; the C1
artifact re-run reproduces `0.295 / 0.535 / no pure-2 band`.

## Intentional
- Doc + committed-artifact corrections only; no product code.
- Archiving bundled because it closes the reviewer-flagged AGENTS.md gap for *this arc's* plans.

## Deferred
- The underlying product fixes stay tracked in #1993 / #2000.

## Parked hardening
- None. Doc-only.

## Verification
- `docs/audits/resolution-audit-csv/investigations/c1_linkage_closure.py` ->
  `merge_distance=0.2945`, `connect_distance=0.5352`, no pure-2 band.
- `12 x 7 / 365 = 0.230` (4.3x low); `12 x 365 / 365 = 12.0` (12x high). ASCII-only; scratch out of the diff.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/FINDINGS.md` | 24 |
| `docs/audits/resolution-audit-csv/INVESTIGATIONS.md` | 16 |
| `docs/audits/resolution-audit-csv/investigations/c1_linkage_closure.py` | 84 |
| `plans/INDEX.md` | 4 |
| `plans/PR-Resolution-Audit-Self-Correction-Round2.md` | 86 |
| `plans/archive/PR-Resolution-Audit-Investigations.md` | 0 |
| `plans/archive/PR-Resolution-Audit-Self-Correction.md` | 0 |
| **Total** | **214** |
