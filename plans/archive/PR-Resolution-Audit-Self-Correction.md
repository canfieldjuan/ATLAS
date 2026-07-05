# PR-Resolution-Audit-Self-Correction

## Why this slice exists

An independent adversarial review of the Resolution Audit CSV math + findings by
GPT-5.5 Pro (portal) confirmed the money math and caught three real over-claims in our
own merged audit docs, plus flagged an annualization-basis issue. This folds the fixes
back in-place with a visible self-corrections log, and logs a new product finding. An
audit that survives external adversarial review with named fixes is the credibility we
want behind the "block paid reports" call. Doc-only; product fixes stay in #1993.
Tracking: #2000.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: workflow/process

1. **C1** -- reword the single-linkage claim; the negative separation gap proves only
   *pairwise* inseparability, closed empirically with the MST connectivity bottleneck
   (`connect 0.535 > merge 0.295` -> no pure-2 threshold). Conclusion stands, now proven.
2. **A4** -- scope "cents reconciles by construction" to base `n x $13.50`; note
   fractional-cent annualized rows still need the raw-total guard.
3. **B2** -- guard must call the same billed-repeat predicate, not a re-hardcoded `tc>=2`.
4. **D1/D2** -- label `$810/$1,944` as the dateless `x12` run-rate (`faq_deflection_report.py:3232`),
   not `x365/window`; annualized dollars framed as run-rates, not confident annual claims.
5. **New finding H-x12** -- the dateless `x12` annualization is an unfounded "one month" assumption.
6. Visible `## Self-corrections` log appended to `INVESTIGATIONS.md`.

### Files touched

- `docs/audits/resolution-audit-csv/INVESTIGATIONS.md`
- `docs/audits/resolution-audit-csv/FINDINGS.md`
- `docs/audits/resolution-audit-csv/SUMMARY.md`
- `docs/audits/resolution-audit-csv/PRESENTATION.md`
- `plans/PR-Resolution-Audit-Self-Correction.md`

### Review Contract

- Acceptance criteria:
  - C1 wording matches the reviewer-run MST closure (`merge=0.295`, `connect=0.535`, no pure-2 band).
  - A4 scoped to `n x $13.50`; B2 says "same predicate"; D1/D2 label the `x12` run-rate basis.
  - New H-x12 finding present with the `:3232` anchor; self-corrections log present.
  - No product code modified; `_audit_scratch/` gitignored and out of the diff.
- Affected surfaces: documentation only; no runtime/API/auth/UI surface.
- Rule mapping: R14 (verify against the codebase) universal and satisfied -- the MST closure
  and the `x12` vs `x365/window` basis were re-run/verified against `faq_deflection_report.py:3226-3234`.

## Mechanism

The C1 MST closure was computed in scratch (`_audit_scratch/c1_linkage_closure.py`,
`all-MiniLM-L6-v2` + `scipy`); the `x12` vs `x365/window` ground truth is confirmed at
`faq_deflection_report.py:3226-3234`. Edits are in-place; no numbers were changed except
to attach the correct basis/qualifier.

## Intentional

- Doc corrections + one new finding; no product code.
- C1 conclusion is preserved (single-linkage cannot separate the fixture) but re-justified
  with the connectivity bottleneck instead of the insufficient separation-gap argument.

## Deferred

- Implementing the underlying product fixes (clustering, money helper, guard, annualization span
  inference) stays tracked in #1993 / #2000.

## Parked hardening

- None. Doc-only.

## Verification

- `_audit_scratch/c1_linkage_closure.py` re-run: `merge_distance=0.295`, `connect_distance=0.535`,
  no pure-2 band -> single-linkage impossibility for the fixture (matches the C1 wording).
- `x810 = 5 x 12 x 13.50`, `x365/23` gives `$1,071/$2,571` -- confirmed; basis labeled.
- ASCII-only on all edited markdown docs; `git diff --check` clean; `_audit_scratch/` out of the diff.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/INVESTIGATIONS.md` | 41 |
| `docs/audits/resolution-audit-csv/FINDINGS.md` | 20 |
| `docs/audits/resolution-audit-csv/SUMMARY.md` | 5 |
| `docs/audits/resolution-audit-csv/PRESENTATION.md` | 2 |
| `plans/PR-Resolution-Audit-Self-Correction.md` | 86 |
| **Total** | **154** |
