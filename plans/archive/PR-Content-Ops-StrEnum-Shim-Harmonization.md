# PR-Content-Ops-StrEnum-Shim-Harmonization

## Why this slice exists

A `HARDENING.md` entry (content-ops/review-contract lane, parked at the #1487
slice-5 review) flagged that `review_contract.py` and `calibration_library.py`
still use the bare `class StrEnum(str, Enum): pass` 3.10 fallback shim, while
`adversarial_pass.py` already sets `__str__ = str.__str__`. The bare form is
inert today, but on the 3.10 fallback path a member formatted via `str()`/an
f-string would emit the class-qualified name (`CalibrationLabel.APPROVED`)
instead of the value (`approved`) -- a latent trap if either module later
interpolates a label. This slice harmonizes all three shims and drains the
HARDENING entry.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Set `__str__ = str.__str__` on the 3.10 fallback `StrEnum` shim in
   `review_contract.py` and `calibration_library.py`, matching
   `adversarial_pass.py`.
2. Remove the now-resolved HARDENING.md entry.

### Files touched

- `HARDENING.md`
- `extracted_content_pipeline/calibration_library.py`
- `extracted_content_pipeline/review_contract.py`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-StrEnum-Shim-Harmonization.md`
- `plans/archive/PR-Content-Ops-Calibration-Admin-Surface.md`

### Review Contract

Acceptance criteria:
- Both fallback shims set `__str__ = str.__str__`, identical to
  `adversarial_pass.py`; all three now match.
- No behavior change on the CI interpreter (3.11 uses the real `StrEnum`); the
  existing package suites still pass.
- The HARDENING.md entry is removed.

Affected surfaces: two `# pragma: no cover` 3.10-only fallback blocks. No runtime
behavior change on 3.11; no API, schema, or logic change.

Risk areas: none beyond the inert shim itself.

Reviewer rules triggered: R1, R10, R14.

## Mechanism

The fallback `StrEnum` (only defined when `from enum import StrEnum` raises on
3.10) now overrides `__str__` with `str.__str__`, so `str(member)` returns the
value rather than `Enum.__str__`'s `ClassName.MEMBER`. This is exactly the form
`adversarial_pass.py` already carries (added and reviewed in #1487), so all three
modules' shims are now consistent.

## Intentional

- **No new test.** The change lives in a `# pragma: no cover` block that only
  executes on Python 3.10 (CI runs 3.11, where the real `StrEnum` is imported and
  this shim is never defined). It is a no-op on the test interpreter, and it
  mirrors the already-reviewed `adversarial_pass.py` shim verbatim; the existing
  package suites confirm no regression. The original parked entry was rated
  effort S / inert.
- **Harmonization, not new behavior.** Both modules are inert today (neither
  formats an enum member into output); this removes the class-wide trap pre-emptively.

## Deferred

- Parked hardening: none new. (This slice drains the only content-ops/review-contract
  entry in HARDENING.md.)

## Verification

- Reviewer rules triggered: R1, R10, R14.
- Passed: pytest of the five review-contract package suites (review_contract,
  calibration_library, adversarial_pass, content_pr, claims_map) -- 101 passed.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 11 |
| `extracted_content_pipeline/calibration_library.py` | 5 |
| `extracted_content_pipeline/review_contract.py` | 5 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-StrEnum-Shim-Harmonization.md` | 82 |
| `plans/archive/PR-Content-Ops-Calibration-Admin-Surface.md` | 0 |
| **Total** | **106** |
