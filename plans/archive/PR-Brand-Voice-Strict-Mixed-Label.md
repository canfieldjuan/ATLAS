# PR-Brand-Voice-Strict-Mixed-Label

## Why this slice exists

The review on PR #1352 approved strict mode and flagged one skip-worthy NIT:
when a file has both blocking findings and advisory `NIT` findings under
`--strict`, the CLI exits 1 correctly but still labels the advisory group as
`WARN`. That can confuse a strict run because `NIT` findings are also strict
failures in that mode.

This slice closes that review NIT as a tiny product-polish follow-up.

## Scope (this PR)

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Product polish

1. Reuse one advisory-heading helper for both advisory-only and mixed
   blocking/advisory output.
2. Keep default mixed output unchanged: blocking findings fail, advisory
   findings print under `WARN`.
3. In strict mixed output, label the advisory group as `FAIL` with the same
   `in strict mode` annotation used by strict advisory-only output.
4. Add focused CLI tests for strict mixed output and default mixed output.

### Review Contract

- Acceptance criteria:
  - [ ] Mixed blocking/advisory output without `--strict` still prints the advisory group under `WARN` and exits 1.
  - [ ] Mixed blocking/advisory output with `--strict` prints the advisory group under `FAIL ... in strict mode` and exits 1.
  - [ ] Strict advisory-only output keeps the same `FAIL ... in strict mode` label and exit 1.
  - [ ] Suggestion text remains visible in all advisory groups.
- Affected surfaces: brand-voice CLI output and validator CLI tests.
- Risk areas: misleading CLI labels, changed default workflow output, hidden suggestions.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `atlas_brain/brand/voice_validator.py`
- `plans/PR-Brand-Voice-Strict-Mixed-Label.md`
- `tests/test_brand_voice_validator.py`

## Mechanism

Move the advisory heading construction into a tiny helper:

```python
def _print_advisory_findings(findings, path, *, strict):
    status = "FAIL" if strict else "WARN"
    strict_note = " in strict mode" if strict else ""
    ...
```

The blocking branch calls this helper when advisory findings are present, and
the advisory-only branch calls the same helper before choosing the strict/non-
strict exit code. That keeps the label behavior consistent across both output
shapes.

## Intentional

- This does not change any exit code; PR #1352 already made those correct.
- This does not change workflow defaults; the workflow still runs without
  `--strict`.
- No docs change is needed because the strict-mode README command remains
  accurate.

## Deferred

- JSON CLI/report mode for structured downstream consumers.
- Stem-aware vocabulary patterns owned in YAML.
- Larger marketing corpus expansion.
- Workflow-level strict policy if the team later wants CI to fail on `NIT`.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_brand_voice_validator.py -q` -- 65 passed.
- Default mixed CLI smoke exits 1 and prints `WARN` for advisory findings.
- Strict mixed CLI smoke exits 1 and prints `FAIL ... in strict mode` for
  advisory findings.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-brand-voice-strict-mixed-label-body.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/brand/voice_validator.py` | 26 |
| `plans/PR-Brand-Voice-Strict-Mixed-Label.md` | 90 |
| `tests/test_brand_voice_validator.py` | 16 |
| **Total** | **132** |
