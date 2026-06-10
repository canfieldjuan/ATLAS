# PR-Brand-Voice-Severity-Gate

## Why this slice exists

PR #1348 made `vocabulary.use` actionable by emitting `NIT` findings with
replacement suggestions, but the CLI still exits 1 on every finding. That means
the new suggestion path is technically advisory in metadata while still blocking
the marketing content workflow.

This slice closes the first deferred item from
`plans/PR-Brand-Voice-Suggested-Fixes.md`: severity-aware workflow behavior that
can warn on `NIT` findings without failing CI.

## Scope (this PR)

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Vertical slice

1. Keep validator findings unchanged: `BLOCKER`, `MAJOR`, and `NIT` are still
   emitted by `BrandVoiceValidator.validate()`.
2. Change the CLI gate so only `BLOCKER` and `MAJOR` findings fail by default.
3. Print `NIT` findings as advisory warnings, including existing suggestion
   lines, while returning exit 0 when no blocking finding exists.
4. Add focused CLI tests for advisory-only, blocking-only, and mixed severity
   behavior.
5. Update the marketing guide so marketers know `NIT` is advisory and
   `MAJOR`/`BLOCKER` still block.

### Review Contract

- Acceptance criteria:
  - [ ] A file with only `NIT` findings prints warnings and exits 0.
  - [ ] A file with `MAJOR` or `BLOCKER` findings still prints failures and exits 1.
  - [ ] Mixed blocking plus advisory findings exit 1 while still printing both groups.
  - [ ] Suggestion text remains visible for advisory `NIT` findings.
- Affected surfaces: brand-voice CLI, marketing content workflow exit behavior, marketing guide.
- Risk areas: accidentally weakening blocking brand checks, hiding advisory suggestions, confusing CLI output.
- Reviewer rules triggered: R1, R2, R10, R12.

### Files touched

- `atlas_brain/brand/voice_validator.py`
- `marketing/README.md`
- `plans/PR-Brand-Voice-Severity-Gate.md`
- `tests/test_brand_voice_validator.py`

## Mechanism

The validator already returns structured findings with severities. This slice
keeps that API stable and changes only CLI interpretation:

```python
blocking = [f for f in findings if f.severity in {"BLOCKER", "MAJOR"}]
advisory = [f for f in findings if f.severity not in {"BLOCKER", "MAJOR"}]
```

The CLI prints blocking findings under a `FAIL` heading and advisory findings
under a `WARN` heading. Exit status is determined only by whether the blocking
list is non-empty. The existing workflow continues to call the CLI unchanged,
so `NIT` findings become workflow warnings without a workflow-specific branch.

## Intentional

- No workflow YAML change is needed; the workflow already delegates pass/fail to
  the CLI exit code.
- The validator API stays unchanged so any future JSON/reporting slice can reuse
  the same structured findings.
- The default blocking severities are hard-coded to `BLOCKER` and `MAJOR` for
  this slice; per-run severity configuration remains out of scope.

## Deferred

- Runtime configuration for strict local runs that fail on `NIT`.
- JSON CLI/report mode for structured downstream consumers.
- Stem-aware vocabulary patterns owned in YAML.
- Larger marketing corpus expansion.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_brand_voice_validator.py -q` -- 61 passed.
- Four seed CLI checks for landing page, blog post, release notes, and tweet --
  PASS.
- Advisory-only CLI smoke for `predictable` exits 0 and prints `WARN` plus the
  suggestion line.
- Blocking CLI smoke for `game-changer!!` exits 1 and prints `FAIL`.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-brand-voice-severity-gate-body.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/brand/voice_validator.py` | 41 |
| `marketing/README.md` | 4 |
| `plans/PR-Brand-Voice-Severity-Gate.md` | 98 |
| `tests/test_brand_voice_validator.py` | 20 |
| **Total** | **163** |
