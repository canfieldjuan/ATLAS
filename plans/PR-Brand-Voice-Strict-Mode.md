# PR-Brand-Voice-Strict-Mode

## Why this slice exists

PR #1350 made `NIT` findings advisory by default so suggested vocabulary fixes
no longer fail the marketing workflow. The same plan deferred a strict local
mode for teams that want advisory findings to block in manual checks or future
automation.

This slice adds that runtime knob without changing the default workflow
behavior.

## Scope (this PR)

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Vertical slice

1. Add a CLI `--strict` flag to `atlas_brain/brand/voice_validator.py`.
2. Keep default CLI behavior unchanged: `NIT` findings warn and exit 0.
3. Make `--strict` fail when advisory `NIT` findings are present.
4. Preserve blocking behavior for `BLOCKER` and `MAJOR` findings in both modes.
5. Add focused CLI tests for strict advisory-only and strict clean runs.
6. Update the marketing guide with the strict-mode local command.

### Review Contract

- Acceptance criteria:
  - [ ] Default `NIT`-only CLI runs still print `WARN` and exit 0.
  - [ ] `--strict` with only `NIT` findings prints the advisory finding and exits 1.
  - [ ] `--strict` on clean content still exits 0.
  - [ ] `BLOCKER`/`MAJOR` findings still fail in default and strict modes.
- Affected surfaces: brand-voice CLI, local marketing documentation, validator tests.
- Risk areas: accidentally changing workflow defaults, hiding suggestions in strict mode, confusing exit codes.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `atlas_brain/brand/voice_validator.py`
- `marketing/README.md`
- `plans/PR-Brand-Voice-Strict-Mode.md`
- `tests/test_brand_voice_validator.py`

## Mechanism

The CLI already splits findings into blocking and advisory groups. This slice
adds a `--strict` boolean and changes only the advisory-only branch:

```python
if advisory_findings and args.strict:
    print("FAIL: Found ... advisory ... in strict mode")
    exit(1)
```

Mixed blocking/advisory files already exit 1 because blocking findings are
present. Strict mode keeps printing advisory findings with suggestion lines so
the stricter exit status does not reduce repair guidance.

## Intentional

- The marketing GitHub workflow remains default/non-strict. `NIT` remains
  advisory in CI unless a future slice opts into strict behavior explicitly.
- The validator API remains unchanged; this is a CLI interpretation option.
- Strict mode is a single boolean, not a configurable severity threshold.

## Deferred

- JSON CLI/report mode for structured downstream consumers.
- Stem-aware vocabulary patterns owned in YAML.
- Larger marketing corpus expansion.
- Workflow-level strict policy if the team later wants CI to fail on `NIT`.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_brand_voice_validator.py -q` -- 64 passed.
- Four seed CLI checks for landing page, blog post, release notes, and tweet --
  PASS.
- Default advisory-only CLI smoke for `predictable` exits 0 and prints `WARN`.
- Strict advisory-only CLI smoke for `predictable` exits 1 and prints `FAIL`
  with the suggestion line.
- Strict clean CLI smoke exits 0.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-brand-voice-strict-mode-body.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/brand/voice_validator.py` | 13 |
| `marketing/README.md` | 7 |
| `plans/PR-Brand-Voice-Strict-Mode.md` | 90 |
| `tests/test_brand_voice_validator.py` | 37 |
| **Total** | **147** |
