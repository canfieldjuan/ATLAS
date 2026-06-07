# PR-Brand-Voice-Structured-Findings

## Why this slice exists

PR #1344 landed the deterministic brand-voice gate, but intentionally left
validator results as flat strings. That was enough for a binary CI gate, but it
blocks the next product steps named in the merged plan: severity levels,
suggested fixes, rule-specific reporting, and cleaner marketer-facing output.

This slice adds the smallest structured result model that keeps the current CI
behavior intact while giving future slices stable fields to consume.

The estimate is slightly over the 400 LOC soft cap because the return-type
change touches the existing failure-detection suite broadly: every old
message-based assertion must remain covered while the new structured fields get
their own focused negative/config tests.

## Scope (this PR)

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Vertical slice

1. Add a lightweight `BrandVoiceFinding` value object with `rule_id`,
   `severity`, `category`, and `message` fields.
2. Change `BrandVoiceValidator.validate(...)` to return findings instead of raw
   strings while preserving binary pass/fail CLI behavior.
3. Require stable rule IDs for configured tone/content rules and validate
   optional rule severities against `BLOCKER`, `MAJOR`, and `NIT`.
4. Update the existing validator tests to assert messages through the structured
   findings and add focused tests for rule IDs, default severities, and invalid
   severity detection.
5. Update the CLI output to include severity and rule ID for each finding.

### Files touched

- `atlas_brain/brand/voice_validator.py`
- `atlas_brain/brand/__init__.py`
- `plans/PR-Brand-Voice-Structured-Findings.md`
- `tests/test_brand_voice_validator.py`

## Mechanism

`BrandVoiceFinding` is a frozen dataclass. The validator emits:

- vocabulary findings with rule IDs shaped as `vocabulary.avoid.<word>` and
  default severity `BLOCKER`;
- tone-rule findings using the configured rule `id` and default severity
  `MAJOR`;
- content-rule findings using the configured rule `id` and default severity
  `BLOCKER`.

The CLI still exits 1 when any finding exists and 0 when none exist, but each
failure line includes `[SEVERITY] rule_id: message`. Tests keep the previous
branch coverage and add structural assertions so this is not only cosmetic.

## Intentional

- This slice does not implement `vocabulary.use` suggested fixes. The structured
  finding model is the substrate for that follow-up.
- The CLI remains text output, not JSON. A JSON/reporting format can be its own
  consumer-facing slice once fields settle.
- `validate(...)` changes return type in this narrow brand package because PR
  #1344 introduced the package and no in-tree caller consumes the return value
  except tests and CLI. There is no compatibility shim for raw strings.
- Severity is metadata only in this slice. The CI gate remains fail-on-any
  finding so the existing workflow stays conservative.
- Local cross-layer caller hints for `validate` and `main` are generic-name false
  positives. They point at unrelated semantic-cache methods and script
  entrypoints, not imports of `atlas_brain.brand.voice_validator`.

## Deferred

- Suggested-fix output for `vocabulary.use`.
- Stem-aware vocabulary patterns owned in YAML.
- JSON CLI/report mode for structured downstream consumers.
- A larger content corpus beyond the four seed examples.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_brand_voice_validator.py -q` -- 52 passed.
- `python atlas_brain/brand/voice_validator.py --file marketing/landing_pages/atlas-platform.md --type landing_page` -- PASS.
- Negative CLI smoke with `This is a game-changer!!` exits 1 and prints the
  expected `[BLOCKER]`/`[MAJOR]` structured rule lines.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-brand-voice-structured-findings-body.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/brand/__init__.py` | ~2 |
| `atlas_brain/brand/voice_validator.py` | ~130 |
| `plans/PR-Brand-Voice-Structured-Findings.md` | ~96 |
| `tests/test_brand_voice_validator.py` | ~240 |
| **Total** | **~472** |
