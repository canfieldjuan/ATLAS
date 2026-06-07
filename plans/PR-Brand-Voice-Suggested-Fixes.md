# PR-Brand-Voice-Suggested-Fixes

## Why this slice exists

The brand-voice YAML has carried a `vocabulary.use` block since PR #1344, but it
was intentionally marked FUTURE because flat string violations had nowhere clean
to carry replacement guidance. PR #1346 added structured findings, so the next
smallest lane slice is to make those existing preferred/discouraged vocabulary
pairs actionable.

This gives marketers a concrete repair hint instead of only a failure message,
while keeping the CI gate deterministic and fail-closed.

## Scope (this PR)

Ownership lane: content-marketing/brand-voice-checks
Slice phase: Vertical slice

1. Add an optional `suggestion` field to `BrandVoiceFinding`.
2. Implement `vocabulary.use` as preferred-to-discouraged mappings: when a
   discouraged word appears, emit a finding that suggests the preferred term.
3. Validate `vocabulary.use` shape at config load so malformed marketer-edited
   YAML fails loudly instead of no-oping.
4. Update CLI output to print the suggestion when a finding carries one.
5. Add focused tests for suggestion fields, CLI output, malformed mapping
   rejection, and clean preferred-term near-misses.
6. Update the seeded marketing copy that used newly discouraged terms so the
   existing corpus remains green once suggestions are enforced.

### Review Contract

- Acceptance criteria:
  - [ ] Discouraged `vocabulary.use` terms emit `NIT` findings with replacement suggestions.
  - [ ] The CLI prints suggestion text when a finding carries one.
  - [ ] Malformed `vocabulary.use` YAML fails at config load, including falsey non-list values.
  - [ ] Existing whole-word vocabulary precision and seeded marketing checks stay green.
- Affected surfaces: brand-voice validator, marketer-owned YAML config, text CLI output, seed marketing copy.
- Risk areas: false positives, malformed config drift, CI gating behavior.
- Reviewer rules triggered: R1, R2, R10, R11, R12.

### Files touched

- `atlas_brain/brand/voice_validator.py`
- `atlas_brain/skills/brand/brand_voice.yml`
- `marketing/blog_posts/why-deterministic-checks.md`
- `marketing/landing_pages/atlas-platform.md`
- `plans/PR-Brand-Voice-Suggested-Fixes.md`
- `tests/test_brand_voice_validator.py`

## Mechanism

The YAML shape is already present:

```yaml
vocabulary:
  use:
    - "deterministic": "predictable"
```

This slice treats each item as `preferred: discouraged`. The validator checks
for whole-word matches of the discouraged term and emits a structured finding:

- `rule_id`: `vocabulary.use.<discouraged>`
- `severity`: `NIT`
- `category`: `vocabulary`
- `message`: `Prefer '<preferred>' over '<discouraged>'`
- `suggestion`: `Use '<preferred>' instead of '<discouraged>'`

The CLI remains text and still exits 1 on any finding, but prints the suggestion
under the finding line.

## Intentional

- Suggested fixes are advisory metadata, but they still fail the current CLI
  gate because the workflow remains fail-on-any-finding. Separating advisory
  severities from blocking workflow behavior is a later policy slice.
- The existing `vocabulary.avoid` hard-block list remains unchanged.
- This does not add stem-aware matching; all vocabulary matching stays
  whole-word to preserve the precision fix from PR #1344.
- The YAML orientation is now explicit: `use` means preferred term first,
  discouraged term second.
- Seed copy edits are limited to the terms this slice now enforces:
  `platform` -> `system` and `predictable` -> `deterministic`.

## Deferred

- Severity-aware workflow behavior that can warn on `NIT` without failing CI.
- Stem-aware vocabulary patterns owned in YAML.
- JSON CLI/report mode for structured downstream consumers.
- A larger content corpus beyond the four seed examples.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_brand_voice_validator.py -q` -- 60 passed.
- `python atlas_brain/brand/voice_validator.py --file marketing/landing_pages/atlas-platform.md --type landing_page` -- PASS.
- `python atlas_brain/brand/voice_validator.py --file marketing/blog_posts/why-deterministic-checks.md --type blog_post` -- PASS.
- `python atlas_brain/brand/voice_validator.py --file marketing/release_notes/2026-06-release.md --type release_notes` -- PASS.
- `python atlas_brain/brand/voice_validator.py --file marketing/tweets/launch-brand-voice-checks.md --type tweet` -- PASS.
- Negative CLI smoke with `The result is predictable for operators.` exits 1 and
  prints `[NIT] vocabulary.use.predictable` plus the suggestion line.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-brand-voice-suggested-fixes-body.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/brand/voice_validator.py` | ~62 |
| `atlas_brain/skills/brand/brand_voice.yml` | ~6 |
| `marketing/blog_posts/why-deterministic-checks.md` | ~2 |
| `marketing/landing_pages/atlas-platform.md` | ~6 |
| `plans/PR-Brand-Voice-Suggested-Fixes.md` | ~117 |
| `tests/test_brand_voice_validator.py` | ~113 |
| **Total** | **~306** |
