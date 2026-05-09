# PR-Campaign-Config-V2: drop the legacy `channel` field on `CampaignGenerationConfig`

## Why this slice exists

The deferred follow-up to PR-Campaign-Channel-Legacy-Cleanup. That
previous slice deprecated the dual-field surface in
`CampaignGenerationConfig`:

```python
channel: str = "email"               # legacy (single-channel)
channels: tuple[str, ...] = ()       # newer (multi-channel)
```

The fallback in `_channels()` was:

```python
raw_value = self._config.channels or (self._config.channel,)
```

That cleanup migrated the in-tree test off the legacy field and
documented the removal path; it explicitly left the field + fallback
in place because **removing them is the breaking change**. This PR
is that breaking change.

The deferred regression test at
`tests/test_extracted_campaign_generation.py:1235-1260`
(`test_generate_legacy_channel_field_still_resolves_when_channels_unset`)
locks the legacy contract until this PR lands. Its docstring already
says:

> After PR-Campaign-Config-V2 removes the field this test goes away
> with it.

This PR removes it together with the field.

## Scope (this PR)

Tightly scoped. **Two structural changes, both in
`campaign_generation.py`, plus the locked-in regression-test removal.**

### Files touched

1. `extracted_content_pipeline/campaign_generation.py`
   - Remove `channel: str = "email"` from `CampaignGenerationConfig`
     (the dataclass field at line 147).
   - Remove the legacy-field comment block at lines 141-146.
   - Drop `or (self._config.channel,)` from `_channels()` line 422.
     The remaining `normalized or ("email",)` final fallback at line
     428 keeps `_channels()` returning at least `("email",)` when
     the host construction supplies no channels at all -- preserving
     the "default to email" behavior without the dead config field.
2. `tests/test_extracted_campaign_generation.py`
   - Remove
     `test_generate_legacy_channel_field_still_resolves_when_channels_unset`
     (lines 1235-1260). It exercises the now-removed code path.

That's it. No other call sites construct
`CampaignGenerationConfig(channel=...)` -- verified via
`grep -rn "CampaignGenerationConfig(channel="` returning only the
deferred test.

## Mechanism

Frozen dataclasses error-fast on unknown kwargs. Hosts that still
construct `CampaignGenerationConfig(channel="...")` will hit a
`TypeError: __init__() got an unexpected keyword argument 'channel'`
at construction time -- not a silent regression. They migrate by
switching to `channels=("email",)`.

Behaviorally, the default channel ("email") is preserved by
`_channels()`'s existing terminal fallback at line 428
(`normalized or ("email",)`). A host that previously relied on the
implicit `channel="email"` default still gets `("email",)` from
`_channels()` when no `channels=` is supplied at construction.

## Intentional

- **Hard removal, no deprecation period.** The previous slice
  documented the removal and gave hosts a path to migrate. A
  semver-major version bump signals the break; a runtime
  `warnings.warn` adds noise without value when the field is genuinely
  gone in the next release.
- **Default behavior preserved.** Hosts that constructed
  `CampaignGenerationConfig()` (no channel kwargs at all) still
  produce drafts on the `email` channel via the terminal fallback in
  `_channels()`. Only construction with explicit `channel="..."` breaks.
- **Locked-in regression test removed in the same PR.** The test was
  put in place specifically to fail when the field is removed; its
  removal is the signal that the contract has been broken
  intentionally.

## Deferred

- `warnings.warn` shim at construction time. Skipped per above.
- Renaming `channels` to `channel` (collapsing back to the singular
  name once multi-channel is the only surface). Possible future
  ergonomics slice; out of scope.
- Cross-package coordination (downstream Atlas hosts that may
  construct `CampaignGenerationConfig(channel=...)`): handled by the
  package version bump and changelog, not this PR's code.
- Schema-level cleanup (the `CampaignDraft.channel` field, the
  `campaigns.channel` postgres column) -- those are **per-draft
  routing**, not config defaults. Keep both. Different semantics,
  different lifecycle, not part of this slice.

## Verification

- `pytest tests/test_extracted_campaign_generation.py` -- the locked
  regression test is deleted; the remaining tests stay green.
- `pytest tests/test_extracted_campaign_postgres_generation.py
   tests/test_extracted_campaign_postgres.py
   tests/test_extracted_content_ops_execution.py
   tests/test_extracted_blog_generation.py` -- no regressions
  (these construct `CampaignDraft(channel="email")` per-draft, not
  `CampaignGenerationConfig(channel=...)`).
- `bash scripts/validate_extracted_content_pipeline.sh` -- clean.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
   extracted_content_pipeline` -- clean.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` --
  Atlas runtime import findings: 0.
- `bash scripts/check_ascii_python.sh` -- clean.
- `grep -rn "CampaignGenerationConfig(channel=" extracted_content_pipeline tests` --
  expected: empty.
- `grep -rn "self._config.channel\b" extracted_content_pipeline` --
  expected: empty.

## Estimated diff size

- `campaign_generation.py`: -8 / +2 (field removal + fallback drop +
  comment cleanup).
- `test_extracted_campaign_generation.py`: -27 / 0 (delete the
  locked-in test).
- Plan doc: ~110 LOC (this file).

Total: ~150 LOC. Well under the 400 LOC PR target. Most of the diff
is the plan doc; the actual code change is ~10 lines net.
