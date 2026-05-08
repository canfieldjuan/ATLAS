# PR: deprecate the legacy `channel` field on CampaignGenerationConfig

## Why this slice exists

``CampaignGenerationConfig`` carries two channel-shape fields:

```python
channel: str = "email"               # legacy (single-channel)
channels: tuple[str, ...] = ()       # newer (multi-channel)
```

The ``_channels()`` resolver does ``self._config.channels or (self._config.channel,)``
-- if ``channels`` is empty, it falls back to a one-tuple from
``channel``. Both fields work; the dual surface has been a noted
deferred-cleanup item across multiple PRs (PR-OptionA-1 through
PR-Consistency-3 all listed it under "Deferred").

The Content Ops post-merge audit flagged the duplication. This PR
does the smallest cleanup that closes the audit item without
breaking existing hosts:

1. Mark ``channel`` as deprecated in the dataclass docstring.
2. Migrate the one in-tree test that constructs
   ``CampaignGenerationConfig(channel=...)`` to use ``channels=``
   instead, so the package's own tests don't exercise the legacy
   path.
3. Document the removal path so the breaking change can land in a
   future versioned slice.

## Scope (this PR)

Documentation + test migration only. **No field removal.** The
fallback chain in ``_channels()`` stays so existing hosts that
construct ``CampaignGenerationConfig(channel="...")`` still work --
they just hit a deprecated path documented in the dataclass.

| Change | File | Lines |
|---|---|---|
| Field-level deprecation note | ``campaign_generation.py`` | 1 |
| Test migration (``channel="linkedin"`` -> ``channels=("linkedin",)``) | ``tests/test_extracted_campaign_generation.py`` | 1 |
| Plan doc (this file) | ``plans/PR-Campaign-Channel-Legacy-Cleanup.md`` | new |

That's it. The internal fallback at
``campaign_generation.py`` line 437 stays intact.

## Intentional (looks wrong but is deliberate)

- **No field removal.** Removing ``channel`` from the dataclass
  would break any host constructing
  ``CampaignGenerationConfig(channel="...")`` with a
  ``TypeError``. Frozen dataclasses don't accept unknown kwargs.
  We don't know who's still on the legacy field. Removal belongs in
  a versioned breaking-change slice, not a quiet cleanup PR.
- **No ``warnings.warn`` deprecation runtime hook.** Could be added
  but adds noise to test logs and host telemetry without much
  benefit until we're committed to actual removal. The dataclass
  docstring + this plan doc are the audit-trail breadcrumb hosts
  will see.
- **In-tree test migration moves to the supported path.** The one
  test that used ``channel="linkedin"`` now constructs
  ``channels=("linkedin",)``. Same generated behavior; package's
  own tests stop exercising the legacy fallback.
- **The fallback in ``_channels()`` stays.** ``self._config.channels
  or (self._config.channel,)``. That's the contract for hosts on
  the legacy field. Removing it is the actual breaking change.

## Removal path (deferred)

Future breaking-change PR (``PR-Campaign-Config-V2`` or similar):
1. Drop ``channel: str = "email"`` from the dataclass.
2. Drop the ``or (self._config.channel,)`` fallback in
   ``_channels()``.
3. Bump the package version (semver-major or equivalent host
   coordination).

Out of scope here.

## Verification

- ``pytest tests/test_extracted_campaign_generation.py`` -> all
  passing (the migrated test still asserts the same behavior).
- No need to re-run the full extracted-pipeline suite; the change is
  scoped to a docstring + a single test param.

## Sibling references

- ``CampaignGenerationConfig`` lives in
  ``extracted_content_pipeline/campaign_generation.py``.
- The ``Deferred`` sections of ``plans/PR-OptionA-1.md`` through
  ``plans/PR-OptionA-5.md`` and
  ``plans/PR-ContentAssets-Consistency-2.md`` all listed this
  cleanup as pending.
