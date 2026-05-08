# PR-ContentAssets-Consistency-2: migrate 3 remaining adapters to shared _jsonb_helpers

## Why this slice exists

Owed since PR #356. PR-ContentAssets-Consistency-1 extracted shared
JSONB / asyncpg helpers into
``extracted_content_pipeline/storage/_jsonb_helpers.py`` and migrated
the four content-asset draft adapters (campaigns, reports, landing
pages, sales briefs). PR #356's review caught that three more
adapters in the package have **byte-identical** ``_jsonb`` /
``_row_dict`` definitions and weren't migrated:

- ``extracted_content_pipeline/campaign_postgres_review.py``
- ``extracted_content_pipeline/campaign_postgres_import.py``
- ``extracted_content_pipeline/podcast_postgres.py``

PR #356's plan doc explicitly listed this as the Consistency-2
follow-up. This PR closes it.

## Scope (this PR)

Pure refactor / cleanup. No behavior change. No new abstraction --
just import the existing shared helpers in place of the local
copy-pastes.

| Adapter | Local helpers replaced |
|---|---|
| ``campaign_postgres_review.py`` | ``_jsonb``, ``_row_dict`` |
| ``campaign_postgres_import.py`` | ``_jsonb`` (no ``_row_dict``) |
| ``podcast_postgres.py`` | ``_jsonb``, ``_row_dict`` |

Shim pattern matches ``campaign_postgres.py`` from PR #356:

```python
from .storage._jsonb_helpers import (
    json_dump_jsonb as _jsonb,
    row_to_dict as _row_dict,
)
```

The aliased import lets call sites continue to use the private
``_jsonb(...)`` / ``_row_dict(...)`` names without touching every
call site -- pure imports change. Backward-compat shim for any
in-tree caller that imported the private names directly (same
rationale as #356's choice for the campaign adapter).

## Intentional (looks wrong but is deliberate)

- **Backwards-compat shim instead of renaming all call sites.** The
  private helper names (``_jsonb``, ``_row_dict``) were referenced
  inside the adapters; renaming every call site would balloon the
  diff with mechanical changes that don't add value. The
  ``import-as`` shim is zero-runtime-cost and keeps the diff focused
  on the helper-source migration.
- **No new abstraction** -- this is the consolidation slice for the
  existing helpers, not a place to introduce a new pattern.
- **No tests added.** The adapters' behavior is unchanged (same
  helpers, same call sites). Existing tests for these three
  adapters lock the behavior; passing those is sufficient
  verification. Adding tests that assert "the import comes from
  ``_jsonb_helpers``" would test the implementation, not the
  contract.
- **``import json`` stays in ``podcast_postgres.py``** because it
  uses ``json.loads`` directly outside the helper functions. The
  other two adapters drop ``import json`` since the helpers now
  cover all JSON needs. (Verified by grep before the edit.)

## Deferred (still on purpose)

- **Five more adapters with the same duplication.** Self-audit
  during this PR found additional local ``_jsonb`` / ``_row_dict``
  definitions in:
  - ``campaign_postgres_export.py`` (``_row_dict``)
  - ``campaign_postgres_seller_targets.py`` (``_row_dict``)
  - ``campaign_postgres_seller_opportunities.py`` (both)
  - ``campaign_postgres_seller_category_intelligence.py`` (both)
  - ``podcast_postgres_import.py`` (``_jsonb``)

  The original PR #356 review only flagged three; the actual
  duplication landscape is wider. Not migrated here to keep this
  slice scoped to what the original review-trail promised. Follow-up
  ``PR-ContentAssets-Consistency-3`` migrates these five in one
  atomic move.
- ``topic`` for blog_post -- still no service-side landing surface.
- ``channel``/``channels`` legacy dual-field cleanup on
  ``CampaignGenerationConfig`` -- separate slice.
- 9 MINOR + 2 NIT findings from the Content Ops audit -- batch
  cleanup PR.

## Verification

- ``pytest`` on the three touched adapter test suites
- ``python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`` -> clean
- ``python scripts/audit_extracted_standalone.py --fail-on-debt`` -> 0
- ``bash scripts/check_ascii_python.sh`` -> passed

## Sibling references

- PR #356 plan: ``plans/PR-ContentAssets-Consistency-1.md``
- Shared helpers source:
  ``extracted_content_pipeline/storage/_jsonb_helpers.py``
