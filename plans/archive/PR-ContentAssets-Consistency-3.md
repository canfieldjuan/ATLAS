# PR-ContentAssets-Consistency-3: migrate 5 more adapters to shared _jsonb_helpers

## Why this slice exists

PR-ContentAssets-Consistency-2 (#380) migrated the three adapters
flagged in PR #356's original review: ``campaign_postgres_review``,
``campaign_postgres_import``, ``podcast_postgres``. Self-audit
during that PR found **five more adapters** with the same byte-
identical ``_jsonb`` / ``_row_dict`` definitions:

- ``campaign_postgres_export.py`` (``_row_dict``)
- ``campaign_postgres_seller_targets.py`` (``_row_dict``)
- ``campaign_postgres_seller_opportunities.py`` (both)
- ``campaign_postgres_seller_category_intelligence.py`` (both)
- ``podcast_postgres_import.py`` (``_jsonb``)

This PR closes that fan-out. After this lands the package has a
single source of truth for these four helpers
(``json_dump_jsonb`` / ``decode_jsonb_field`` / ``parse_command_tag``
/ ``row_to_dict``) -- no remaining local copies in any
``extracted_content_pipeline/*.py`` file.

## Scope (this PR)

Pure refactor / cleanup. No behavior change. Same ``import-as`` shim
pattern as Consistency-1 / -2.

| Adapter | Local helpers replaced | ``import json`` |
|---|---|---|
| ``campaign_postgres_export.py`` | ``_row_dict`` | kept (custom serializer uses ``json.dumps``) |
| ``campaign_postgres_seller_targets.py`` | ``_row_dict`` | n/a (file didn't import json) |
| ``campaign_postgres_seller_opportunities.py`` | both | kept (uses ``json.loads`` / ``JSONDecodeError`` outside helpers) |
| ``campaign_postgres_seller_category_intelligence.py`` | both | dropped (no other json. usage) |
| ``podcast_postgres_import.py`` | ``_jsonb`` | dropped (no other json. usage) |

The ``import-as`` shim:

```python
from .storage._jsonb_helpers import (
    json_dump_jsonb as _jsonb,
    row_to_dict as _row_dict,
)
```

Call sites continue to use the private ``_jsonb(...)`` / ``_row_dict(...)``
names; only the helper source changes. Same backwards-compat
rationale as the prior consistency PRs.

## Intentional (looks wrong but is deliberate)

- **Backwards-compat shim instead of renaming all call sites.**
  Pure imports change. Same shim pattern as Consistency-1 / -2.
- **No new abstraction.** Helpers exist; we're just consolidating
  the adapter end of the wire.
- **No tests added.** Behavior unchanged; existing adapter tests
  lock the contract. Adding "the import comes from _jsonb_helpers"
  tests would test implementation, not contract -- same rationale
  as Consistency-2.
- **Per-file ``import json`` decision documented in the table.**
  Three files retain ``import json`` because they use
  ``json.loads`` / ``json.JSONDecodeError`` / a custom
  ``json.dumps`` serializer outside the shared helpers; two drop
  the import cleanly.

## Deferred (still on purpose)

- ``topic`` for blog_post -- still no service-side landing surface.
- ``channel``/``channels`` legacy dual-field cleanup on
  ``CampaignGenerationConfig`` -- separate slice.
- 9 MINOR + 2 NIT findings from the Content Ops audit -- batch
  cleanup PR.

## Verification

- ``pytest`` on the touched adapter test suites
- ``python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`` -> clean
- ``python scripts/audit_extracted_standalone.py --fail-on-debt`` -> 0
- ``bash scripts/check_ascii_python.sh`` -> passed
- Final grep: ``grep -rn "^def _jsonb\|^def _row_dict"
  extracted_content_pipeline/`` returns zero remaining local
  definitions in any ``*_postgres*.py`` adapter.

## Sibling references

- PR-ContentAssets-Consistency-1 plan:
  ``plans/PR-ContentAssets-Consistency-1.md``
- PR-ContentAssets-Consistency-2 plan:
  ``plans/PR-ContentAssets-Consistency-2.md``
- Shared helpers source:
  ``extracted_content_pipeline/storage/_jsonb_helpers.py``
