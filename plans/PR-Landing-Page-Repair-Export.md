# PR-Landing-Page-Repair-Export

## Why this slice exists

Landing page generation now records quality repair telemetry in draft metadata,
but the host export path still omits that telemetry from the review/export
contract. That leaves operators able to see parse attempts and readiness fields
without seeing whether the landing page passed only after a quality repair.

## Scope (this PR)

1. Add landing page quality repair attempts to exported landing page rows.
2. Add landing page quality repair history to exported landing page rows.
3. Cover JSON and CSV export behavior with focused regression tests.

### Files touched

- `extracted_content_pipeline/landing_page_export.py`
- `tests/test_extracted_landing_page_export.py`
- `plans/PR-Landing-Page-Repair-Export.md`

## Mechanism

The metadata summary helper already flattens generated metadata into review columns.
This slice extends that same summary with:

```python
"generation_quality_repair_attempts": metadata.get("generation_quality_repair_attempts")
"generation_quality_repair_history": metadata.get("generation_quality_repair_history")
```

The export column list adds both fields beside the existing generation attempt
metadata. CSV rendering already serializes list and mapping values through the
shared `_csv_value()` helper, so nested repair history can be exported without a
new serializer.

## Intentional

- No generation changes: the telemetry is already written by the landing page
  generator.
- No UI changes: generated asset review surfaces were handled by earlier slices.
- The export keeps missing metadata as `None`, matching existing token and parse
  attempt export behavior.

## Deferred

- `PR-Landing-Page-Repair-Analytics` can decide whether these export fields need
  aggregate reporting across generated assets.

## Verification

- `pytest tests/test_extracted_landing_page_export.py -q` - 8 passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Export contract | ~8 |
| Export tests | ~46 |
| Plan doc | ~58 |
| **Total** | **~112** |
