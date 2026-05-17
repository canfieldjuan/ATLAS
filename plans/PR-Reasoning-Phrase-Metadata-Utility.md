# Reasoning Phrase Metadata Utility

## Why this slice exists

PR #564 split Atlas-owned review enrichment into
`atlas_brain.reasoning.review_enrichment`, but that pack still imported
phrase metadata helpers from `atlas_brain.autonomous.tasks`. That kept a deep
reasoning-to-task dependency in the new reasoning boundary.

The helpers are pure readers over enrichment dictionaries. They do not depend
on task orchestration, storage, LLM calls, or Atlas autonomous task state. This
slice promotes them into `atlas_brain.reasoning.phrase_metadata` and leaves the
old task module as a compatibility wrapper.

## Scope

1. Add `atlas_brain.reasoning.phrase_metadata` as the canonical helper module.
2. Convert `atlas_brain.autonomous.tasks._b2b_phrase_metadata` into a re-export
   wrapper so existing imports keep working.
3. Redirect reasoning and B2B service imports to the canonical reasoning
   module.
4. Update tests and docs that pin the old helper location.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `plans/PR-Reasoning-Phrase-Metadata-Utility.md`
- `atlas_brain/reasoning/phrase_metadata.py`
- `atlas_brain/reasoning/review_enrichment.py`
- `atlas_brain/reasoning/evidence_engine.py`
- `atlas_brain/autonomous/tasks/_b2b_phrase_metadata.py`
- `atlas_brain/autonomous/tasks/_b2b_witnesses.py`
- `atlas_brain/autonomous/tasks/_b2b_field_contracts.py`
- `atlas_brain/autonomous/tasks/b2b_enrichment.py`
- `atlas_brain/services/b2b/enrichment_contract.py`
- `atlas_brain/services/b2b/enrichment_pain_competition.py`
- `tests/test_b2b_phrase_metadata.py`
- `tests/test_atlas_reasoning_evidence_engine_aliases.py`

## Mechanism

`atlas_brain.reasoning.phrase_metadata` becomes the canonical module for
read-only phrase metadata access. The old task-layer module imports and
re-exports the same helpers, so existing callers do not break while reasoning
code can depend on a reasoning-owned utility.

The Atlas review enrichment pack switches its pricing phrase gate to the new
module. B2B service helpers that already expose phrase metadata as a contract
also import from the new module, keeping one canonical implementation.

## Intentional

- No helper behavior changes.
- No public function signature changes.
- No extraction-core API expansion in this slice.
- The old `_b2b_phrase_metadata` import path remains supported for callers that
  have not migrated yet.

## Deferred

- Moving write-time phrase metadata normalization out of B2B enrichment. That
  path still owns task/service-specific enrichment writes.
- Updating every historical doc reference outside the touched reasoning and B2B
  helper boundary.

## Verification

- python -m py_compile on touched Python files - passed.
- pytest tests/test_b2b_phrase_metadata.py
  tests/test_atlas_reasoning_evidence_engine_aliases.py
  tests/test_b2b_phase2_subject_gate.py tests/test_b2b_phase3_polarity_gate.py
  - 90 passed.
- pytest tests/test_b2b_enrichment_contract.py - 42 passed.
- git diff --check - passed.
- bash scripts/local_pr_review.sh - passed after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Coordination docs | ~10 |
| Plan doc | ~70 |
| New reasoning helper module | ~115 |
| Compatibility wrapper trim | ~140 |
| Import/doc/test updates | ~45 |
| **Total** | ~380 |
