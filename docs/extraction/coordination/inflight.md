# In-Flight PRs

Last updated: 2026-05-04T01:10Z by claude-2026-05-03-b

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1h, in flight) | PR-C1h: Route `extracted_content_pipeline/reasoning/archetypes.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/archetypes.py` (drop ~590-line drifted fork; replace with thin re-export wrapper from `extracted_reasoning_core.archetypes`). Existing `tests/test_extracted_reasoning_archetypes.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/archetypes.py`; `tests/test_extracted_reasoning_archetypes.py` |
| (PR-B4a, in flight) | PR-B4a: Blog quality pack (deterministic core + Atlas adapter) | NEW: `extracted_quality_gate/blog_pack.py` (pure `evaluate_blog_post`). EDIT: `extracted_quality_gate/{__init__.py, manifest.json, README.md, STATUS.md}`. EDIT: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` (`_apply_blog_quality_gate` delegates to pack; helper `_required_vendor_names` added). NEW: `tests/test_extracted_quality_gate_blog_pack.py` (30 tests). | claude-2026-05-03-b | `extracted_quality_gate/blog_pack.py`; `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`; the new blog-pack test file |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
