# PR: Content Ops Admin Cost Labels

## Why this slice exists

Content Ops LLM traces now carry asset, source, account, and cache metadata, and
the Content Ops screen can show usage and cache diagnostics. The broader admin
cost feed still renders those calls as generic `content_ops.llm.complete` rows,
which makes the cross-product cost dashboard harder to scan when Content Ops
traffic is mixed with enrichment, reasoning, and blog-generation traffic.

This slice adds a narrow source-side labeling layer for Content Ops rows in the
admin-cost recent-call serializer. It does not create a new dashboard or read
path; it makes the existing recent-cost surface use the metadata already being
recorded.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Teach the admin-cost recent-call description helper to recognize Content Ops
   LLM rows by span name or `metadata.product`.
2. Render a readable title such as `Content Ops Landing Page` from
   `metadata.asset_type`.
3. Include source/cache detail text from existing trace metadata.
4. Surface Content Ops asset/source/cache fields as structured recent-call
   fields for future UI use.
5. Add a focused admin-cost route test for Content Ops recent-call labeling and
   filtering.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Admin-Cost-Labels.md` | Plan doc for the admin cost labeling slice. |
| `atlas_brain/api/admin_costs.py` | Add Content Ops recent-call labeling and structured metadata fields. |
| `tests/test_admin_costs.py` | Cover Content Ops recent-call labeling through the hosted admin route. |

## Mechanism

`_describe_recent_call(...)` gets a Content Ops branch ahead of the generic
pipeline fallback. It recognizes the canonical `content_ops.llm.complete` span
or `metadata.product == content_ops`, then builds a title from `asset_type` and
a compact detail string from `source_material_type`, `cache_mode`,
`cache_reason`, and `cache_result`.

`_serialize_recent_llm_call(...)` also exposes those same metadata values as
nullable fields. This keeps the backend as the single place that understands
Content Ops trace metadata, while the UI can remain a straightforward renderer.

## Intentional

- This only labels recent-call rows. It does not change cost aggregation,
  billing math, cache policy, or Content Ops usage-summary routes.
- This does not add a new admin endpoint. The existing `/admin/costs/recent`
  route already has the filters operators need.
- This does not parse prompts or generated content. It uses trace metadata
  only.

## Deferred

- Future PR: add Content Ops-specific rollups to `/admin/costs/cache-health` if
  operators need cross-product cache summaries outside the Content Ops screen.
- Future PR: render the new structured fields in the Intel UI admin-cost page
  if the current title/detail fields are not enough.
- Parked hardening: none. Root `HARDENING.md` has no active
  cost-surfacing parked items; `ATLAS-HARDENING.md` contains blog/deep-dive
  content items outside this lane.

## Verification

- python -m pytest tests/test_admin_costs.py::test_recent_calls_labels_content_ops_trace_metadata tests/test_admin_costs.py::test_recent_calls_returns_granular_cache_fields tests/test_admin_costs.py::test_recent_calls_filters_and_surfaces_battle_card_context -q
  — 3 passed, 1 warning.
- python -m compileall -q atlas_brain/api/admin_costs.py tests/test_admin_costs.py
  — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <git-path>/codex-pr-bodies/admin-cost-labels.md
  — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Admin-cost serializer | ~55 |
| Tests | ~45 |
| **Total** | **~180** |

This stays below the 400 LOC soft cap.
