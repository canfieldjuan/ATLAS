# PR: Content Ops Intervention Reasoning Provider

## Why this slice exists

The AI Content Ops deferred backlog now has the live persistence smoke and blog
blueprint ingestion path closed. The next P1 gap is the reasoning bridge:
file-backed and DB-backed reasoning providers work, but Atlas intervention output
is still not consumable by Content Ops unless another process copies it into a
campaign reasoning file or table.

This PR adds the smallest host-side bridge from persisted Atlas intervention
reports into the existing `CampaignReasoningContextProvider` port. It keeps the
extracted package decoupled from Atlas and makes the provider opt-in so existing
installs keep their current DB/file behavior.

## Scope

1. Add an opt-in intervention-report provider factory in
   `atlas_brain/_content_ops_reasoning.py`.
2. Read the latest `intelligence_reports` row with `report_type='intervention'`
   matching the request's target/company/vendor selectors.
3. Normalize that row into the existing campaign reasoning context shape.
4. Extend provider selection/status to include `intervention` between DB and
   file fallback.
5. Update tests and coordination docs for the new slice.

### Files touched

- `plans/PR-Content-Ops-Intervention-Reasoning-Provider.md`
- `docs/extraction/coordination/inflight.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `atlas_brain/_content_ops_reasoning.py`
- `tests/test_atlas_content_ops_reasoning.py`

## Mechanism

Operators opt in with `ATLAS_CONTENT_OPS_REASONING_INTERVENTION_ENABLED=true`.
The factory binds to the host asyncpg pool, like the existing DB provider, and
returns a lightweight provider object. On each `read_campaign_reasoning_context`
call it builds the same target/company/vendor selector set used by the file and
DB providers, finds the newest matching intervention report by lower-cased
`entity_name`, and converts `report_text`, `structured_data`, and
`pressure_snapshot` into canonical reasoning, top theses, proof points, reference
ids, and scope summary.

Provider chooser priority becomes `DB > intervention > file > none`: explicit
campaign reasoning rows stay authoritative, intervention output becomes the
Atlas-native automatic fallback, and file remains the staging/manual fallback.

## Intentional

- This is host-side only. `extracted_content_pipeline` does not import Atlas or
  know about intervention reports.
- The provider is opt-in. Existing installs with only DB/file providers keep the
  same behavior.
- The provider reads `intelligence_reports`, which is currently global Atlas
  storage. This is suitable for Atlas-hosted internal deployments. Tenant-owned
  intervention storage remains a later hardening slice if the table gets an
  `account_id` or equivalent ownership column.
- It does not run the intervention pipeline. It only consumes already-persisted
  reports.

## Deferred

- Tenant-owned intervention storage / account-scoped `intelligence_reports`.
- Admin UI for choosing which intervention report feeds a Content Ops run.
- Writing intervention outputs into `campaign_reasoning_contexts` as durable
  per-account rows.
- Richer parsing of stage text into structured content recommendations.

## Verification

- Focused reasoning tests passed: 26 tests.
- Python compile check passed for the touched production and test modules.
- Diff whitespace check passed.

## Estimated diff size

| Area | Estimated LOC |
| --- | ---: |
| Production provider and chooser updates | ~300 |
| Tests | ~170 |
| Plan and coordination docs | ~130 |
| **Total** | **~600** |

This is over the soft 400 LOC target because the provider is a new host adapter
plus tests for factory behavior, selector reads, chooser priority, and status
reporting. The production change stays in one existing host seam; no extracted
package code is touched.
