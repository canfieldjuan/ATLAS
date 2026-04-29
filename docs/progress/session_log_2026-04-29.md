# Session log — 2026-04-29

Snapshot of where each branch of work ended so the next session can pick up
without reorientation. Continues from `session_log_2026-04-27.md`.

## What landed this session

Single audit-chain commit:

| Commit | Subject |
|---|---|
| `d682561d` | Close B2B access and ProductClaim action gates |

15 files / +1290 / -43 — covers the access-boundary slice, action/export
inheritance slice, campaign action gates, and the M1+L2 tightening.

(Adjacent commits `6d1fd5f7` "Phase 1 progress: eligibility module +
verification harness" and uncommitted scrape/enrichment edits are concurrent
scrape-pipeline work, not from this session's audit chain.)

## Branch state at pause

### 1. Systemic issue audit (NEW this session)
**Status:** 20 SYS items logged + 5 deep dives + first 2 Highs closed end-to-end.

- Audit log: `docs/progress/systemic_issue_investigation_log_2026-04-28.md`
  (committed). Tracks SYS-001 through SYS-020 across tenant boundary, action
  gates, renderer contracts, autonomous liveness, and schema contracts.
- Five deep dives capture next-fix shape: Access Boundary, Action/Export
  Inheritance, Report Renderer Contract, Autonomous Liveness, Schema
  Contracts.
- Prioritization buckets:
  1. Access boundary sweep — *partially closed* (SYS-002, 005-operator-routes,
     006, 007 done via Patch A/B1; SYS-001, 003, 016, 017 still queued)
  2. Action/export inheritance — *closed* (SYS-004, SYS-008)
  3. Report/UI contract completion — *queued* (SYS-009, 010, 013, 018)
  4. Autonomous liveness — *queued* (SYS-019)
  5. Schema contracts — *queued* (SYS-020)
  6. Claim substrate hardening — *queued* (SYS-011, 012, 014, 015)

### 2. Access boundary (Patch A/B1)
**Status:** Closed. Boundary harness pinned 20 routes.

- 4 routers converted: `vendor_targets` (CRUD + report-generation),
  `b2b_vendor_briefing` (preview/generate/send-batch/list), `admin_costs`
  (router-level), `seller_campaigns` (router-level).
- New `tests/test_b2b_access_boundaries.py`: 20 routes parametrized over
  `/api/v1/...` paths, asserting `401 + "Authentication required"` AND
  `db touched before auth` AssertionError fires (proves auth-before-DB
  ordering).
- Public briefing routes (`gate`, `checkout`, `checkout-session`,
  `report-data`) intentionally unauthenticated.

### 3. Action/export inheritance (SYS-004 + SYS-008)
**Status:** Closed end-to-end across 5 surfaces with conflict-bypass tightened.

- **Fetch:** `_fetch_opportunities` and `_fetch_accounts_in_motion_opportunities`
  filter on `claim.report_allowed` before LLM call. `force=True` does NOT
  bypass.
- **Replay:** `_store_replayed_campaign_entry` raises ValueError before DB
  touch when the payload lacks a report-safe gate.
- **Persist:** Campaign metadata stores `opportunity_claim` (full),
  `opportunity_claim_gate` (compact), and `opportunity_claims[:20]`
  (aggregate) for audit.
- **Approve / queue / bulk:** `_enforce_campaign_product_claim_gate` raises
  409 on all 4 paths before status changes. Tests assert `pool.execute_calls
  == []`.
- **CSV export:** High-intent CSV defaults to `report_safe_only=true`;
  tenant alias forwards the flag. Campaign CSV includes 7 claim columns
  (intentionally exports all campaigns regardless of gate state — documented
  asymmetry, see L1 below).
- **Conflict-bypass tightened (M1):** `_campaign_payload_has_report_safe_product_claim`
  and `_campaign_metadata_has_report_safe_product_claim` now require all
  present claim shapes (`opportunity_claim`, `opportunity_claims`,
  `opportunity_claim_gate`) to agree on report-safe. No short-circuit hole.
- **Documented residuals:**
  - **L1**: Campaign CSV asymmetric default vs high-intent CSV. Accepted as
    intentional policy because upstream approval/queue gates already enforce
    the contract on the campaign lifecycle.
  - **M3**: Hardcoded `analysis_window_days=90` in campaign filter. Parked
    pending broader API change to expose `as_of_date` / window override.

### 4. Tier 2 A/B (carried from 2026-04-27)
**Status:** Unchanged. Harness still ready to run. Routing change deferred
until the markdown readout.

### 5. Phase 9 Step 7
**Status:** Soak waiting on May 4 batch + May 5 verification per
`docs/progress/product_claim_contract_plan_2026-04-26.md`. Lineage flag
still parked globally.

### 6. Patch 6 chain
**Status:** Unchanged. Closed except 6a2c (incumbent_strengths gate, parked
pending VENDOR-scope `strength_theme` substrate).

### 7. Win/Loss Predictor v2
**Status:** Unchanged. Production-honest end-to-end across all 6 steps.

## Concurrent work (not from this session)

The worktree has uncommitted scrape/enrichment edits owned by a separate
concurrent thread:

```
M atlas_brain/autonomous/tasks/b2b_enrichment.py
M atlas_brain/services/scraping/parsers/reddit.py
M tests/test_b2b_enrichment.py
?? atlas_brain/services/b2b/enrichment_stage_runs.py
?? atlas_brain/storage/migrations/282_b2b_enrichment_stage_runs.sql
?? atlas_brain/storage/migrations/283_b2b_enrichment_stage_runs_work_fingerprint.sql
```

`6d1fd5f7` "Phase 1 progress: eligibility module + verification harness"
also landed on main during this session but came from that other thread.

## Operational triggers to watch

1. **May 4, 2026 evening (~21:15 scheduler-TZ)** — Phase 9 soak runs against
   the 7 activated competitive sets per the runbook in
   `product_claim_contract_plan_2026-04-26.md`.
2. **May 5, 2026 morning** — Soak verification commands per the same runbook.
   Lineage flag decision gates on this.
3. **Whenever the Tier 2 A/B harness is run** — markdown lands at
   `docs/progress/tier2_model_ab_<run-date>.md`. Routing decision belongs
   to whoever reads it.

## Next-thrust menu

In priority order from the audit log's prioritization (lines 303-310):

1. **SYS-018 + SYS-009 — renderer contract completion (UI thrust).**
   `DangerousStructuredFieldGate` over `cross_vendor_battles`,
   `weakness_analysis`, `vendor_weaknesses`, `recommended_plays` per the
   Report Renderer Contract Deep Dive. Broadest remaining UI bypass; pure
   code work, fix shape laid out.
2. **SYS-005 — briefing public/admin router split.** Operator-flow vs
   sales-flow partition. Patch A/B1 converted the operator endpoints to
   `require_auth`, but the routing structure still mixes public and
   operator routes in one router. Design-heavier.
3. **SYS-001 — `b2b_dashboard.py` 70 `optional_auth` endpoints.** Largest
   remaining boundary blast radius. Parked pending public/demo vs tenant
   classification — needs product-policy decision before code.
4. **SYS-003 — CRM webhook auth split.** Webhook signing design + operator
   query auth. Design-heavy.
5. **SYS-019 — autonomous liveness audit.** Domain-queue eligibility
   diagnostic for `b2b_reasoning_synthesis` and stuck-running cleanup.
6. **SYS-020 — schema contracts.** Promote evidence/opportunity wrappers
   to Pydantic response models with required gate fields.

Operator's lean from the 2026-04-28 closeout: SYS-018+SYS-009 next, because
it's the broadest UI bypass with a pre-defined fix shape and same
audit-loop rhythm.

## Pause-point hygiene

- All session work committed in `d682561d`. No half-finished slice in this
  audit chain.
- Audit log (`systemic_issue_investigation_log_2026-04-28.md`) is current
  and committed; SYS-004 / SYS-008 rows reflect closure across all surfaces.
- Boundary harness (20 routes) protects against drift on all 4 converted
  routers + the 6 already-protected routes.
- Documented residuals (L1 export asymmetry, M3 hardcoded window) live in
  this session log + the audit log; not blocking.
- Phase 9 May 4–5 trigger is the next standing operational deadline.
