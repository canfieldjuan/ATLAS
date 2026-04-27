# Session log — 2026-04-27

Snapshot of where each branch of work ended so the next session can pick up
without reorientation.

## What landed today

Commits on `main` (newest first):

| Commit | Subject |
|---|---|
| `04dd16c9` | Add Tier 2 model A/B harness with pinned metrics |
| `b3e36499` | Harden Win/Loss Predictor against contract drift |
| `1b7802ee` | Make Win/Loss Predictor gates honest end-to-end |
| `7f5326f0` | Document Phase 9 soak activation runbook |
| `a3f78fc4` | Document Phase 9 canary and fix vendor briefing confidence format |
| `4deb4be0` | Render gated battle-card displacement reasoning section |
| `0e0a0930` | Gate ChallengerBriefDetail head-to-head winner call with ProductClaim |

(`99eab2a0` "Close manual-path scrape dedupe coverage" is concurrent
scrape-pipeline work, not from this session's arc.)

## Branch state at pause

### 1. Tier 2 A/B (Sonnet 4.5 vs Haiku 4.5)
**Status:** Harness shipped, helpers pinned. **Routing change deferred.**

- Harness: `scripts/tier2_model_ab.py` — read-only against production data,
  calls Haiku only, writes markdown rollup. CLI flags: `--sample-size`,
  `--limit`, `--dry-run`, `--review-id`, `--max-cost-usd`, `--output`.
- Tests: `tests/test_tier2_model_ab.py` — 22 sanity tests pinning enum
  sets, promotion counting, attribution proxy, Jaccard.
- Production model is `anthropic/claude-sonnet-4-5` per `.env`
  (`ATLAS_B2B_CHURN_ENRICHMENT_TIER2_OPENROUTER_MODEL`).
- Recommended invocation sequence (operator-run):
  ```
  python scripts/tier2_model_ab.py --dry-run --limit 5
  python scripts/tier2_model_ab.py --review-id <UUID> --max-cost-usd 0.05
  python scripts/tier2_model_ab.py --sample-size 100 --max-cost-usd 1.00
  ```
- Decision gate: read the markdown rollup. If Haiku matches Sonnet on
  JSON validity, enum completeness, and promotion conservatism while
  running materially cheaper, a follow-up patch can flip
  `enrichment_tier2_openrouter_model` to Haiku. Otherwise stay on Sonnet.

### 2. Phase 9 EvidenceClaim contract + shadow-mode validation
**Status:** Soft-closed. Soak waiting on May 4 batch + May 5 verification.

- Scoped canary passed for ClickUp / Pipedrive / Monday.com (DM churn and
  price complaint paths agree, both suppressed; direct evidence remains 0
  across all 6 comparisons).
- Full Monday batch did not populate broad shadow rows because there were
  0 active scheduled competitive sets; activation SQL is documented in
  `docs/progress/product_claim_contract_plan_2026-04-26.md` under "Soak
  activation runbook".
- **Next operational trigger:** May 5, 2026 morning verification. The
  runbook captures the exact commands and pass criteria; whoever picks
  it up should read that section first.
- Lineage flag stays parked globally until coverage is proven.

### 3. Patch 6 chain (ProductClaim contract through reports/UI)
**Status:** Closed except 6a2c.

- 6a1, 6a2a, 6a2b, 6a3a, 6a3b all shipped and audited.
- 6a2c (`incumbent_strengths` ProductClaim gate) is parked because it
  needs a VENDOR-scope `strength_theme` aggregator first. The gate logic
  itself is small (mirrors 6a2b); the substrate is the work. This is the
  remaining "code thrust" candidate.

### 4. Win/Loss Predictor v2
**Status:** Production-honest end-to-end. No open items.

- Step 1 (data integrity gates): closed by `1b7802ee` — nullable
  probability persistence, DB invariant
  `CHECK (is_gated OR win_probability IS NOT NULL)` in migration 306,
  compare/recent/get/export all honor null.
- Steps 2-3 contract drift: closed by `b3e36499` — LLM strategy output
  passes through `_coerce_strategy_output()` before reaching
  `WinLossResponse`; calibration loader fails soft to static weights
  with a warning; compare UI fails closed when `is_gated` is missing.
- Steps 4-6 (persistence / compare / export): inherit Step 1's defenses.
- 38 backend + 4 UI tests passing across both commits.

## Concurrent work (not from this session)

The worktree has uncommitted scrape-pipeline edits unrelated to the above:

```
M atlas_brain/api/b2b_scrape.py
M atlas_brain/autonomous/tasks/b2b_scrape_intake.py
M atlas_brain/services/scraping/parsers/g2.py
M atlas_brain/services/scraping/parsers/peerspot.py
M atlas_brain/services/scraping/parsers/trustpilot.py
```

These are owned by a separate concurrent thread; this session did not
touch them. (Note: `99eab2a0` "Close manual-path scrape dedupe coverage"
landed on main during this session but came from that other thread.)

## Operational triggers to watch

1. **May 5, 2026 morning** — Phase 9 soak verification per the runbook in
   `product_claim_contract_plan_2026-04-26.md`. Lineage flag decision
   gates on this.
2. **Whenever the Tier 2 A/B harness is run** — markdown lands at
   `docs/progress/tier2_model_ab_<run-date>.md`. Routing decision belongs
   to whoever reads it.

## Next-thrust menu

Two real candidates when picking work back up:

- **6a2c substrate.** Real code thrust. Self-contained scope: build a
  VENDOR-scope `strength_theme` aggregator (likely mirroring the 6a2a
  shape — incumbent-side aggregator returning a list of typed claim
  rows), then the gate is a small follow-on patch on
  `b2b_battle_cards.py:_apply_battle_card_displacement_product_claim_gate`
  extended to a third field (`incumbent_strengths`). Closes the last open
  item from Patch 6.
- **B2B Churn product-readiness audit.** Audit pass over the b2b churn
  surface looking for blockers to sale: multi-tenant auth boundaries,
  plan-gate enforcement, observability, error-handling at API edges.
  Now that Win/Loss Predictor is honest, this is the natural extension
  of the "cannot be sold as a prototype" framing.

Operator's lean at pause: product-readiness audit, because blockers
found now save more time than blockers found after more feature surface
lands on top. 6a2c is good code work but the gap it closes is narrower.

## Pause-point hygiene

- All session work committed. No half-finished slice.
- Plan doc (`product_claim_contract_plan_2026-04-26.md`) is current.
- Runbook for May 5 soak verification is captured and discoverable.
- Tier 2 harness is ready; routing change is intentionally pending the
  read.
