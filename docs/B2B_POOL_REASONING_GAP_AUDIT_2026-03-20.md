# B2B Pool Reasoning Gap Audit (2026-03-20)

## Scope

Verify the six-pool reasoning claims against:

- Runtime code paths (pool build, compression, synthesis contracts, downstream consumers)
- Live DB snapshots (latest category dynamics, latest synthesis contracts, latest account pool rows)

## Verification Results

### 1) Evidence Vault pool

Status: Confirmed (strongest wired pool)

Evidence:

- Evidence vault is loaded as one of the six synthesis layers: `atlas_brain/autonomous/tasks/_b2b_shared.py:3150`
- Evidence vault scoring includes weakness/strength items and metric/provenance aggregates: `atlas_brain/autonomous/tasks/_b2b_pool_compression.py:193`
- Synthesis always requires `causal_narrative` (core wedge section): `atlas_brain/autonomous/tasks/_b2b_synthesis_validation.py:36`

Notes:

- No immediate structural gap found in code wiring.

---

### 2) Segment pool

Status: Confirmed (partial contract coverage, underused fields)

Evidence:

- Segment builder produces rich fields (`affected_roles`, `affected_departments`, `affected_company_sizes`, `budget_pressure`, `contract_segments`, `usage_duration_segments`, `top_use_cases_under_pressure`, `buying_stage_distribution`): `atlas_brain/autonomous/tasks/_b2b_shared.py:5113`
- Compression scores only roles/departments/contracts and emits only two segment aggregates (`dm_churn_rate`, `price_increase_rate`): `atlas_brain/autonomous/tasks/_b2b_pool_compression.py:308`
- `usage_duration_segments`, `top_use_cases_under_pressure`, `affected_company_sizes`, and `buying_stage_distribution` are not scored into packet items in `_score_segment`: `atlas_brain/autonomous/tasks/_b2b_pool_compression.py:308`
- Downstream persisted contracts only keep what LLM outputs in `segment_playbook` (no deterministic carry-through for all segment structures): `atlas_brain/autonomous/tasks/_b2b_reasoning_contracts.py:132`

Correction to original claim:

- Contract segments do reach synthesis input via `_score_segment`; usage-duration distribution does not.

---

### 3) Temporal pool

Status: Confirmed

Evidence:

- Temporal builder collects `sentiment_trajectory`: `atlas_brain/autonomous/tasks/_b2b_shared.py:5301`
- Temporal compression scores timeline summary, keyword spikes, deadlines, but does not score `sentiment_trajectory`: `atlas_brain/autonomous/tasks/_b2b_pool_compression.py:426`

---

### 4) Displacement pool

Status: Confirmed, broader than stated

Evidence:

- Synthesis pool loader reads canonical displacement dynamics table (`b2b_displacement_dynamics`): `atlas_brain/autonomous/tasks/_b2b_shared.py:3188`
- Battle cards still fetch displacement via legacy review-level extractor: `atlas_brain/autonomous/tasks/b2b_battle_cards.py:531`
- Churn reports also fetch the same legacy displacement extractor: `atlas_brain/autonomous/tasks/b2b_churn_reports.py:241`
- Accounts in motion also fetch the same legacy displacement extractor: `atlas_brain/autonomous/tasks/b2b_accounts_in_motion.py:897`
- Legacy extractor definition (from `b2b_reviews` competitors JSON): `atlas_brain/autonomous/tasks/_b2b_shared.py:2256`

Conclusion:

- This is not only a battle-card issue; at least 3 downstream consumers currently run a parallel displacement materialization path.

---

### 5) Category pool

Status: Partially confirmed (thin + frequently missing in contracts; root cause is mixed)

Code evidence:

- Category builder shape is intentionally narrow (`market_regime`, `council_summary`, counts): `atlas_brain/autonomous/tasks/_b2b_shared.py:5456`
- Category compression scores regime and council summary only: `atlas_brain/autonomous/tasks/_b2b_pool_compression.py:557`
- Contract synthesis treats category as optional when explicit contracts are present: `atlas_brain/autonomous/tasks/_b2b_reasoning_contracts.py:229`
- Validation required sections do not include category reasoning: `atlas_brain/autonomous/tasks/_b2b_synthesis_validation.py:36`
- Battle-card deterministic `category_council` falls back to ecosystem context, not category pool contract: `atlas_brain/autonomous/tasks/b2b_battle_cards.py:247`
- Deterministic competitive landscape uses `category_council` field, not `category_reasoning` directly: `atlas_brain/autonomous/tasks/_b2b_shared.py:960`

Live DB evidence (latest snapshot):

- Latest categories: 13
- `council_summary` null: 0
- `council_summary.conclusion` blank: 13
- Latest synthesis vendors: 55
- Vendors with `category_reasoning`: 37
- Vendors missing/blank category regime+narrative: 18

Interpretation:

- "Category reasoning often empty/thin" is true in output.
- It is not only because council rows are null; category contract is optional in validation/materialization and often omitted by synthesis.

---

### 6) Accounts pool

Status: Confirmed

Evidence:

- Canonical account pool is generated and persisted in core run: `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py:2452`, `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py:2469`
- Six-pool loader includes `b2b_account_intelligence`: `atlas_brain/autonomous/tasks/_b2b_shared.py:3154`
- Accounts-in-motion task does not read `b2b_account_intelligence`; it rebuilds from high-intent + timeline + company signals + quotes and scores locally: `atlas_brain/autonomous/tasks/b2b_accounts_in_motion.py:833`
- `b2b_account_intelligence` is otherwise only referenced by pool loader and core insert path: `atlas_brain/autonomous/tasks/_b2b_shared.py:3154`, `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py:2469`

Live DB evidence (latest snapshot):

- Account pool vendors: 55
- `summary.total_accounts` aggregate: 86
- `summary.active_eval_signal_count` aggregate: 35

Interpretation:

- Per-account reasoning is not sourced from the account pool in accounts-in-motion.
- The pool exists and is populated, but downstream path currently bypasses it.

---

### 7) "Only synthesis reads all six pools directly"

Status: Confirmed

Evidence:

- `fetch_all_pool_layers` definition: `atlas_brain/autonomous/tasks/_b2b_shared.py:3139`
- Only call site: `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py:60`

## Logged Issues (Prioritized)

### P1 - Dual displacement materialization across consumers

- Keep one canonical displacement source for consumers (prefer synthesis contracts sourced from `b2b_displacement_dynamics`).
- Current risk: conflicting displacement narratives between synthesis and deterministic reports/cards.

### P1 - Accounts-in-motion bypasses canonical account pool

- Rebuild logic in accounts-in-motion diverges from `b2b_account_intelligence`.
- Current risk: duplicate scoring logic and inconsistent account sets.

### P1 - Category contract optional and unstable in downstream battle context

- `category_reasoning` is not required by synthesis validation and is often missing.
- Deterministic battle card category context uses cross-vendor/ecosystem fallback path instead of category pool contract.

### P2 - Segment compression drops several rich segment fields

- `usage_duration_segments`, `affected_company_sizes`, `top_use_cases_under_pressure`, `buying_stage_distribution` do not become scored pool items.

### P2 - Temporal sentiment not scored into synthesis packet

- `sentiment_trajectory` exists in temporal pool but is not part of temporal scoring/aggregates.

### P2 - Consumer-level pool fragmentation

- Downstream consumers use mixed source paths; synthesis is the only all-pool integrator.

## Initial Fix Plan

### Phase A: Source-of-truth alignment (P1)

1. Displacement
   - Add a shared reader utility for displacement from synthesis contracts (`displacement_reasoning` first, then fallback to dynamics row if needed).
   - Update battle cards, churn reports, accounts-in-motion to use shared displacement reader before legacy `_fetch_competitive_displacement`.

2. Accounts
   - Add a shared account reader that loads latest `b2b_account_intelligence` per vendor.
   - Refactor accounts-in-motion to start from account pool records, then apply enrichment overlays (Apollo, quotes, timeline, quality penalties).

3. Category
   - Make `category_reasoning` a required synthesis contract section (with empty-safe values allowed but section must exist).
   - In battle cards, drive deterministic category context from `reasoning_contracts.category_reasoning` before cross-vendor/ecosystem fallback.

### Phase B: Compression completeness (P2)

4. Segment compression
   - Extend `_score_segment` to emit scored items for:
     - `usage_duration_segments`
     - `affected_company_sizes` (distribution details where available)
     - `top_use_cases_under_pressure`
     - `buying_stage_distribution`

5. Temporal compression
   - Extend `_score_temporal` to include `sentiment_trajectory` as scored/aggregate signals.

### Phase C: Integration hardening

6. Contract-gap observability
   - Add metric counters for `reasoning_contract_gaps` by consumer and contract name.
   - Add report-level logging when consumers fall back to non-contract data.

7. Validation + tests
   - Add tests to enforce:
     - category contract presence
     - displacement reader parity across consumers
     - accounts-in-motion starting from account pool data
     - segment/temporal new fields present in compressed payload and traceable citations

